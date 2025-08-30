// Triangles → Gaussians (compute)
//
// Bind group / set layout:
//   set(0): inputs (RO storages + optional uniform)  -- declared but not used here
//   set(1): TriToSplatParams (dynamic uniform)      -- used for gaussian_count
//   set(2): planar RW storage for the 3D gaussians  -- written by this pass

//////////////////////////////////////////////////////////////
// set(0) inputs (placeholders; types/usage can be adapted) //
//////////////////////////////////////////////////////////////

@group(0) @binding(0)
var<storage, read> mesh_positions : array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> mesh_indices   : array<u32>;

@group(0) @binding(2)
var<storage, read> mesh_extra     : array<u32>;

struct MeshInParams {
    verts : u32,
    indices : u32,
    tris : u32,
    _pad : u32,
};

@group(0) @binding(3)
var<uniform> IN : MeshInParams;

//////////////////////////////////////
// set(1): dynamic per-view params  //
//////////////////////////////////////

struct Params {
    gaussian_count : u32,
};
@group(1) @binding(0)
var<uniform> P : Params;

///////////////////////////////////////////
// set(2): planar RW gaussian attributes //
///////////////////////////////////////////

// Matches the planar 3D format: position+visibility, rotation (quat),
// scale+opacity, and spherical harmonics planes.
//
// NOTE: spherical harmonics are stored as a flat array of vec4; the common
// layout is 12 vec4 planes per gaussian (192 bytes). Adjust if your fork differs.

@group(2) @binding(0)
var<storage, read_write> position_visibility : array<vec4<f32>>;

@group(2) @binding(1)
var<storage, read_write> spherical_harmonics : array<vec4<f32>>;

@group(2) @binding(2)
var<storage, read_write> rotation_quat       : array<vec4<f32>>;

@group(2) @binding(3)
var<storage, read_write> scale_opacity       : array<vec4<f32>>;

const PLANES_PER_GAUSSIAN : u32 = 12u;

@compute @workgroup_size(256, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let triangle_idx = gid.x;
    if (triangle_idx >= P.gaussian_count) {
        return;
    }

    // Check if we have enough triangles
    if (triangle_idx >= IN.tris) {
        return;
    }

    // Get triangle indices
    let i0 = mesh_indices[triangle_idx * 3u + 0u];
    let i1 = mesh_indices[triangle_idx * 3u + 1u]; 
    let i2 = mesh_indices[triangle_idx * 3u + 2u];

    // Get triangle vertices
    let p0 = mesh_positions[i0].xyz;
    let p1 = mesh_positions[i1].xyz;
    let p2 = mesh_positions[i2].xyz;

    // Compute triangle centroid
    let centroid = (p0 + p1 + p2) / 3.0;

    // Compute triangle basis vectors like CPU version
    let u = p1 - p0;
    let v = p2 - p0;

    // Construct coordinate system
    let x_axis = normalize(u);
    let z_axis = normalize(cross(u, v));
    let y_axis = cross(z_axis, x_axis);

    // Convert to quaternion [w, x, y, z] format expected by bevy_gaussian_splatting
    // Based on the CPU code and glossary notes about quaternion order
    let mat = mat3x3<f32>(x_axis, y_axis, z_axis);
    let quat = mat3_to_quat(mat);

    // Compute scale based on triangle dimensions
    let u_len = length(u);
    let v_on_y = abs(dot(v, y_axis));
    let face_scale = 0.001; // Small thickness for face

    // Position (xyz) + visibility (w)
    position_visibility[triangle_idx] = vec4<f32>(centroid, 1.0);

    // Rotation quaternion in [w,x,y,z] order
    rotation_quat[triangle_idx] = quat;

    // Scale (xyz) + Opacity (w)
    scale_opacity[triangle_idx] = vec4<f32>(u_len, v_on_y, face_scale, 1.0);

    // Write base color term into the first SH plane (simple white)
    let base = triangle_idx * PLANES_PER_GAUSSIAN;
    if (base < arrayLength(&spherical_harmonics)) {
        spherical_harmonics[base] = vec4<f32>(1.0, 1.0, 1.0, 0.0);
    }
}

// Convert 3x3 matrix to quaternion in [w,x,y,z] format
fn mat3_to_quat(m: mat3x3<f32>) -> vec4<f32> {
    let trace = m[0][0] + m[1][1] + m[2][2];
    var quat: vec4<f32>;
    
    if (trace > 0.0) {
        let s = sqrt(trace + 1.0) * 2.0; // s = 4 * qw
        quat.w = 0.25 * s;
        quat.x = (m[2][1] - m[1][2]) / s;
        quat.y = (m[0][2] - m[2][0]) / s;
        quat.z = (m[1][0] - m[0][1]) / s;
    } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
        let s = sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0; // s = 4 * qx
        quat.w = (m[2][1] - m[1][2]) / s;
        quat.x = 0.25 * s;
        quat.y = (m[0][1] + m[1][0]) / s;
        quat.z = (m[0][2] + m[2][0]) / s;
    } else if (m[1][1] > m[2][2]) {
        let s = sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0; // s = 4 * qy
        quat.w = (m[0][2] - m[2][0]) / s;
        quat.x = (m[0][1] + m[1][0]) / s;
        quat.y = 0.25 * s;
        quat.z = (m[1][2] + m[2][1]) / s;
    } else {
        let s = sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0; // s = 4 * qz
        quat.w = (m[1][0] - m[0][1]) / s;
        quat.x = (m[0][2] + m[2][0]) / s;
        quat.y = (m[1][2] + m[2][1]) / s;
        quat.z = 0.25 * s;
    }
    
    return quat;
}
