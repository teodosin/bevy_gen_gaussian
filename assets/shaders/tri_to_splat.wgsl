// Spherical harmonics data for a single Gaussian.
// This must match the layout expected by bevy_gaussian_splatting: array<f32, 48>
// where 48 = SH_COEFF_COUNT (16 coefficients per channel * 3 channels)
struct SphericalHarmonic {
    coefficients: array<f32, 48>,
}

// Input buffers (read-only)
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;

// Output buffers (read-write)
// These now match the memory layout of PlanarStorageGaussian3d
@group(2) @binding(0) var<storage, read_write> out_position_visibility: array<vec4<f32>>;
@group(2) @binding(1) var<storage, read_write> out_spherical_harmonics: array<SphericalHarmonic>;
@group(2) @binding(2) var<storage, read_write> out_rotation: array<vec4<f32>>;
@group(2) @binding(3) var<storage, read_write> out_scale_opacity: array<vec4<f32>>;

// Helper function to create a quaternion from two unit vectors
fn quat_from_unit_vectors(u: vec3<f32>, v: vec3<f32>) -> vec4<f32> {
    // Check for parallel vectors to avoid issues with cross product
    if (dot(u, v) > 0.99999) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0); // Identity quaternion
    }
    if (dot(u, v) < -0.99999) {
        // Handle 180 degree rotation
        var axis = vec3<f32>(1.0, 0.0, 0.0);
        if (abs(u.x) > 0.9) {
            axis = vec3<f32>(0.0, 1.0, 0.0);
        }
        let w = cross(u, axis);
        let q = vec4<f32>(w.x, w.y, w.z, 0.0);
        return normalize(q);
    }

    let w = cross(u, v);
    let q = vec4<f32>(w.x, w.y, w.z, 1.0 + dot(u, v));
    return normalize(q);
}


@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tri_idx = global_id.x;

    // Guard against out-of-bounds access if the number of triangles isn't a multiple of the workgroup size.
    let indices_len = arrayLength(&indices);
    if (tri_idx * 3u >= indices_len) {
        return;
    }

    let i0 = indices[tri_idx * 3u];
    let i1 = indices[tri_idx * 3u + 1u];
    let i2 = indices[tri_idx * 3u + 2u];

    let p0 = positions[i0].xyz;
    let p1 = positions[i1].xyz;
    let p2 = positions[i2].xyz;

    // --- Calculate Splat Properties ---
    let center = (p0 + p1 + p2) / 3.0;

    let v0 = p1 - p0;
    let v1 = p2 - p0;
    let normal = normalize(cross(v0, v1));

    // Simple rotation to align z-axis with the triangle normal
    let rotation = quat_from_unit_vectors(vec3<f32>(0.0, 0.0, 1.0), normal);

    // Simple scale based on triangle edge lengths
    let scale_x = length(v0) * 0.33;
    let scale_y = length(v1) * 0.33;
    let scale_z = 0.01; // Surfel thickness
    let opacity = 1.0;

    // --- Write to Output Buffers ---
    out_position_visibility[tri_idx] = vec4<f32>(center, 1.0);
    out_rotation[tri_idx] = vec4<f32>(rotation.w, rotation.x, rotation.y, rotation.z);
    out_scale_opacity[tri_idx] = vec4<f32>(scale_x, scale_y, scale_z, opacity);

    // --- Set Spherical Harmonics ---
    // We will set a simple, flat white color.
    // The first coefficient (DC term) controls the base color.
    // C0 = 0.28209479177387814
    let C0 = 1.0;
    var sh: SphericalHarmonic;
    
    // Initialize all coefficients to zero
    for (var i = 0; i < 48; i = i + 1) {
        sh.coefficients[i] = 0.0;
    }
    
    // Set the DC terms for RGB channels (coefficients 0, 16, 32)
    // These are the first coefficient of each of the 3 color channels
    sh.coefficients[0] = C0;   // Red DC term
    sh.coefficients[1] = C0;  // Green DC term  
    sh.coefficients[2] = C0;  // Blue DC term
    
    out_spherical_harmonics[tri_idx] = sh;
}