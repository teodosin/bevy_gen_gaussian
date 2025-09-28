// Spherical harmonics data for a single Gaussian.
// This must match the layout expected by bevy_gaussian_splatting: array<f32, 48>
// where 48 = SH_COEFF_COUNT (16 coefficients per channel * 3 channels)
struct SphericalHarmonic {
    coefficients: array<f32, 48>,
}


// Input buffers (read-only)
@group(0) @binding(0) var<storage, read>    positions:     array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>    indices:       array<u32>;

// Per-view params (dynamic uniform)
struct TriToSplatParams {
    gaussian_count:   u32,
    elapsed_seconds:  f32,
    duration_seconds: f32,
    _pad:             f32,
    sphere_center:    vec3<f32>,
    sphere_radius:    f32,
    // Lighting params
    light_dir:        vec3<f32>,
    _pad2:            f32,
    base_color:       vec3<f32>,
    _pad3:            f32,
}
@group(1) @binding(0) var<uniform> params: TriToSplatParams;


// Output buffers (read-write)
// These now match the memory layout of PlanarStorageGaussian3d
@group(2) @binding(0) var<storage, read_write>      out_position_visibility:     array<vec4<f32>>;
@group(2) @binding(1) var<storage, read_write>      out_spherical_harmonics:     array<SphericalHarmonic>;
@group(2) @binding(2) var<storage, read_write>      out_rotation:                array<vec4<f32>>;
@group(2) @binding(3) var<storage, read_write>      out_scale_opacity:           array<vec4<f32>>;







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





// --- Additional math / noise helpers ---
fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    // Quaternions in (x, y, z, w)
    let av = a.xyz;
    let bv = b.xyz;
    let w = a.w * b.w - dot(av, bv);
    let v = a.w * bv + b.w * av + cross(av, bv);
    return vec4<f32>(v, w);
}

fn quat_from_axis_angle(axis: vec3<f32>, angle: f32) -> vec4<f32> {
    let half = 0.5 * angle;
    let s = sin(half);
    return vec4<f32>(normalize(axis) * s, cos(half));
}

// Normalized LERP between two quaternions (shortest path)
fn quat_nlerp(a: vec4<f32>, b_in: vec4<f32>, t: f32) -> vec4<f32> {
    var b = b_in;
    // Ensure shortest path
    if (dot(a, b) < 0.0) {
        b = -b;
    }
    return normalize(mix(a, b, t));
}

fn fade3(t: vec3<f32>) -> vec3<f32> {
    // Smooth interpolation curve per component (Perlin's 3t^2 - 2t^3)
    return t * t * (3.0 - 2.0 * t);
}

fn hash31(i: vec3<i32>) -> f32 {
    // Hash a 3D integer grid point to [-1, 1]
    let p = vec3<f32>(i);
    let h = sin(dot(p, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453;
    return fract(h) * 2.0 - 1.0;
}

fn value_noise3d(p: vec3<f32>) -> f32 {
    let i0 = vec3<i32>(floor(p));
    let f0 = fract(p);
    let u = fade3(f0);

    let i000 = i0 + vec3<i32>(0, 0, 0);
    let i100 = i0 + vec3<i32>(1, 0, 0);
    let i010 = i0 + vec3<i32>(0, 1, 0);
    let i110 = i0 + vec3<i32>(1, 1, 0);
    let i001 = i0 + vec3<i32>(0, 0, 1);
    let i101 = i0 + vec3<i32>(1, 0, 1);
    let i011 = i0 + vec3<i32>(0, 1, 1);
    let i111 = i0 + vec3<i32>(1, 1, 1);

    let n000 = hash31(i000);
    let n100 = hash31(i100);
    let n010 = hash31(i010);
    let n110 = hash31(i110);
    let n001 = hash31(i001);
    let n101 = hash31(i101);
    let n011 = hash31(i011);
    let n111 = hash31(i111);

    let nx00 = mix(n000, n100, u.x);
    let nx10 = mix(n010, n110, u.x);
    let nx01 = mix(n001, n101, u.x);
    let nx11 = mix(n011, n111, u.x);
    let nxy0 = mix(nx00, nx10, u.y);
    let nxy1 = mix(nx01, nx11, u.y);
    let nxyz = mix(nxy0, nxy1, u.z);
    return nxyz; // [-1, 1]
}

fn noise_vec3(p: vec3<f32>) -> vec3<f32> {
    // Decorrelate channels by offsetting the sampling domain
    let nx = value_noise3d(p + vec3<f32>(19.1, 0.0, 0.0));
    let ny = value_noise3d(p + vec3<f32>(0.0, 33.4, 0.0));
    let nz = value_noise3d(p + vec3<f32>(0.0, 0.0, 47.2));
    return vec3<f32>(nx, ny, nz);
}

// Hash helpers for deterministic per-triangle randomness
fn hash11(n: f32) -> f32 {
    return fract(sin(n * 17.0 + 0.1) * 43758.5453123);
}

fn hash21(n: f32) -> vec2<f32> {
    return vec2<f32>(
        hash11(n + 1.3),
        hash11(n + 7.5),
    );
}

// Rotate vector by quaternion q (x,y,z,w) assuming q is normalized
fn rotate_vec_by_quat(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
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


    // --- Calculate Splat Properties (targets) ---
    let center = (p0 + p1 + p2) / 3.0;

    let v0 = p1 - p0;
    let v1 = p2 - p0;
    let normal = normalize(cross(v0, v1));


    // Simple rotation to align z-axis with the triangle normal
    let base_rotation = quat_from_unit_vectors(vec3<f32>(0.0, 0.0, 1.0), normal);


    // Simple scale based on triangle edge lengths (target thickness kept small)
    let target_scale_x = length(v0) * 0.33;
    let target_scale_y = length(v1) * 0.33;
    let target_scale_z = 0.01; // Surfel thickness - TODO: Use the MeshToGaussian component for this

    // --- Time-based interpolation from origin singularity ---
    // Looping time base: 5-second cycle
    let cycle = 5.0;
    let cycle_time = fract(params.elapsed_seconds / cycle) * cycle;
    let duration = max(params.duration_seconds, 0.0001);
    var t = clamp(cycle_time / duration, 0.0, 1.0);
    // Ease in-out (smoothstep)
    t = t * t * (3.0 - 2.0 * t);

    // Random starting positions sampled on the surface of a sphere (SDF sphere) around the mesh
    // Deterministic per-triangle to keep temporal coherence
    let uv = hash21(f32(tri_idx));
    let z  = 1.0 - 2.0 * uv.x;               // z in [-1,1]
    let a  = 6.28318530718 * uv.y;           // angle
    let r  = sqrt(max(0.0, 1.0 - z * z));
    let dir = vec3<f32>(r * cos(a), r * sin(a), z);
    let start_pos = params.sphere_center + dir * params.sphere_radius;
    var pos_out   = mix(start_pos, center, t);

    // Start visible with a modest base size boost (~30%) and non-zero opacity
    let start_scale_factor = 3.6; // 130% of final per-axis size at t=0
    var scale_x   = mix(target_scale_x * start_scale_factor, target_scale_x, t);
    var scale_y   = mix(target_scale_y * start_scale_factor, target_scale_y, t);
    var scale_z   = mix(target_scale_z * start_scale_factor, target_scale_z, t);
    let opacity   = mix(0.3, 1.0, t); // visible at start, fully opaque by t=1

    // --- Coherent 3D noise field for position, scale, and rotation ---
    // Bell-shaped envelope peaking at t=0.5, zero at t=0 and t=1
    let env = 4.0 * t * (1.0 - t);

    // Time-advected sampling point (approx 4D noise)
    let noise_freq = 1.25;              // world->noise frequency (higher = finer noise)
    let flow_dir = normalize(vec3<f32>(0.73, 0.52, -0.41));
    let flow_speed = 0.35;              // units per second in noise space
    let time = cycle_time;              // looping time
    let p_field = center * noise_freq + flow_dir * (time * flow_speed);

    // Multi-channel noise vector in [-1, 1]^3
    let nvec = noise_vec3(p_field);

    // Position offset scales with local triangle size and envelope
    let base_len = 0.5 * (length(v0) + length(v1));
    let pos_amp = base_len * 0.8 * env;
    pos_out += nvec * pos_amp;

    // Anisotropic scale boost: only scale up (biased to be >= final size)
    let scale_boost = 1.0; // up to +100% per-axis at peak
    let scale_mult = vec3<f32>(1.0, 1.0, 1.0) + max(nvec, vec3<f32>(0.0)) * (scale_boost * env);
    scale_x = scale_x * scale_mult.x;
    scale_y = scale_y * scale_mult.y;
    scale_z = scale_z * scale_mult.z;

    // Ensure we never shrink below the final target size so jitter keeps them >= final
    scale_x = max(scale_x, target_scale_x);
    scale_y = max(scale_y, target_scale_y);
    scale_z = max(scale_z, target_scale_z);

    // Blend rotation from start (sphere normal) to target (triangle normal)
    let start_rotation = quat_from_unit_vectors(vec3<f32>(0.0, 0.0, 1.0), dir);
    let base_blend = quat_nlerp(start_rotation, base_rotation, t);

    // Noise-driven rotation around a coherent axis
    var axis = normalize(nvec);
    if (length(axis) < 1e-3) {
        axis = normal; // fallback to geometric normal when noise is near zero
    }
    let rot_noise = value_noise3d(p_field + vec3<f32>(57.2, -19.5, 78.4)); // [-1,1]
    let rot_max = 1.2; // radians (~69 deg) at peak
    let angle = rot_noise * rot_max * env;
    let q_noise = quat_from_axis_angle(axis, angle);
    let rotation_out = quat_mul(q_noise, base_blend);


    // --- Write to Output Buffers ---
    out_position_visibility[tri_idx]    = vec4<f32>(pos_out, 1.0);
    out_rotation[tri_idx]               = vec4<f32>(rotation_out.w, rotation_out.x, rotation_out.y, rotation_out.z);
    out_scale_opacity[tri_idx]          = vec4<f32>(scale_x, scale_y, scale_z, opacity);


    // --- Per-frame lighting and SH color ---
    // Current oriented normal is the rotated +Z
    let cur_normal = normalize(rotate_vec_by_quat(vec3<f32>(0.0, 0.0, 1.0), rotation_out));
    let L = normalize(params.light_dir);
    let ndotl = clamp(dot(cur_normal, L), 0.0, 1.0);
    let base = params.base_color;
    // Simple Lambert-style shading with no ambient: darker when away from light
    let rgb = base * ndotl;

    // SH DC coefficient encodes color via Y00 = 0.2821
    let sh_coeff_r = rgb.r / 0.2821;
    let sh_coeff_g = rgb.g / 0.2821;
    let sh_coeff_b = rgb.b / 0.2821;
    
    var sh: SphericalHarmonic;
    
    // Initialize all coefficients to zero
    for (var i = 0; i < 48; i = i + 1) {
        sh.coefficients[i] = 0.0;
    }
    

    // Set the DC terms for RGB channels
    sh.coefficients[0] = sh_coeff_r;   // Red DC term
    sh.coefficients[1] = sh_coeff_g;   // Green DC term  
    sh.coefficients[2] = sh_coeff_b;   // Blue DC term
    
    
    out_spherical_harmonics[tri_idx] = sh;
}