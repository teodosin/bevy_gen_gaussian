struct FluidParams {
    gaussian_count: u32,
    dt: f32,
    elapsed: f32,
    _pad0: f32,
    bounds_min: vec2<f32>,
    bounds_max: vec2<f32>,
    damping: f32,
    speed_limit: f32,
    swirl_strength: f32,
    _pad1: f32,
    force: vec2<f32>,
};
@group(0) @binding(0) var<uniform> params: FluidParams;

@group(1) @binding(0) var<storage, read_write> out_position_visibility: array<vec4<f32>>;
@group(2) @binding(0) var<storage, read_write> velocities: array<vec2<f32>>;

@compute @workgroup_size(256, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.gaussian_count) { return; }

    var pv = out_position_visibility[i];
    var p = pv.xyz;
    var v = velocities[i];

    let r = vec2<f32>(p.x, p.y);
    let rlen = max(length(r), 1e-3);
    let tang = vec2<f32>(-r.y, r.x) / rlen;
    v += tang * params.swirl_strength * params.dt;
    v += params.force * params.dt;

    let spd = length(v);
    if (spd > params.speed_limit) {
        v = normalize(v) * params.speed_limit;
    }

    p.x += v.x * params.dt;
    p.y += v.y * params.dt;

    let bmin = params.bounds_min;
    let bmax = params.bounds_max;
    let bounce = 0.8;
    if (p.x < bmin.x) { p.x = bmin.x; v.x = -v.x * bounce; }
    if (p.x > bmax.x) { p.x = bmax.x; v.x = -v.x * bounce; }
    if (p.y < bmin.y) { p.y = bmin.y; v.y = -v.y * bounce; }
    if (p.y > bmax.y) { p.y = bmax.y; v.y = -v.y * bounce; }

    v *= params.damping;

    out_position_visibility[i] = vec4<f32>(p, pv.w);
    velocities[i] = v;
}
