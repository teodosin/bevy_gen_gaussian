// shaders/tri_to_splat.wgsl
// Writes triangle->splat into bevy_gaussian_splatting planar storage buffers.
// Outputs use the renderer's group(2) layout via the bindings import.

#define_import_path bevy_gen_gaussian::tri_to_splat

// Import renderer bindings explicitly so symbols are in scope as group(2)
#import bevy_gaussian_splatting::bindings::{
  position_visibility,
  spherical_harmonics,
  rotation,
  scale_opacity,
}
struct Uniforms {
  world_from_mesh : mat4x4<f32>,
  thickness       : f32,   // normal-axis stddev (meters)
  visibility      : f32,   // 0..1
  opacity         : f32,   // 0..1
  tri_count       : u32,
}

@group(0) @binding(0) var<storage, read>        positions : array<vec3<f32>>; // vertex positions
@group(0) @binding(1) var<storage, read>        indices   : array<u32>;       // u32 triplets
@group(0) @binding(2) var<uniform>              U : Uniforms;

fn quat_from_mat3(m: mat3x3<f32>) -> vec4<f32> { // very small, stable conversion
  let tr = m[0][0] + m[1][1] + m[2][2];
  if (tr > 0.0) {
    let s = sqrt(tr + 1.0) * 2.0;
    return vec4<f32>((m[2][1]-m[1][2])/s, (m[0][2]-m[2][0])/s, (m[1][0]-m[0][1])/s, 0.25*s);
  }
  // (degenerate fallback; triangles rarely hit this)
  return vec4<f32>(0.0,0.0,0.0,1.0);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let t = gid.x;
  if (t >= U.tri_count) { return; }

  let i0 = indices[3u*t+0u];
  let i1 = indices[3u*t+1u];
  let i2 = indices[3u*t+2u];

  var p0 = vec4<f32>(positions[i0], 1.0);
  var p1 = vec4<f32>(positions[i1], 1.0);
  var p2 = vec4<f32>(positions[i2], 1.0);

  p0 = U.world_from_mesh * p0;
  p1 = U.world_from_mesh * p1;
  p2 = U.world_from_mesh * p2;

  let c  = (p0.xyz + p1.xyz + p2.xyz) / 3.0;
  let e0 = p1.xyz - p0.xyz;
  let e1 = p2.xyz - p0.xyz;
  let n  = normalize(cross(e0, e1));

  // Build tangent frame aligned with the triangle
  let x  = normalize(e0);
  let y  = normalize(cross(n, x));
  let R  = mat3x3<f32>(x, y, n);
  let q  = quat_from_mat3(R);

  // Estimate ellipse axes in-plane (covariance of the 3 projected points)
  let q0 = vec2<f32>(dot(p0.xyz - c, x), dot(p0.xyz - c, y));
  let q1 = vec2<f32>(dot(p1.xyz - c, x), dot(p1.xyz - c, y));
  let q2 = vec2<f32>(dot(p2.xyz - c, x), dot(p2.xyz - c, y));

  let M  = mat2x2<f32>(
     (q0.x*q0.x + q1.x*q1.x + q2.x*q2.x) / 3.0,
     (q0.x*q0.y + q1.x*q1.y + q2.x*q2.y) / 3.0,
     (q0.y*q0.x + q1.y*q1.x + q2.y*q2.x) / 3.0,
     (q0.y*q0.y + q1.y*q1.y + q2.y*q2.y) / 3.0
  );
  let tr = M[0][0] + M[1][1];
  let det = M[0][0]*M[1][1] - M[0][1]*M[1][0];
  let disc = sqrt(max(0.0, tr*tr - 4.0*det));
  let s1 = sqrt(max(1e-6, 0.5*(tr + disc)));
  let s2 = sqrt(max(1e-6, 0.5*(tr - disc)));

  let inflate = 1.25;          // small coverage safety
  let sx = inflate * s1;
  let sy = inflate * s2;
  let sz = max(1e-6, U.thickness);

  // NOTE: These variables come from the renderer bindings import and are bound as group(2)
  position_visibility[t] = vec4<f32>(c, U.visibility);
  rotation[t]            = q;
  scale_opacity[t]       = vec4<f32>(sx, sy, sz, U.opacity);
  // spherical_harmonics remains unchanged (defaults to 0); color will be black until filled elsewhere
}
