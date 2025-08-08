# Gaussian Splat Implementation Backup

This file contains the successful Gaussian splat implementation that we developed, before reverting to opaque billboards to avoid transparency sorting issues.

## Key Components

### 1. Gaussian Splat Shader (gaussian_splat.wgsl)
```wgsl
#import bevy_pbr::{
    mesh_functions,
    mesh_view_bindings::view,
    view_transformations::position_world_to_clip,
}

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    
    // Get the world transform for this instance (contains splat center position)
    var world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);
    
    // Extract the splat center position from the transform
    let splat_center = world_from_local[3].xyz;
    
    // Generate color based on world position for variety
    let color_base = splat_center * 0.1;
    let instance_color = vec4<f32>(
        abs(sin(color_base.x)) * 0.7 + 0.3,
        abs(sin(color_base.y + 2.0)) * 0.7 + 0.3,
        abs(sin(color_base.z + 4.0)) * 0.7 + 0.3,
        1.0
    );
    
    // Camera position and view direction
    let camera_position = view.world_position.xyz;
    let to_camera = normalize(camera_position - splat_center);
    
    // Create billboard-aligned vectors for a camera-facing splat
    let world_up = vec3<f32>(0.0, 1.0, 0.0);
    let right = normalize(cross(world_up, to_camera));
    let up = cross(to_camera, right);
    
    // Fixed splat scale (we can make this dynamic later)
    let splat_scale = 1.0;
    
    // Transform vertex position from quad space to world space
    let world_pos = splat_center + 
                   right * vertex.position.x * splat_scale + 
                   up * vertex.position.y * splat_scale;
    
    out.clip_position = position_world_to_clip(world_pos);
    
    // UV coordinates go from -1 to 1 for the quad vertices
    out.uv = vertex.position.xy;
    out.color = instance_color;
    
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Calculate distance from center (UV coordinates go from -1 to 1)
    let center_dist = length(in.uv);
    
    // Gaussian falloff - creates the classic splat shape
    let gaussian = exp(-center_dist * center_dist * 2.0);
    
    // Apply gaussian to alpha for proper blending
    var splat_color = in.color;
    splat_color.a *= gaussian;
    
    // Discard pixels with very low alpha to improve performance
    if (splat_color.a < 0.01) {
        discard;
    }
    
    return splat_color;
}
```

### 2. Material Configuration for Transparency
```rust
impl Material for GaussianSplatMaterial {
    fn vertex_shader() -> ShaderRef {
        SHADER_ASSET_PATH.into()
    }

    fn fragment_shader() -> ShaderRef {
        SHADER_ASSET_PATH.into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend  // For transparency
        // OR AlphaMode::Add  // For additive blending (reduces flickering)
    }

    fn depth_bias(&self) -> f32 {
        0.001  // Small depth bias to reduce z-fighting
    }
}
```

### 3. Normal Vector Support for Splat Orientation
```rust
// Simple normal lookup table for splat orientation
fn get_normal_from_index(index: u8) -> Vec3 {
    match index % 6 {
        0 => Vec3::X,      // Right
        1 => Vec3::NEG_X,  // Left
        2 => Vec3::Y,      // Up
        3 => Vec3::NEG_Y,  // Down
        4 => Vec3::Z,      // Forward
        5 => Vec3::NEG_Z,  // Back
        _ => Vec3::Y,      // Default up
    }
}

#[derive(Clone, Copy)]
pub struct GaussianInstance { 
    pub pos: Vec3, 
    pub color: [u8; 4],
    pub normal: Vec3,  // For splat orientation
}
```

### 4. Material-Based Color Variation
```rust
// Add material-based variation for visual interest
let material_color = match material_id % 5 {
    0 => base_color, // Keep base color
    1 => Color::srgb(...), // Redder variation
    2 => Color::srgb(...), // Greener variation
    3 => Color::srgb(...), // Bluer variation
    4 => Color::srgb(...), // Yellower variation
    _ => base_color,
};
```

## Issues Encountered

1. **Transparency Sorting**: Transparent Gaussian splats caused flickering due to GPU depth sorting issues
2. **Alpha Blending**: Standard alpha blending requires back-to-front rendering which is expensive for thousands of overlapping splats
3. **Additive Blending**: Reduced flickering but could become too bright with many overlaps
4. **Z-Fighting**: Multiple splats at similar depths created visual artifacts

## Solutions Attempted

1. **AlphaMode::Blend**: Standard transparency, but with sorting issues
2. **AlphaMode::Add**: Additive blending, reduced flickering but brightness issues
3. **Depth Bias**: Small bias to reduce z-fighting
4. **Alpha Reduction**: Reduced overall alpha to prevent over-bright overlaps

## Future Implementation Ideas

1. **GPU Radix Sorting**: Implement proper depth sorting on GPU (like bevy_gaussian_splatting)
2. **Order-Independent Transparency**: Use techniques like weighted blended OIT
3. **Temporal Reprojection**: Reduce popping with frame-to-frame coherence
4. **Subgroup Sorting**: Wait for wgpu subgroup support for better GPU sorting
5. **Hybrid Approach**: Use opaque billboards for dense areas, transparent splats for sparse areas
