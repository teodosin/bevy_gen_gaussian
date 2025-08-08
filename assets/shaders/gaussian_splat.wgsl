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
    
    // Create circular splats by discarding pixels outside radius
    if (center_dist > 1.0) {
        discard;
    }
    
    // Gaussian falloff for smooth circular shape
    let gaussian = exp(-center_dist * center_dist * 2.0);
    
    // Apply gaussian to color intensity for opaque circular splats
    var splat_color = in.color;
    splat_color = vec4<f32>(splat_color.rgb * gaussian, 1.0);
    
    // Discard very dark pixels for clean edges
    if (length(splat_color.rgb) < 0.1) {
        discard;
    }
    
    return splat_color;
}
