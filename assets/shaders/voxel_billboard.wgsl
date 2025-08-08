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
    @location(0) world_position: vec4<f32>,
};

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    
    // Get the world transform for this instance
    var world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);
    
    // Extract the billboard center position from the transform
    let billboard_center = world_from_local[3].xyz;
    
    // Create billboard-aligned vectors
    let camera_position = view.world_position.xyz;
    let to_camera = normalize(camera_position - billboard_center);
    
    // Create a right vector perpendicular to the view direction and world up
    let world_up = vec3<f32>(0.0, 1.0, 0.0);
    let right = normalize(cross(world_up, to_camera));
    let up = cross(to_camera, right);
    
    // Create billboard-aligned position
    let billboard_pos = billboard_center + 
                       right * vertex.position.x + 
                       up * vertex.position.y;
    
    out.world_position = vec4<f32>(billboard_pos, 1.0);
    out.clip_position = position_world_to_clip(billboard_pos);
    
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Simple bright magenta to test visibility
    return vec4<f32>(1.0, 0.0, 1.0, 1.0);
}
