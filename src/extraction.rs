use bevy::prelude::*;
use crate::edit::VoxelWorld;
use crate::metrics::Metrics;

#[derive(Clone, Copy)]
pub struct BillboardInstance { pub pos: Vec3, pub color: [u8;4] }

#[derive(Resource, Default)]
pub struct SurfaceBuffer { pub instances: Vec<BillboardInstance>, pub dirty: bool }

#[derive(Resource, Default)]
pub struct LastInstanceCount(pub u64);

pub fn extract_surface(
    mut voxel_world: ResMut<VoxelWorld>,
    mut surface_buffer: ResMut<SurfaceBuffer>,
    mut metrics: ResMut<Metrics>,
    mut last_instance_count: ResMut<LastInstanceCount>,
) {
    if !voxel_world.dirty { return; }

    surface_buffer.instances.clear();
    
    // Debug: Check total count manually
    let total_count = voxel_world.chunk.count();
    println!("Surface extraction: chunk reports {} voxels", total_count);
    
    // Debug: Try to get a specific voxel that should exist
    let test_voxel = voxel_world.chunk.get(IVec3::new(0, 0, 0));
    println!("Test voxel at (0,0,0): {:?}", test_voxel);
    
    surface_buffer.instances.extend(
        voxel_world.chunk.iter().map(|(position, voxel_data)| {
            // Enhanced depth-based coloring with material variation
            let depth = position.y as f32 / 32.0; // 0.0 at bottom, 1.0 at top
            let material_id = voxel_data.material;
            
            // Base color from depth
            let base_color = if depth < 0.3 {
                // Lower levels: Red tones (underground/foundation)
                Color::srgb(
                    0.8 + depth * 0.2,     // 0.8 - 1.0 red
                    depth * 0.4,           // 0.0 - 0.12 green
                    depth * 0.2,           // 0.0 - 0.06 blue
                )
            } else if depth < 0.7 {
                // Middle levels: Green tones (ground/surface)
                let mid_depth = (depth - 0.3) / 0.4; // Normalize to 0-1
                Color::srgb(
                    mid_depth * 0.3,       // 0.0 - 0.3 red
                    0.7 + mid_depth * 0.3, // 0.7 - 1.0 green
                    mid_depth * 0.2,       // 0.0 - 0.2 blue
                )
            } else {
                // Upper levels: Blue tones (sky/air)
                let high_depth = (depth - 0.7) / 0.3; // Normalize to 0-1
                Color::srgb(
                    high_depth * 0.2,      // 0.0 - 0.2 red
                    high_depth * 0.4,      // 0.0 - 0.4 green
                    0.8 + high_depth * 0.2, // 0.8 - 1.0 blue
                )
            };
            
            // Add material-based variation for visual interest
            let rgba = base_color.to_srgba();
            let material_color = match material_id % 5 {
                0 => base_color, // Keep base color
                1 => Color::srgb(rgba.red * 1.2, rgba.green * 0.8, rgba.blue * 0.9), // Redder
                2 => Color::srgb(rgba.red * 0.8, rgba.green * 1.2, rgba.blue * 0.9), // Greener
                3 => Color::srgb(rgba.red * 0.9, rgba.green * 0.8, rgba.blue * 1.2), // Bluer
                4 => Color::srgb(rgba.red * 1.1, rgba.green * 1.1, rgba.blue * 0.7), // Yellower
                _ => base_color,
            };
            
            BillboardInstance {
                pos: Vec3::new(
                    position.x as f32 + 0.5, 
                    position.y as f32 + 0.5, 
                    position.z as f32 + 0.5
                ),
                color: material_color.to_srgba().to_u8_array(),
            }
        })
    );

    surface_buffer.dirty = true;
    voxel_world.dirty = false; // Reset the voxel world dirty flag
    metrics.instance_count = surface_buffer.instances.len() as u64;
    
    // Only print when instance count changes
    if metrics.instance_count != last_instance_count.0 {
        println!("Surface extraction: {} instances", metrics.instance_count);
        last_instance_count.0 = metrics.instance_count;
    }
}
