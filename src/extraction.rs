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
    surface_buffer.instances.extend(
        voxel_world.chunk.iter().map(|position| {
            // Enhanced depth-based coloring for better visual feedback
            let depth = position.y as f32 / 32.0; // 0.0 at bottom, 1.0 at top
            
            let color = if depth < 0.3 {
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
            
            BillboardInstance {
                pos: Vec3::new(
                    position.x as f32 + 0.5, 
                    position.y as f32 + 0.5, 
                    position.z as f32 + 0.5
                ),
                color: color.to_srgba().to_u8_array(),
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
