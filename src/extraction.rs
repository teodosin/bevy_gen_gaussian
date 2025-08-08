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
    voxel_world: Res<VoxelWorld>,
    mut surface_buffer: ResMut<SurfaceBuffer>,
    mut metrics: ResMut<Metrics>,
    mut last_instance_count: ResMut<LastInstanceCount>,
) {
    if !voxel_world.dirty { return; }

    surface_buffer.instances.clear();
    surface_buffer.instances.extend(
        voxel_world.chunk.iter().map(|position| {
            let color = Color::srgb(
                position.x as f32 / 32.0,
                position.y as f32 / 32.0,
                position.z as f32 / 32.0,
            );
            BillboardInstance {
                pos: Vec3::new(position.x as f32, position.y as f32, position.z as f32),
                color: color.to_srgba().to_u8_array(),
            }
        })
    );

    surface_buffer.dirty = true;
    metrics.instance_count = surface_buffer.instances.len() as u64;
    if metrics.instance_count != last_instance_count.0 {
        println!("extract_surface: {} instances", metrics.instance_count);
        last_instance_count.0 = metrics.instance_count;
    }
}
