use bevy::prelude::*;
use crate::edit::VoxelWorld;

#[derive(Resource, Default, Debug)]
pub struct Metrics {
    pub frame_time_ms: f32,
    pub fps: f32,
    pub voxel_count: u64,
    pub instance_count: u64,
    pub edits_applied: u64,
}

pub fn flush_metrics(
    mut metrics: ResMut<Metrics>,
    time: Res<Time>,
    world: Option<Res<VoxelWorld>>,
) {
    metrics.frame_time_ms = time.delta_secs() * 1000.0;
    metrics.fps = 1.0 / time.delta_secs();
    if let Some(world) = world {
        metrics.voxel_count = world.chunk.count() as u64;
    }
}
