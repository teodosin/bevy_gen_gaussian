use bevy::prelude::*;
use crate::voxel::*;
use bevy::math::IVec3;
use crate::metrics::Metrics;

#[derive(Clone, Copy)]
pub enum EditOp { Set(IVec3), Clear(IVec3) }

#[derive(Resource, Default)]
pub struct EditBatch { pub ops: Vec<EditOp> }

#[derive(Resource, Default)]
pub struct VoxelWorld { pub chunk: VoxelChunkSimple, pub dirty: bool }

pub fn queue_set(mut batch: ResMut<EditBatch>, p: IVec3) {
    batch.ops.push(EditOp::Set(p));
    println!("queue_set: queued set at {:?}", p);
}

pub fn apply_edits(mut world: ResMut<VoxelWorld>, mut batch: ResMut<EditBatch>, mut metrics: ResMut<Metrics>) {
    if batch.ops.is_empty() { return; }
    let mut applied = 0u64;
    for op in batch.ops.drain(..) {
        match op {
            EditOp::Set(p) => {
                world.chunk.set(p, 1);
                applied += 1;
            },
            EditOp::Clear(p) => {
                world.chunk.clear(p);
                applied += 1;
            }
        }
    }
    world.dirty = true;
    metrics.edits_applied += applied;
    println!("apply_edits: applied {applied} new ops (total_applied = {})", metrics.edits_applied);
}
