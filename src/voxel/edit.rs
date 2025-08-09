use bevy::prelude::*;
use super::grid::*;
use bevy::math::IVec3;
use super::metrics::Metrics;

#[derive(Clone, Copy)]
pub enum EditOp { Set(IVec3), Clear(IVec3) }

#[derive(Resource, Default)]
pub struct EditBatch { pub ops: Vec<EditOp> }

#[derive(Resource)]
pub struct VoxelWorld { pub chunk: VoxelChunkSimple, pub dirty: bool }

impl Default for VoxelWorld {
    fn default() -> Self {
        Self {
            chunk: VoxelChunkSimple::new(),
            dirty: false,
        }
    }
}

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
                // Test if we can read back what we just set
                if applied <= 3 {
                    let readback = world.chunk.get(p);
                    println!("  Set {:?} -> readback: {:?}", p, readback);
                }
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
