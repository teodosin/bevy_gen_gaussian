//! Global size & coordinate constants

pub const VOXEL_SIZE: f32 = 1.0; // meters (temporary)
pub const CHUNK_SIZE: i32 = 32; // one test chunk for MVP
pub const SUB_BRICK: i32 = 8;   // dirty region granularity

#[inline]
pub fn world_to_local(x: f32) -> i32 { x.floor() as i32 }
