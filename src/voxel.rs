use bevy::math::IVec3;
use std::collections::HashSet;

pub type MaterialId = u8;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Voxel(pub IVec3);

#[derive(Default)]
pub struct VoxelChunkSimple {
    filled: HashSet<IVec3>,
}

impl VoxelChunkSimple {
    pub fn set(&mut self, p: IVec3, _mat: MaterialId) { self.filled.insert(p); }
    pub fn clear(&mut self, p: IVec3) { self.filled.remove(&p); }
    pub fn iter(&self) -> impl Iterator<Item=&IVec3> { self.filled.iter() }
    pub fn count(&self) -> usize { self.filled.len() }
}
