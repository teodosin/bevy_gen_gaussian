use bevy::math::IVec3;

pub type MaterialId = u8;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Voxel(pub IVec3);

/// Compact voxel data: 2 bytes per voxel
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VoxelData {
    pub material: MaterialId,  // 256 different materials
    pub normal_index: u8,      // Index into normal lookup table (0-255)
}

impl VoxelData {
    pub fn new(material: MaterialId, normal_index: u8) -> Self {
        Self { material, normal_index }
    }
    
    pub fn with_material(material: MaterialId) -> Self {
        Self { material, normal_index: 0 } // Default normal
    }
}

/// Fast Vec-based chunk storage instead of HashMap
pub struct VoxelChunkSimple {
    // Dense storage for a 32x32x32 chunk
    data: Vec<Option<VoxelData>>, // 32*32*32 = 32768 elements
    size: usize, // Chunk size (32 for now)
}

impl VoxelChunkSimple {
    pub fn new() -> Self {
        const CHUNK_SIZE: usize = 32;
        Self {
            data: vec![None; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE],
            size: CHUNK_SIZE,
        }
    }
    
    fn pos_to_index(&self, p: IVec3) -> Option<usize> {
        if p.x < 0 || p.y < 0 || p.z < 0 || 
           p.x >= self.size as i32 || p.y >= self.size as i32 || p.z >= self.size as i32 {
            return None;
        }
        Some((p.z as usize * self.size * self.size) + (p.y as usize * self.size) + p.x as usize)
    }
    
    pub fn set(&mut self, p: IVec3, material: MaterialId) {
        if let Some(index) = self.pos_to_index(p) {
            self.data[index] = Some(VoxelData::with_material(material));
        }
    }
    
    pub fn set_with_normal(&mut self, p: IVec3, material: MaterialId, normal_index: u8) {
        if let Some(index) = self.pos_to_index(p) {
            self.data[index] = Some(VoxelData::new(material, normal_index));
        }
    }
    
    pub fn clear(&mut self, p: IVec3) {
        if let Some(index) = self.pos_to_index(p) {
            self.data[index] = None;
        }
    }
    
    pub fn get(&self, p: IVec3) -> Option<VoxelData> {
        self.pos_to_index(p).and_then(|index| self.data[index])
    }
    
    pub fn is_set(&self, p: IVec3) -> bool {
        self.get(p).is_some()
    }
    
    /// Iterator over all set voxels with their positions
    pub fn iter(&self) -> impl Iterator<Item = (IVec3, VoxelData)> + '_ {
        self.data.iter().enumerate().filter_map(|(index, data)| {
            data.as_ref().map(|&voxel_data| {
                let x = (index % self.size) as i32;
                let y = ((index / self.size) % self.size) as i32;
                let z = (index / (self.size * self.size)) as i32;
                (IVec3::new(x, y, z), voxel_data)
            })
        })
    }
    
    /// Iterator over just positions (for compatibility)
    pub fn positions(&self) -> impl Iterator<Item = IVec3> + '_ {
        self.iter().map(|(pos, _)| pos)
    }
    
    pub fn count(&self) -> usize {
        self.data.iter().filter(|d| d.is_some()).count()
    }
}

impl Default for VoxelChunkSimple {
    fn default() -> Self {
        Self::new()
    }
}
