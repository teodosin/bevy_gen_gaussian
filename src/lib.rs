pub mod voxel;
pub mod gaussian;
pub mod sdf_module;
pub mod debug;

use bevy::prelude::*;

pub struct GenGaussianPlugin;

impl Plugin for GenGaussianPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(voxel::VoxelPlugin);
    }
}

// Public API exports - re-export from voxel module
pub use voxel::{
    Voxel, VoxelChunkSimple, VoxelData, MaterialId,
    EditOp, EditBatch, VoxelWorld, queue_set,
    BrushSettings, BrushMode, apply_sphere_brush, apply_box_brush, cast_editing_ray, RaycastMode, generate_terrain,
    LastInstanceCount, BillboardTag, VoxelBillboard, Metrics, DebugOverlayPlugin
};
pub use bevy_panorbit_camera::PanOrbitCamera;

// Re-export the main Gaussian APIs for convenience
pub use gaussian::*;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::gaussian::{
        creation::*,
        transform::*,
        settings::*,
    };
    pub use bevy_gaussian_splatting::{
        Gaussian3d, 
        PlanarGaussian3d, 
        PlanarGaussian3dHandle,
        CloudSettings,
        RasterizeMode,
    };
    pub use bevy::prelude::*;
}
