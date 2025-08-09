/// Bevy Gen Gaussian - Utility functions for creating and manipulating Gaussian clouds
/// 
/// This crate provides modular, composable functions for working with Gaussian clouds
/// in Bevy, designed to complement the bevy_gaussian_splatting crate.

pub mod gaussian;
pub mod voxel;
pub mod sdf_module;
pub mod debug;

// Re-export the main APIs for convenience
pub use gaussian::*;

// Optional: re-export the voxel plugin for users who want the full voxel system
pub use voxel::VoxelPlugin;

// Re-export bevy_gaussian_splatting types for convenience
pub use bevy_gaussian_splatting::{
    Gaussian3d, 
    PlanarGaussian3d, 
    PlanarGaussian3dHandle,
    CloudSettings,
    RasterizeMode,
};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

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
