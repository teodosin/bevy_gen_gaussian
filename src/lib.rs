pub mod gaussian;
pub mod sdf_module;
pub mod debug;

use bevy::prelude::*;
use bevy_gaussian_splatting::GaussianSplattingPlugin;

pub struct GenGaussianPlugin;

fn log_gen_gaussian_startup() {
    info!(
        "GenGaussianPlugin Startup: adding GaussianSplattingPlugin + GenGaussianGpuPlugin"
    );
}

impl Plugin for GenGaussianPlugin {
    fn build(&self, app: &mut App) {
        // Do not add voxel plugin by default; it's orthogonal and can panic if not configured.
        // Ensure the gaussian splatting renderer and assets are registered globally.
        app.add_systems(Startup, log_gen_gaussian_startup);
        app.add_plugins(GaussianSplattingPlugin);
        // Our GPU mesh->gaussian conversion systems
        app.add_plugins(gaussian::GenGaussianGpuPlugin);
    }
}


pub use bevy_panorbit_camera::PanOrbitCamera;

// Re-export the main Gaussian APIs for convenience
pub use gaussian::*;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::gaussian::{
        cpu_mesh_to_gaussians::*,
        cpu_transform::*,
        settings::*,
        gpu_mesh_to_gaussians::*,
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
