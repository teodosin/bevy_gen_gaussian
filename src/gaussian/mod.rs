// Gaussian module - pure functions for creating and manipulating Gaussian clouds

pub mod cpu_mesh_to_gaussians;
pub mod gpu_mesh_to_gaussians;

pub mod cpu_transform;
pub mod settings;

// Re-export the main public API
pub use cpu_mesh_to_gaussians::*;
pub use gpu_mesh_to_gaussians::*;

pub use cpu_transform::*;
pub use settings::*;




use bevy::prelude::*;





/// Component to mark and configure mesh to Gaussian conversion.
/// This component can be attached to an entity with a Mesh to generate Gaussians from it.
///
/// The conversion can be done once or in real-time as the mesh changes.
/// A separate system will handle the conversion then based on these settings.
#[derive(Component, Debug, Clone, Reflect)]
pub struct MeshToGaussian {
    pub mode: MeshToGaussianMode,
    pub surfel_thickness: f32,
    pub hide_source_mesh: bool,
    pub realtime: bool,
}

impl Default for MeshToGaussian {
    fn default() -> Self {
        Self {
            mode: MeshToGaussianMode::TrianglesOneToOne,
            surfel_thickness: 0.01,
            hide_source_mesh: true,
            realtime: false,
        }
    }
}



#[derive(Debug, Clone, Copy, PartialEq, Eq, Reflect)]
pub enum MeshToGaussianMode {
    /// Generates one gaussian splat for each triangle in the mesh.
    TrianglesOneToOne,
    // Future modes like `Vertices1to1` or `AreaSampled` can be added here.
}





// --- GPU Conversion Plugin and Components ---

/// Plugin for the GPU-accelerated mesh to gaussian conversion pipeline.
pub struct GenGaussianGpuPlugin;

impl Plugin for GenGaussianGpuPlugin {
    fn build(&self, app: &mut App) {
        // We will fill this in with systems and resources in the next steps.
    }
}