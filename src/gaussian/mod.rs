// Gaussian module - pure functions for creating and manipulating Gaussian clouds

pub mod cpu_mesh_to_gaussians;
pub mod cpu_transform;
pub mod settings;

// Re-export the main public API
pub use cpu_mesh_to_gaussians::*;
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

#[derive(Debug, Clone, Copy, Reflect)]
pub enum MeshToGaussianMode {
    TrianglesOneToOne,
}