/// SDF (Signed Distance Function) module
/// 
/// This module provides functions for creating, combining, and converting
/// signed distance functions to Gaussian clouds.
use bevy::ecs::component::Component;



pub mod primitives;
pub mod operations;
pub mod conversion;

// Re-export main API
pub use primitives::*;
pub use operations::*; 
pub use conversion::*;





/// Component to hold a list of SDF edits. 
#[derive(Component)]
pub struct SDFList {
    pub sdfs: Vec<Box<dyn SDF>>,
}




pub struct SDFEdit {
    pub sdf: Box<dyn SDF>,
    pub operation: SDFOperation,
}