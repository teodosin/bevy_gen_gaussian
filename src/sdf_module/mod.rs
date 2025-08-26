/// SDF (Signed Distance Function) module
/// 
/// This module provides functions for creating, combining, and converting
/// signed distance functions to Gaussian clouds.
use bevy::ecs::component::Component;



pub mod primitives;
pub mod operations;

// Re-export main API
pub use primitives::*;
pub use operations::*; 





/// Component to hold a list of SDF edits. 
#[derive(Component)]
pub struct SDFList {
    pub sdfs: Vec<SDFEdit>,
}




pub struct SDFEdit {
    pub sdf: Box<dyn SDF>,
    pub operation: SDFOperation,
}