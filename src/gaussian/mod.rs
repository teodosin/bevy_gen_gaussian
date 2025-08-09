// Gaussian module - pure functions for creating and manipulating Gaussian clouds

pub mod creation;
pub mod transform;
pub mod settings;

// Re-export the main public API
pub use creation::*;
pub use transform::*;
pub use settings::*;
