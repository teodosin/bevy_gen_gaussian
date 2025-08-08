pub mod constants;
pub mod voxel;
pub mod edit;
pub mod extraction;
pub mod billboard;
pub mod metrics;
pub mod debug_overlay;

use bevy::prelude::*;
use bevy_panorbit_camera::PanOrbitCameraPlugin;

pub struct GenGaussianPlugin;

impl Plugin for GenGaussianPlugin {
    fn build(&self, app: &mut App) {
        app
            .init_resource::<metrics::Metrics>()
            .init_resource::<edit::EditBatch>()
            .init_resource::<edit::VoxelWorld>()
            .init_resource::<extraction::SurfaceBuffer>()
            .init_resource::<billboard::BillboardGpu>()
            .add_systems(Update, (
                edit::apply_edits,
                // Disabled billboard/surface systems for voxel-only mode
                // extraction::extract_surface.after(edit::apply_edits),
                // billboard::update_billboard_instances.after(extraction::extract_surface),
                metrics::flush_metrics.after(edit::apply_edits),
            ))
            .add_plugins((
                PanOrbitCameraPlugin,
                // debug_overlay::DebugOverlayPlugin, // Disabled - examples handle their own UI
            ));
    }
}

// Public API exports
pub use voxel::{Voxel, VoxelChunkSimple};
pub use edit::{EditOp, EditBatch, VoxelWorld, queue_set};
pub use extraction::LastInstanceCount;
pub use billboard::BillboardTag;
pub use metrics::Metrics;
pub use bevy_panorbit_camera::PanOrbitCamera;
pub use debug_overlay::DebugOverlayPlugin;
