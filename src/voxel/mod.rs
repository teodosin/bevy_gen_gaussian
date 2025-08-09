pub mod constants;
pub mod grid;
pub mod edit;
pub mod sdf;
pub mod extraction;
pub mod billboard;
pub mod billboard_render;
pub mod billboard_instanced;
pub mod metrics;
pub mod debug_overlay;

use bevy::prelude::*;
use bevy_panorbit_camera::PanOrbitCameraPlugin;

pub struct VoxelPlugin;

impl Plugin for VoxelPlugin {
    fn build(&self, app: &mut App) {
        app
            .init_resource::<metrics::Metrics>()
            .init_resource::<edit::EditBatch>()
            .init_resource::<edit::VoxelWorld>()
            .init_resource::<extraction::SurfaceBuffer>()
            .init_resource::<billboard::BillboardGpu>()
            .add_plugins((
                PanOrbitCameraPlugin,
                MaterialPlugin::<billboard_instanced::VoxelBillboardMaterial>::default(),
                debug_overlay::DebugOverlayPlugin,
            ))
            .add_systems(Update, (
                edit::apply_edits,
                extraction::extract_surface.after(edit::apply_edits),
                // Only use the instanced billboard system now for performance
                billboard_instanced::manage_billboard_instances.after(extraction::extract_surface),
                metrics::flush_metrics.after(edit::apply_edits),
            ))
            .add_systems(Startup, (
                // Only setup instanced billboards
                billboard_instanced::setup_instanced_billboards,
            ));
    }
}

// Public API exports
pub use grid::{Voxel, VoxelChunkSimple, VoxelData, MaterialId};
pub use edit::{EditOp, EditBatch, VoxelWorld, queue_set};
pub use sdf::{BrushSettings, BrushMode, apply_sphere_brush, apply_box_brush, cast_editing_ray, RaycastMode, generate_terrain};
pub use extraction::LastInstanceCount;
pub use billboard::BillboardTag;
pub use billboard_render::VoxelBillboard;
pub use metrics::Metrics;
pub use bevy_panorbit_camera::PanOrbitCamera;
pub use debug_overlay::DebugOverlayPlugin;
