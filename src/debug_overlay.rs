use bevy::prelude::*;
use crate::metrics::Metrics;

#[derive(Component)]
struct DebugOverlayRoot;

pub struct DebugOverlayPlugin;
impl Plugin for DebugOverlayPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_ui)
            .add_systems(Update, update_ui);
    }
}

fn setup_ui(mut commands: Commands) {
    commands.spawn((
        Text::new("Loading..."),
        TextFont {
            font_size: 12.0,
            ..default()
        },
        TextColor(Color::srgb(1.0, 1.0, 1.0)),
        DebugOverlayRoot,
    ));
}

fn update_ui(mut q: Query<&mut Text, With<DebugOverlayRoot>>, metrics: Res<Metrics>, time: Res<Time>) {
    if let Ok(mut text) = q.single_mut() {
        **text = format!(
            "Voxel Debug Info\n\
            \n\
            Controls:\n\
            • F: Fill voxels\n\
            • Mouse: Left-click + drag to orbit\n\
            • Shift/Right-click + drag to pan\n\
            • Scroll wheel to zoom\n\
            • Touch: Single finger to orbit\n\
            • Touch: Two fingers to pan/zoom\n\
            \n\
            Stats:\n\
            • FPS: {:.1}\n\
            • Voxels: {}\n\
            • Instances: {}\n\
            • Edits: {}\n\
            • Time: {:.2}s",
            metrics.fps,
            metrics.voxel_count,
            metrics.instance_count,
            metrics.edits_applied,
            time.elapsed_secs()
        );
    }
}
