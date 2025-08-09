use bevy::prelude::*;
use super::metrics::Metrics;

#[derive(Component)]
struct DebugOverlayRoot;

#[derive(Resource)]
struct FpsReportTimer {
    timer: Timer,
}

impl Default for FpsReportTimer {
    fn default() -> Self {
        Self {
            timer: Timer::from_seconds(10.0, TimerMode::Repeating),
        }
    }
}

pub struct DebugOverlayPlugin;
impl Plugin for DebugOverlayPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FpsReportTimer>()
            .add_systems(Startup, setup_ui)
            .add_systems(Update, (update_ui, periodic_fps_report));
    }
}

fn setup_ui(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(10.0),
                right: Val::Px(10.0),
                padding: UiRect::all(Val::Px(10.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.7)),
            DebugOverlayRoot,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Loading..."),
                TextFont {
                    font_size: 11.0,
                    ..default()
                },
                TextColor(Color::srgb(1.0, 1.0, 1.0)),
            ));
        });
}

fn update_ui(
    mut q: Query<&mut Text>,
    overlay_query: Query<Entity, With<DebugOverlayRoot>>,
    children_query: Query<&Children>,
    metrics: Res<Metrics>, 
    time: Res<Time>
) {
    if let Ok(overlay_entity) = overlay_query.single() {
        if let Ok(children) = children_query.get(overlay_entity) {
            if let Some(&text_entity) = children.first() {
                if let Ok(mut text) = q.get_mut(text_entity) {
                    **text = format!(
                        "Voxel Debug Info\n\
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
        }
    }
}

fn periodic_fps_report(
    mut fps_timer: ResMut<FpsReportTimer>,
    metrics: Res<Metrics>,
    time: Res<Time>,
) {
    fps_timer.timer.tick(time.delta());
    
    if fps_timer.timer.just_finished() {
        println!("FPS Report: {:.1} fps | {} voxels | {} instances", 
                 metrics.fps, metrics.voxel_count, metrics.instance_count);
    }
}
