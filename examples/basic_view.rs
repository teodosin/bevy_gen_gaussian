use bevy::prelude::*;
use bevy_gen_gaussian::{
    GenGaussianPlugin,
    EditOp, EditBatch, VoxelWorld, LastInstanceCount, Metrics,
};
use bevy_panorbit_camera::PanOrbitCamera;

#[derive(Component)]
struct InfoText;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window { title: "basic_view".into(), ..default() }),
                ..default()
            }),
            GenGaussianPlugin,
        ))
        .insert_resource(LastInstanceCount::default())
        .add_systems(Startup, (setup, setup_ui))
        .add_systems(Update, (temp_input, draw_debug, update_info_text))
        .run();
}

fn setup(mut commands: Commands) {
    // UI camera for the overlay text
    commands.spawn(Camera2d);
    
    // 3D camera for voxel rendering with natural trackpad controls
    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: 45.0_f32.to_radians(), // Narrower FOV to reduce fisheye effect
            ..default()
        }),
        PanOrbitCamera {
            // Set focal point (what the camera should look at)
            focus: Vec3::new(16.0, 16.0, 16.0),
            // Set the starting position, relative to focus
            yaw: Some(0.8),
            pitch: Some(0.4),
            radius: Some(64.0),
            // Natural trackpad behavior - no clicks required!
            button_orbit: MouseButton::Left,  // Still works for mouse
            button_pan: MouseButton::Right,   // Still works for mouse
            // Touch/trackpad controls (these work automatically!)
            touch_enabled: true,
            trackpad_pinch_to_zoom_enabled: true,
            trackpad_sensitivity: 1.0,
            orbit_sensitivity: 1.0,
            pan_sensitivity: 0.001,
            zoom_sensitivity: 0.8,
            // Allow the camera to go upside down for full freedom
            allow_upside_down: true,
            ..default()
        },
    ));

    // Directional light to illuminate the scene
    commands.spawn((
        DirectionalLight { 
            illuminance: 20000.0,
            ..default() 
        },
        Transform::from_xyz(50.0, 100.0, 50.0)
    ));
}

fn setup_ui(mut commands: Commands) {
    commands.spawn((
        Text::new("Voxel Debug Info"),
        TextFont {
            font_size: 12.0,
            ..default()
        },
        TextColor(Color::srgb(1.0, 1.0, 1.0)),
        InfoText,
    ));
}

fn temp_input(mut batch: ResMut<EditBatch>, keys: Res<ButtonInput<KeyCode>>) {
    if keys.just_pressed(KeyCode::KeyF) {
        for z in 0..32 {
            for y in 0..32 {
                for x in 0..32 {
                    if (x + y + z) % 11 == 0 {
                        batch.ops.push(EditOp::Set(IVec3::new(x, y, z)));
                    }
                }
            }
        }
    }
   
}

fn update_info_text(
    mut text_query: Query<&mut Text, With<InfoText>>,
    metrics: Res<Metrics>,
    time: Res<Time>,
) {
    let Ok(mut text) = text_query.single_mut() else { return };
    
    **text = format!("
        Controls:
        • F: Fill voxels
        • Trackpad: Natural gestures (orbit, pan, zoom)
        • Mouse: Left drag = orbit, Right drag = pan
        • Scroll/Pinch: Zoom in/out
        
        Stats:
        • FPS: {:.1}
        • Voxels: {}
        • Instances: {}
        • Edits: {}
        • Time: {:.2}s",
        metrics.fps,
        metrics.voxel_count,
        metrics.instance_count,
        metrics.edits_applied,
        time.elapsed_secs()
    );
}

fn draw_debug(world: Res<VoxelWorld>, mut gizmos: Gizmos) {
    for p in world.chunk.iter() {
        gizmos.cuboid(Transform::from_xyz(p.x as f32 + 0.5, p.y as f32 + 0.5, p.z as f32 + 0.5).with_scale(Vec3::splat(0.9)), Color::srgb(0.2,0.7,1.0));
    }
}
