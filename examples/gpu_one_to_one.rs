//! # Mesh to Gaussian Cloud Converter
//!
//! This example covers the simplest case of converting a mesh to a Gaussian splat cloud,
//! generating one splat per triangle in the mesh. It uses the GPU-accelerated conversion
//! pipeline provided by the `bevy_gen_gaussian` crate.
//!
//! For an end user, the conversion is as simple as adding a `MeshToGaussian` component
//! to an entity with a mesh. The `GenGaussianPlugin` sets up the necessary systems
//! to process these entities and generate the Gaussian clouds.
//!
//! Controls:
//! - WASD: Orbit camera around the model
//! - Q/E: Zoom in and out

use bevy::prelude::*;

use bevy_gaussian_splatting::{ GaussianCamera };
use bevy::ui::Val::Px;
use bevy_gen_gaussian::{GenGaussianPlugin, MeshToGaussian, MeshToGaussianMode, TriToSplatParams};

/// Path to the mesh asset to convert
const MESH_PATH: &str = "scenes/monkey.glb";







fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(GenGaussianPlugin)

        .add_systems(Startup,
            (
                setup_scene,
                setup_ui,
                load_mesh
            )
        )

        .add_systems(Update, (
            camera_controls,
            update_info_text,
        ))

        .run();
}







// --- Resources ---

/// Resource holding the scene handle while waiting for it to load
#[derive(Resource, Default)]
struct PendingMeshScene(Handle<Scene>);







// --- Components ---

/// Marker for the UI info text
#[derive(Component)]
struct InfoText;







// --- Systems ---

/// Set up the 3D scene with camera and lighting
fn setup_scene(mut commands: Commands) {
    // 2D UI camera for overlay text: give it a higher order to render after 3D
    commands.spawn((
        Camera2d,
        Camera { order: 10, ..default() },
    ));
    
    // 3D camera for Gaussian rendering - positioned to view the model
    commands.spawn((
        GaussianCamera { warmup: true },
        Camera3d::default(),
        Camera {
            order: 0,
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        Transform::from_translation(Vec3::new(0.0, 1.0, 8.0))
            .looking_at(Vec3::ZERO, Vec3::Y),
        TriToSplatParams {
            gaussian_count: 1_000,
            light_dir: Vec3::new(0.6, 0.7, 0.4).normalize(),
            base_color: Vec3::new(0.55, 0.62, 0.75),
            ..default()
        }
    ));

    // Directional light to illuminate the scene
    commands.spawn((
        DirectionalLight::default(),
        Transform::from_xyz(2.0, 4.0, 2.0)
            .looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Make CPU-side sort trigger more responsive
    commands.insert_resource(bevy_gaussian_splatting::sort::SortConfig { period_ms: 16 });
}







/// Set up the UI overlay showing controls and status
fn setup_ui(
    mut commands: Commands
) {

    commands.spawn((
        Text::new("Loading mesh..."),
        TextFont {
            font_size: 24.0,
            ..default()
        },
        TextColor(Color::srgb(1.0, 1.0, 1.0)),
        Node {
            position_type:  PositionType::Absolute,
            top:            Px(10.0),
            left:           Px(10.0),
            ..default()
        },
        InfoText,
    ));
}







/// Load the mesh asset and prepare for conversion
fn load_mesh(
    mut commands:   Commands,
    assets:         Res<AssetServer>
) {

    let scene: Handle<Scene> = assets.load(MESH_PATH.to_string() + "#Scene0");

    commands.insert_resource(PendingMeshScene(scene.clone()));

    commands.spawn((
        SceneRoot(scene),
        Transform::default(),
        Visibility::Visible,
        MeshToGaussian {
            mode:               MeshToGaussianMode::TrianglesOneToOne,
            surfel_thickness:   0.01,
            hide_source_mesh:   true,
            realtime:           false,
        },
    ));
}







// === Interactive Controls ===

/// Camera orbit controls using WASD keys and QE for zoom
fn camera_controls(
    mut camera_query:   Query<&mut Transform, With<GaussianCamera>>,
    input:              Res<ButtonInput<KeyCode>>,
    time:               Res<Time>,
) {

    let Ok(mut camera_transform) = camera_query.single_mut() else { return };
    
    const ROTATION_SPEED: f32   = 1.5; // radians per second
    const ZOOM_SPEED: f32       = 5.0;     // units per second
    
    let mut distance = camera_transform.translation.length();
    

    // Convert current position to spherical coordinates
    let current_pos     = camera_transform.translation;
    let mut azimuth     = current_pos.z.atan2(current_pos.x);   // rotation around Y-axis
    let mut elevation   = (current_pos.y / distance).asin();  // angle from XZ-plane
    

    // Handle rotation input
    if input.pressed(KeyCode::KeyD) {
        azimuth += ROTATION_SPEED * time.delta_secs();
    }
    if input.pressed(KeyCode::KeyA) {
        azimuth -= ROTATION_SPEED * time.delta_secs();
    }
    if input.pressed(KeyCode::KeyW) {
        elevation += ROTATION_SPEED * time.delta_secs();
    }
    if input.pressed(KeyCode::KeyS) {
        elevation -= ROTATION_SPEED * time.delta_secs();
    }
    
    // Handle zoom input
    if input.pressed(KeyCode::KeyE) || input.pressed(KeyCode::NumpadAdd) {
        distance -= ZOOM_SPEED * time.delta_secs();
    }
    if input.pressed(KeyCode::KeyQ) || input.pressed(KeyCode::NumpadSubtract) {
        distance += ZOOM_SPEED * time.delta_secs();
    }
    
    // Clamp values to reasonable bounds
    elevation   = elevation.clamp(-std::f32::consts::FRAC_PI_2 + 0.1, std::f32::consts::FRAC_PI_2 - 0.1);
    distance    = distance.clamp(1.0, 50.0);
    
    // Convert back to Cartesian coordinates
    let new_position = Vec3::new(
        distance * elevation.cos() * azimuth.cos(),
        distance * elevation.sin(),
        distance * elevation.cos() * azimuth.sin(),
    );
    
    // Update camera transform
    camera_transform.translation = new_position;
    camera_transform.look_at(Vec3::ZERO, Vec3::Y);
}







/// Update the UI text showing controls and current state
fn update_info_text(
    mut text_query: Query<&mut Text, With<InfoText>>,
) {

    let Ok(mut text) = text_query.single_mut() else { return };

    **text = format!(
        "Mesh to Gaussian Splats Demo\n\
        \n\
        Camera Controls:\n\
        • WASD: Orbit camera\n\
        • Q/E: Zoom in/out\n\
    ",

    );
}
