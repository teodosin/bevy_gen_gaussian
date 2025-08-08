use bevy::prelude::*;
use bevy_gen_gaussian::{
    GenGaussianPlugin,
    EditOp, EditBatch, LastInstanceCount, Metrics,
    BrushSettings, BrushMode, apply_sphere_brush, generate_terrain,
};
use bevy_panorbit_camera::PanOrbitCamera;

#[derive(Component)]
struct InfoText;

#[derive(Resource)]
struct TerrainSeed(u32);

impl Default for TerrainSeed {
    fn default() -> Self {
        Self(42) // Default seed
    }
}

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window { 
                    title: "SDF Editing - Brush Sculpting".into(), 
                    ..default() 
                }),
                ..default()
            }),
            GenGaussianPlugin,
        ))
        .insert_resource(LastInstanceCount::default())
        .insert_resource(BrushSettings::default())
        .insert_resource(TerrainSeed::default())
        .add_systems(Startup, (setup, setup_ui))
        .add_systems(Update, (
            sdf_input_system, 
            procedural_generation, 
            brush_ui_system,
            update_info_text
        ))
        .run();
}

fn setup(mut commands: Commands) {
    // UI camera for the overlay text
    commands.spawn(Camera2d);
    
    // 3D camera for voxel rendering
    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: 45.0_f32.to_radians(),
            ..default()
        }),
        PanOrbitCamera {
            focus: Vec3::new(16.0, 16.0, 16.0),
            yaw: Some(0.8),
            pitch: Some(0.4),
            radius: Some(64.0),
            button_orbit: MouseButton::Left,
            button_pan: MouseButton::Right,
            touch_enabled: true,
            trackpad_pinch_to_zoom_enabled: true,
            trackpad_sensitivity: 1.0,
            orbit_sensitivity: 1.0,
            pan_sensitivity: 0.001,
            zoom_sensitivity: 0.8,
            allow_upside_down: true,
            ..default()
        },
    ));

    // Lighting
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
        Text::new("SDF Editing Demo"),
        TextFont {
            font_size: 12.0,
            ..default()
        },
        TextColor(Color::srgb(1.0, 1.0, 1.0)),
        InfoText,
    ));
}

fn procedural_generation(
    mut batch: ResMut<EditBatch>, 
    mut terrain_seed: ResMut<TerrainSeed>,
    brush: Res<BrushSettings>,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
) {
    // Generate new terrain with a random seed - respects brush mode
    if keys.just_pressed(KeyCode::KeyT) {
        // Update seed to get different terrain each time
        terrain_seed.0 = terrain_seed.0.wrapping_mul(1664525).wrapping_add(1013904223);
        let seed = terrain_seed.0 as f32;
        
        // Generate terrain but apply based on brush mode
        let mut temp_batch = EditBatch::default();
        generate_terrain(&mut temp_batch, seed, (32, 24, 32));
        
        // Convert operations based on brush mode
        for op in temp_batch.ops {
            match op {
                EditOp::Set(pos) => {
                    match brush.mode {
                        BrushMode::Add => batch.ops.push(EditOp::Set(pos)),
                        BrushMode::Subtract => batch.ops.push(EditOp::Clear(pos)),
                        BrushMode::Paint => batch.ops.push(EditOp::Set(pos)), // Same as add for now
                    }
                }
                EditOp::Clear(pos) => {
                    // Clear operations are inverted
                    match brush.mode {
                        BrushMode::Add => batch.ops.push(EditOp::Clear(pos)),
                        BrushMode::Subtract => batch.ops.push(EditOp::Set(pos)),
                        BrushMode::Paint => batch.ops.push(EditOp::Clear(pos)),
                    }
                }
            }
        }
    }

    // Generate animated sphere that moves around - respects brush mode
    if keys.just_pressed(KeyCode::KeyS) {
        let t = time.elapsed_secs();
        let center = Vec3::new(
            16.0 + (t * 0.5).sin() * 8.0,
            12.0 + (t * 0.3).cos() * 4.0,
            16.0 + (t * 0.7).cos() * 8.0,
        );
        let radius = 4.0 + (t * 0.8).sin().abs() * 2.0;
        
        apply_sphere_brush(&mut batch, center, radius, brush.mode);
    }

    // Generate multiple spheres pattern with randomized positions - respects brush mode
    if keys.just_pressed(KeyCode::KeyM) {
        // Use current time as additional randomness
        let time_seed = (time.elapsed_secs() * 1000.0) as u32;
        let mut rng_state = terrain_seed.0.wrapping_add(time_seed);
        
        // Generate 5-8 random spheres
        let sphere_count = 5 + (rng_state % 4) as usize;
        
        for _ in 0..sphere_count {
            // Simple LCG for random numbers
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let x = (rng_state % 24) as f32 + 4.0;
            
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let y = (rng_state % 16) as f32 + 4.0;
            
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let z = (rng_state % 24) as f32 + 4.0;
            
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let radius = ((rng_state % 200) as f32 / 100.0) + 2.0; // radius 2-4
            
            let center = Vec3::new(x, y, z);
            apply_sphere_brush(&mut batch, center, radius, brush.mode);
        }
    }

    // Clear all
    if keys.just_pressed(KeyCode::KeyC) {
        for x in 0..32 {
            for y in 0..32 {
                for z in 0..32 {
                    batch.ops.push(EditOp::Clear(IVec3::new(x, y, z)));
                }
            }
        }
    }
}

fn sdf_input_system(
    mut batch: ResMut<EditBatch>,
    mut brush: ResMut<BrushSettings>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    camera_query: Query<(&Camera, &GlobalTransform), (With<Camera3d>, Without<Camera2d>)>,
    windows: Query<&Window>,
) {
    // Brush size adjustment - use +/- keys which are more intuitive
    if keys.just_pressed(KeyCode::Minus) {
        brush.radius = (brush.radius - 0.5).max(0.5);
        println!("Brush radius: {:.1}", brush.radius);
    }
    if keys.just_pressed(KeyCode::Equal) { // This is the + key without shift
        brush.radius = (brush.radius + 0.5).min(10.0);
        println!("Brush radius: {:.1}", brush.radius);
    }

    // Brush mode switching
    if keys.just_pressed(KeyCode::Digit1) {
        brush.mode = BrushMode::Add;
    }
    if keys.just_pressed(KeyCode::Digit2) {
        brush.mode = BrushMode::Subtract;
    }
    if keys.just_pressed(KeyCode::Digit3) {
        brush.mode = BrushMode::Paint;
    }

    // Mouse-based SDF brush editing with proper raycasting
    if mouse.pressed(MouseButton::Left) && keys.pressed(KeyCode::ShiftLeft) {
        let Ok((camera, camera_transform)) = camera_query.single() else { 
            println!("Failed to get camera");
            return;
        };
        let Ok(window) = windows.single() else { 
            println!("Failed to get window");
            return;
        };
        
        if let Some(cursor_pos) = window.cursor_position() {
            println!("Cursor position: {:?}", cursor_pos);
            
            // Cast a ray from the camera through the cursor position
            let Ok(ray) = camera.viewport_to_world(camera_transform, cursor_pos) else { 
                println!("Failed to cast ray");
                return;
            };
            
            println!("Ray origin: {:?}, direction: {:?}", ray.origin, ray.direction);
            
            // Find intersection with a horizontal plane at y=12 (middle of our voxel space)
            let plane_y = 12.0;
            
            // Calculate intersection with the plane
            let ray_dir_y = ray.direction.y;
            println!("Ray Y direction: {}", ray_dir_y);
            
            if ray_dir_y.abs() > 0.001 { // Avoid division by zero
                let t = (plane_y - ray.origin.y) / ray_dir_y;
                println!("Intersection parameter t: {}", t);
                
                if t > 0.0 { // Ray goes towards the plane
                    let intersection_point = ray.origin + ray.direction * t;
                    println!("Intersection point: {:?}", intersection_point);
                    
                    // Only apply brush if intersection is within our voxel bounds
                    if intersection_point.x >= 0.0 && intersection_point.x < 32.0 &&
                       intersection_point.z >= 0.0 && intersection_point.z < 32.0 {
                        println!("Applying brush at: {:?}", intersection_point);
                        apply_sphere_brush(&mut batch, intersection_point, brush.radius, brush.mode);
                    } else {
                        println!("Intersection out of bounds: {:?}", intersection_point);
                    }
                } else {
                    println!("Ray pointing away from plane (t = {})", t);
                }
            } else {
                println!("Ray parallel to plane (ray_dir_y = {})", ray_dir_y);
            }
        } else {
            println!("No cursor position");
        }
    }

    // Alternative: Hold Ctrl+Click to apply brush at a fixed depth
    if mouse.just_pressed(MouseButton::Left) && keys.pressed(KeyCode::ControlLeft) {
        let Ok((camera, camera_transform)) = camera_query.single() else { return };
        let Ok(window) = windows.single() else { return };
        
        if let Some(cursor_pos) = window.cursor_position() {
            let Ok(ray) = camera.viewport_to_world(camera_transform, cursor_pos) else { return };
            
            // Apply brush at a fixed distance from camera (useful for building in 3D)
            let distance = 20.0;
            let brush_center = ray.origin + ray.direction * distance;
            
            // Clamp to voxel bounds
            let clamped_center = Vec3::new(
                brush_center.x.clamp(0.0, 31.0),
                brush_center.y.clamp(0.0, 31.0),
                brush_center.z.clamp(0.0, 31.0),
            );
            
            apply_sphere_brush(&mut batch, clamped_center, brush.radius, brush.mode);
        }
    }

    // Keyboard-based brush strokes for testing
    if keys.just_pressed(KeyCode::Space) {
        let center = Vec3::new(16.0, 12.0, 16.0);
        apply_sphere_brush(&mut batch, center, brush.radius, brush.mode);
    }
}

fn brush_ui_system(
    brush: Res<BrushSettings>,
) {
    // This system could be expanded to show brush preview in the 3D world
    // For now, it's just a placeholder for brush state management
    if brush.is_changed() {
        println!("Brush: radius={:.1}, mode={:?}", brush.radius, brush.mode);
    }
}

fn update_info_text(
    mut text_query: Query<&mut Text, With<InfoText>>,
    metrics: Res<Metrics>,
    brush: Res<BrushSettings>,
    time: Res<Time>,
) {
    let Ok(mut text) = text_query.single_mut() else { return };
    
    **text = format!("
        SDF Editing Controls:
        • T: Generate random terrain (respects brush mode!)
        • S: Generate animated moving sphere (respects brush mode!)
        • M: Generate random spheres pattern (respects brush mode!)
        • C: Clear all
        • SPACE: Apply brush at center
        • Shift+Drag: Apply brush on horizontal plane
        • Ctrl+Click: Apply brush at fixed distance
        • - / +: Decrease/increase brush size
        • 1/2/3: Add/Subtract/Paint modes
        
        Brush Settings:
        • Radius: {:.1}
        • Mode: {:?}
        
        Camera:
        • Trackpad: Natural gestures
        • Mouse: Left=orbit, Right=pan, Scroll=zoom
        
        Stats:
        • FPS: {:.1}
        • Voxels: {}
        • Instances: {}
        • Edits: {}
        • Time: {:.2}s",
        brush.radius,
        brush.mode,
        metrics.fps,
        metrics.voxel_count,
        metrics.instance_count,
        metrics.edits_applied,
        time.elapsed_secs()
    );
}
