use bevy::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(bevy_gen_gaussian::GenGaussianPlugin)

        .add_systems(Startup,
            (
                setup_scene,
                setup_ui,
            )
        )

        .run();
}

fn setup_scene(
    mut commands: Commands,
) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
        GlobalTransform::default(),
    ));
}

fn setup_ui(
    mut commands: Commands,
) {
    commands.spawn((
        Camera2d,
        Camera { order: 10, ..default() },
    ));
}