use bevy::prelude::*;
use super::extraction::SurfaceBuffer;

#[derive(Component)]
pub struct VoxelBillboard;

#[derive(Resource)]
pub struct BillboardMesh(pub Handle<Mesh>);

#[derive(Resource)]
pub struct BillboardMaterial(pub Handle<StandardMaterial>);

pub fn setup_billboard_rendering(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Create billboard mesh (small square)
    let mesh = meshes.add(Rectangle::new(0.6, 0.6));
    commands.insert_resource(BillboardMesh(mesh));
    
    // Create unlit material for billboards
    let material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        alpha_mode: AlphaMode::Blend,
        unlit: true,
        ..default()
    });
    commands.insert_resource(BillboardMaterial(material));
}

pub fn render_billboards_cpu(
    mut commands: Commands,
    surface_buffer: Res<SurfaceBuffer>,
    billboard_mesh: Res<BillboardMesh>,
    billboard_material: Res<BillboardMaterial>,
    billboard_query: Query<Entity, With<VoxelBillboard>>,
    camera_query: Query<&Transform, (With<Camera3d>, Without<VoxelBillboard>)>,
) {
    if !surface_buffer.dirty {
        return;
    }
    
    // Clear existing billboards
    for entity in billboard_query.iter() {
        commands.entity(entity).despawn();
    }
    
    // Get camera position for billboarding
    let Ok(camera_transform) = camera_query.single() else { return };
    let camera_pos = camera_transform.translation;
    
    // Spawn new billboards for each instance
    for instance in &surface_buffer.instances {
        let world_pos = instance.pos;
        
        // Calculate billboard orientation (face camera)
        let direction = (camera_pos - world_pos).normalize_or_zero();
        let up = Vec3::Y;
        let right = up.cross(direction).normalize_or_zero();
        let corrected_up = direction.cross(right);
        
        // Create rotation matrix and convert to quaternion
        let rotation_matrix = Mat3::from_cols(right, corrected_up, direction);
        let rotation = Quat::from_mat3(&rotation_matrix);
        
        // Convert color from u8 array to Color
        let color = Color::srgba_u8(
            instance.color[0],
            instance.color[1], 
            instance.color[2],
            instance.color[3]
        );
        
        // Create material with voxel color
        commands.spawn((
            Mesh3d(billboard_mesh.0.clone()),
            MeshMaterial3d(billboard_material.0.clone()),
            Transform::from_translation(world_pos)
                .with_rotation(rotation),
            VoxelBillboard,
        ));
    }
}
