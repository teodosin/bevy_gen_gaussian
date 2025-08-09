use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderRef};
use bevy::render::storage::ShaderStorageBuffer;
use bevy::reflect::TypePath;
use super::extraction::SurfaceBuffer;

const SHADER_ASSET_PATH: &str = "shaders/voxel_billboard.wgsl";

#[derive(Component)]
pub struct VoxelBillboardMarker;

#[derive(Resource)]
pub struct VoxelBillboardAssets {
    pub mesh: Handle<Mesh>,
    pub material: Handle<VoxelBillboardMaterial>,
    pub color_buffer: Handle<ShaderStorageBuffer>,
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct VoxelBillboardMaterial {
    // Simple material for camera-facing billboards
}

impl Material for VoxelBillboardMaterial {
    fn vertex_shader() -> ShaderRef {
        SHADER_ASSET_PATH.into()
    }

    fn fragment_shader() -> ShaderRef {
        SHADER_ASSET_PATH.into()
    }
}

pub fn setup_instanced_billboards(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<VoxelBillboardMaterial>>,
    mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
) {
    // Create initial empty color buffer
    let initial_colors: Vec<[f32; 4]> = vec![[0.0, 0.0, 0.0, 1.0]; 1];
    let color_buffer = buffers.add(ShaderStorageBuffer::from(initial_colors));
    
    // Create billboard mesh (larger square for visibility)
    let mesh = meshes.add(Rectangle::new(2.0, 2.0));
    
    // Create simple material
    let material = materials.add(VoxelBillboardMaterial {
        // Simple material for testing
    });
    
    println!("Setup: Created billboard material and mesh assets");
    
    commands.insert_resource(VoxelBillboardAssets {
        mesh,
        material,
        color_buffer,
    });
}

pub fn manage_billboard_instances(
    mut commands: Commands,
    mut surface_buffer: ResMut<SurfaceBuffer>,
    billboard_assets: Res<VoxelBillboardAssets>,
    mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
    billboard_query: Query<Entity, With<VoxelBillboardMarker>>,
) {
    if !surface_buffer.dirty {
        return;
    }
    
    // Clear existing billboards
    for entity in billboard_query.iter() {
        commands.entity(entity).despawn();
    }
    
    if surface_buffer.instances.is_empty() {
        surface_buffer.dirty = false;
        return;
    }

    // Update color buffer with new data
    let colors: Vec<[f32; 4]> = surface_buffer.instances.iter()
        .map(|instance| [
            instance.color[0] as f32 / 255.0,
            instance.color[1] as f32 / 255.0,
            instance.color[2] as f32 / 255.0,
            instance.color[3] as f32 / 255.0,
        ])
        .collect();
    
    if let Some(buffer) = buffers.get_mut(&billboard_assets.color_buffer) {
        buffer.set_data(&colors);
    }
    
    // Spawn camera-facing billboard instances
    for (i, instance) in surface_buffer.instances.iter().enumerate() {
        commands.spawn((
            Mesh3d(billboard_assets.mesh.clone()),
            MeshMaterial3d(billboard_assets.material.clone()),
            Transform::from_translation(instance.pos),
            VoxelBillboardMarker,
        ));
        
        // Debug first few entities
        if i < 3 {
            println!("Spawned billboard {} at pos: {:?}", i, instance.pos);
        }
    }
    
    // Reset dirty flag and log result
    println!("Instanced billboards: spawned {} instances", surface_buffer.instances.len());
    surface_buffer.dirty = false;
}
