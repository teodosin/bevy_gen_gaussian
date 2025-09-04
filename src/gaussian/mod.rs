// Gaussian module - pure functions for creating and manipulating Gaussian clouds
pub mod cpu_mesh_to_gaussians;
pub mod gpu_mesh_to_gaussians;
pub mod settings;

// Re-export the main public API
pub use cpu_mesh_to_gaussians::*;
pub use gpu_mesh_to_gaussians::*;
pub use settings::*;

use bevy::{
    prelude::{Mesh3d, *},
    render::mesh::{Indices, VertexAttributeValues},
};

use bevy_gaussian_splatting::{
    gaussian::f32::{PositionVisibility, Rotation, ScaleOpacity},
    SphericalHarmonicCoefficients,
    sort::SortMode,
};







/// Component to mark and configure mesh to Gaussian conversion.
#[derive(Component, Debug, Clone, Reflect)]
pub struct MeshToGaussian {
    pub mode:               MeshToGaussianMode,
    pub surfel_thickness:   f32,
    pub hide_source_mesh:   bool,
    pub realtime:           bool,
}

impl Default for MeshToGaussian {
    fn default() -> Self {
        Self {
            mode:               MeshToGaussianMode::TrianglesOneToOne,
            surfel_thickness:   0.01,
            hide_source_mesh:   true,
            realtime:           false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Reflect)]
pub enum MeshToGaussianMode {
    /// Generates one gaussian splat for each triangle in the mesh.
    TrianglesOneToOne,
}







// --- GPU Conversion Plugin and Components ---

/// Plugin for the GPU-accelerated mesh to gaussian conversion pipeline.
pub struct GenGaussianGpuPlugin;

impl Plugin for GenGaussianGpuPlugin {

    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                process_new_meshes_for_gpu_conversion,
                update_tri_to_splat_params,
                debug_entities,
            ),
        );
        app.add_plugins(TriToSplatPlugin);
    }
}







/// Backreference from a source entity to the spawned cloud asset handle.
#[derive(Component, Clone)]
pub struct MeshToGaussianCloud(pub Handle<bevy_gaussian_splatting::PlanarGaussian3d>);


/// Marker to prevent reprocessing a mesh every frame if `realtime` is false.
#[derive(Component)]
pub struct ConvertedOnce;


/// Component on the cloud entity that links it back to its source entity.
#[derive(Component, Clone, Copy, Debug)]
pub struct CloudOf(pub Entity);







/// Finds entities with `MeshToGaussian`, waits for their mesh to load, then creates a correctly
/// sized and positioned Gaussian cloud.
fn process_new_meshes_for_gpu_conversion(
    mut commands:       Commands,
    mut clouds:         ResMut<Assets<bevy_gaussian_splatting::PlanarGaussian3d>>,
    meshes:             Res<Assets<Mesh>>,
    mut visibility_q:   Query<&mut Visibility>,
    source_q:           Query<(Entity, &MeshToGaussian), Without<ConvertedOnce>>,
    children_q:         Query<&Children>,
    mesh_3d_q:          Query<&Mesh3d>,
    
    transform_q:        Query<&GlobalTransform> // Query for transforms to correctly position the cloud.
) {


    // Helper now returns the mesh handle AND its global transform.
    fn find_descendant_mesh_with_transform(
        root:           Entity,
        children_q:     &Query<&Children>,
        mesh_3d_q:      &Query<&Mesh3d>,
        transform_q:    &Query<&GlobalTransform>,
    ) -> Option<(Handle<Mesh>, GlobalTransform)> {

        let mut stack = vec![root];
        
        while let Some(entity) = stack.pop() {
            if let Ok(mesh_3d) = mesh_3d_q.get(entity) {
                if let Ok(transform) = transform_q.get(entity) {
                    return Some((mesh_3d.0.clone(), *transform));
                }
            }
            if let Ok(children) = children_q.get(entity) {
                stack.extend(children.iter());
            }
        }
        None
    }



    for (source_entity, config) in &source_q {

        // Find the mesh and its transform.
        let Some((mesh_handle, mesh_transform)) = find_descendant_mesh_with_transform(
            source_entity,
            &children_q,
            &mesh_3d_q,
            &transform_q
        ) else {
            continue;
        };

        let Some(mesh) = meshes.get(&mesh_handle) else {
            continue;
        };

        let Some(VertexAttributeValues::Float32x3(pos)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) else {
            // TODO: Use change detection instead
             if !config.realtime { 
                commands
                    .entity(source_entity)
                    .insert(ConvertedOnce);
                }

             continue;
        };

        let positions: Vec<[f32; 4]> = pos
            .iter()
            .map(|p| [p[0], p[1], p[2], 1.0])
            .collect();

        let indices: Vec<u32> = match mesh.indices() {
            Some(Indices::U16(xs))  => xs.iter().map(|&i| i as u32).collect(),
            Some(Indices::U32(xs))  => xs.clone(),
            None                    => (0..positions.len() as u32).collect(),
        };


        let tri_count = (indices.len() / 3) as u32;
        if tri_count == 0 {
            if !config.realtime { commands.entity(source_entity).insert(ConvertedOnce); }
            continue;
        }

        info!("Processing mesh for {:?}: found {} triangles.", source_entity, tri_count);


        let zero_pv     = PositionVisibility            { position:     [0.0; 3], visibility: 0.0 };
        let zero_sh     = SphericalHarmonicCoefficients { coefficients: [0.0; 48] };
        let zero_rot    = Rotation                      { rotation:     [0.0; 4] };
        let zero_so     = ScaleOpacity                  { scale:        [0.0; 3], opacity: 0.0 };

        let cloud_asset = bevy_gaussian_splatting::PlanarGaussian3d {
            position_visibility:    vec![zero_pv;   tri_count as usize],
            spherical_harmonic:     vec![zero_sh;   tri_count as usize],
            rotation:               vec![zero_rot;  tri_count as usize],
            scale_opacity:          vec![zero_so;   tri_count as usize],
        };

        let cloud_handle = clouds.add(cloud_asset);


        // IMPORTANT
        // GPU conversion is currently not functional, because bevy_gaussian_splatting's
        // Radix sort implementation is broken. Rayon CPU sorting is used instead, and 
        // that requires the transforms of the splats to be known by the CPU entity. Since
        // we don't read back these transforms from our compute shader, the Rayon sorter
        // can't sort them. The following block sets the positions manually on the CPU and
        // is only here to demonstrate this issue. Once Radix sorting is working, the data
        // won't have to leave the GPU.
        if let Some(cloud) = clouds.get_mut(&cloud_handle) {
            for (i, tri) in indices.chunks(3).enumerate() {
                if tri.len() < 3 { break; }
                let p0  = pos[tri[0] as usize];
                let p1  = pos[tri[1] as usize];
                let p2  = pos[tri[2] as usize];
                let centroid = [
                    (p0[0] + p1[0] + p2[0]) / 3.0,
                    (p0[1] + p1[1] + p2[1]) / 3.0,
                    (p0[2] + p1[2] + p2[2]) / 3.0,
                ];
                if let Some(pv) = cloud.position_visibility.get_mut(i) {
                    pv.position = centroid;
                    pv.visibility = 1.0;
                }
            }
        }
        // --- end demonstration ---



        // Spawn the cloud entity
        commands.spawn((
            bevy_gaussian_splatting::PlanarGaussian3dHandle(cloud_handle.clone()),
            bevy_gaussian_splatting::CloudSettings {
                sort_mode: SortMode::Rayon,
                ..Default::default()
            },
            Name::new("GeneratedGaussianCloud"),
            CloudOf(source_entity),
            gpu_mesh_to_gaussians::TriToSplatCpuInput {
                positions,
                indices,
                tri_count,
            },
            // Apply the captured transform of the original mesh.
            mesh_transform,
            Visibility::Visible,
        ));


        if config.hide_source_mesh {
            if let Ok(mut visibility) = visibility_q.get_mut(source_entity) {
                *visibility = Visibility::Hidden;
                info!("Hid source mesh entity {:?}", source_entity);
            }
        }

        // TODO: Somehow implement change detection instead
        commands
            .entity(source_entity)
            .insert(MeshToGaussianCloud(cloud_handle));

        if !config.realtime {
            commands
                .entity(source_entity)
                .insert(ConvertedOnce);
        }
    }
}







/// Keep TriToSplatParams updated on cameras.
fn update_tri_to_splat_params(
    mut commands:       Commands,
    q_cloud_inputs:     Query<&gpu_mesh_to_gaussians::TriToSplatCpuInput>,
    q_cameras:          Query<Entity, With<Camera3d>>,
) {

    let input_count = q_cloud_inputs.iter().count();

    bevy::log::info!("update_tri_to_splat_params: found {} cloud inputs", input_count);
    
    let mut max_gauss = 0u32;

    for input in &q_cloud_inputs {
        max_gauss = max_gauss.max(input.tri_count);
    }

    if max_gauss == 0 { 
        bevy::log::info!("update_tri_to_splat_params: no gaussians to process");
        return; 
    }

    let camera_count = q_cameras.iter().count();
    bevy::log::info!("update_tri_to_splat_params: updating {} cameras with max_gauss={}", camera_count, max_gauss);

    for cam in &q_cameras {
        commands.entity(cam).insert(gpu_mesh_to_gaussians::TriToSplatParams {
            gaussian_count: max_gauss,
        });
    }
}







/// Debug system to track what entities exist and their components
fn debug_entities(
    q_clouds:           Query<Entity, With<bevy_gaussian_splatting::PlanarGaussian3dHandle>>,
    q_inputs:           Query<Entity, With<gpu_mesh_to_gaussians::TriToSplatCpuInput>>,
    q_mesh_to_gauss:    Query<Entity, With<MeshToGaussian>>,
) {

    let cloud_count             = q_clouds.iter().count();
    let input_count             = q_inputs.iter().count(); 
    let mesh_to_gauss_count     = q_mesh_to_gauss.iter().count();
    
    // Only log periodically to avoid spam
    static mut FRAME_COUNT: u32 = 0;
    unsafe {
        FRAME_COUNT += 1;
        if FRAME_COUNT % 60 == 0 {
            bevy::log::info!("DEBUG: clouds={}, inputs={}, mesh_to_gauss={}", 
                cloud_count, input_count, mesh_to_gauss_count);
        }
    }
}