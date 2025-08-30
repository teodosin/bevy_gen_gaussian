// Gaussian module - pure functions for creating and manipulating Gaussian clouds

pub mod cpu_mesh_to_gaussians;
pub mod gpu_mesh_to_gaussians;

pub mod cpu_transform;
pub mod settings;

// Re-export the main public API
pub use cpu_mesh_to_gaussians::*;
pub use gpu_mesh_to_gaussians::*;

pub use cpu_transform::*;
pub use settings::*;




use bevy::prelude::*;





/// Component to mark and configure mesh to Gaussian conversion.
/// This component can be attached to an entity with a Mesh to generate Gaussians from it.
///
/// The conversion can be done once or in real-time as the mesh changes.
/// A separate system will handle the conversion then based on these settings.
#[derive(Component, Debug, Clone, Reflect)]
pub struct MeshToGaussian {
    pub mode: MeshToGaussianMode,
    pub surfel_thickness: f32,
    pub hide_source_mesh: bool,
    pub realtime: bool,
}

impl Default for MeshToGaussian {
    fn default() -> Self {
        Self {
            mode: MeshToGaussianMode::TrianglesOneToOne,
            surfel_thickness: 0.01,
            hide_source_mesh: true,
            realtime: false,
        }
    }
}



#[derive(Debug, Clone, Copy, PartialEq, Eq, Reflect)]
pub enum MeshToGaussianMode {
    /// Generates one gaussian splat for each triangle in the mesh.
    TrianglesOneToOne,
    // Future modes like `Vertices1to1` or `AreaSampled` can be added here.
}





// --- GPU Conversion Plugin and Components ---

/// Plugin for the GPU-accelerated mesh to gaussian conversion pipeline.
pub struct GenGaussianGpuPlugin;

impl Plugin for GenGaussianGpuPlugin {
    fn build(&self, app: &mut App) {
        fn log_gpu_plugin_startup() {
            info!("GenGaussianGpuPlugin Startup: registering TriToSplatPlugin");
        }
        fn log_added_mesh_to_gaussian(
            q: Query<(Entity, Option<&Name>, &MeshToGaussian), Added<MeshToGaussian>>,
        ) {
            for (e, name, cfg) in &q {
                info!(
                    "Added MeshToGaussian on entity {:?} ({}), mode={:?}, hide_source_mesh={}, realtime={}",
                    e,
                    name.map(|n| n.as_str()).unwrap_or("<unnamed>"),
                    cfg.mode,
                    cfg.hide_source_mesh,
                    cfg.realtime
                );
            }
        }
        // When MeshToGaussian is added, create a Gaussian cloud entity as a child and hide the source if requested.
        fn setup_mesh_to_gaussian(
            mut commands: Commands,
            mut clouds: ResMut<Assets<bevy_gaussian_splatting::PlanarGaussian3d>>,
            added: Query<(Entity, Option<&Name>, &MeshToGaussian), Added<MeshToGaussian>>,
        ) {
            for (entity, name, cfg) in &added {
                // For now, allocate a fixed-size cloud; compute pass will write into it.
                let default_count = 1024usize;
                let cloud = bevy_gaussian_splatting::random_gaussians_3d(default_count);
                let handle = clouds.add(cloud);

                info!(
                    "setup_mesh_to_gaussian: spawning cloud ({} gaussians) for entity {:?} ({})",
                    default_count,
                    entity,
                    name.map(|n| n.as_str()).unwrap_or("<unnamed>")
                );

                // Spawn as a top-level entity so it's not affected by source visibility.
                commands.spawn((
                    bevy_gaussian_splatting::PlanarGaussian3dHandle(handle.clone()),
                    bevy_gaussian_splatting::CloudSettings::default(),
                    Name::new("GeneratedGaussianCloud"),
                    Transform::default(),
                    Visibility::Visible,
                ));

                // Attach a backreference to the source so we can fill this cloud once the mesh is available
                commands.entity(entity).insert(MeshToGaussianCloud(handle));

                if cfg.hide_source_mesh {
                    // TODO: hide only mesh-bearing descendants of `entity` instead of the whole root.
                    info!("setup_mesh_to_gaussian: hide_source_mesh is true (not yet hiding GLTF descendants)");
                }
            }
        }
    // moved to module scope below

        /// Build CPU-side inputs (positions/indices) for the compute path and attach them to the cloud entity.
        /// This does NOT convert to gaussians on CPU; it only prepares data for the compute shader.
        fn build_tri_to_splat_cpu_inputs(
            mut commands: Commands,
            sources: Query<(&MeshToGaussian, &MeshToGaussianCloud, Entity)>,
            clouds_q: Query<(Entity, &bevy_gaussian_splatting::PlanarGaussian3dHandle, Option<&Name>), Without<CloudOf>>, 
            children_q: Query<&Children>,
            mesh_q: Query<&Mesh3d>, 
            meshes: Res<Assets<Mesh>>,
        ) {
            // Helper: search descendants for first Mesh3d
            fn find_desc_mesh(
                root: Entity,
                children_q: &Query<&Children>,
                mesh_q: &Query<&Mesh3d>,
            ) -> Option<Handle<Mesh>> {
                let mut stack = vec![root];
                while let Some(e) = stack.pop() {
                    if let Ok(m) = mesh_q.get(e) { return Some(m.0.clone()); }
                    if let Ok(children) = children_q.get(e) {
                        for child in children.iter() {
                            stack.push(child.clone());
                        }
                    }
                }
                None
            }

            // For each source→cloud pair, locate the cloud entity and attach CloudOf + TriToSplatCpuInput
            for (_cfg, cloud_handle, source_e) in &sources {
                // Find the cloud entity by handle
                let mut cloud_entity_opt: Option<(Entity, Option<&Name>)> = None;
                for (e, h, name_opt) in &clouds_q {
                    if h.0 == cloud_handle.0 { cloud_entity_opt = Some((e, name_opt)); break; }
                }
                let Some((cloud_e, name_opt)) = cloud_entity_opt else { continue };

                // Already linked? Skip (CloudOf present)
                // clouds_q ensured Without<CloudOf>, so we are first-time linking

                // Find a mesh under the source
                let Some(mesh_h) = find_desc_mesh(source_e, &children_q, &mesh_q) else { continue };
                let Some(mesh) = meshes.get(&mesh_h) else { continue };

                // Read positions and indices on CPU to ship to GPU as input buffers
                use bevy::render::mesh::VertexAttributeValues;
                let Some(VertexAttributeValues::Float32x3(pos)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) else { continue };
                let positions: Vec<[f32; 4]> = pos.iter().map(|p| [p[0], p[1], p[2], 1.0]).collect();
                let indices: Vec<u32> = match mesh.indices() {
                    Some(bevy::render::mesh::Indices::U16(xs)) => xs.iter().map(|&i| i as u32).collect(),
                    Some(bevy::render::mesh::Indices::U32(xs)) => xs.clone(),
                    None => (0..positions.len() as u32).collect(),
                };

                let tri_count = (indices.len() / 3) as u32;
                info!(
                    "link cloud {:?} ({}): source={:?}, verts={}, tris={}",
                    cloud_e,
                    name_opt.map(|n| n.as_str()).unwrap_or("") ,
                    source_e,
                    positions.len(),
                    tri_count
                );

                // Attach link + CPU-side inputs on the cloud entity
                commands.entity(cloud_e).insert((
                    CloudOf(source_e),
                    crate::gaussian::gpu_mesh_to_gaussians::TriToSplatCpuInput {
                        positions,
                        indices,
                        tri_count,
                    },
                ));
            }
        }

        // Keep TriToSplatParams updated on cameras; pick a conservative upper bound from clouds
        fn update_tri_to_splat_params(
            mut commands: Commands,
            q_cloud_inputs: Query<&crate::gaussian::gpu_mesh_to_gaussians::TriToSplatCpuInput>,
            q_cameras: Query<Entity, With<Camera3d>>,
        ) {
            let mut max_gauss = 0u32;
            for input in &q_cloud_inputs { max_gauss = max_gauss.max(input.tri_count); }
            if max_gauss == 0 { return; }
            for cam in &q_cameras {
                commands.entity(cam).insert(crate::gaussian::gpu_mesh_to_gaussians::TriToSplatParams { gaussian_count: max_gauss });
            }
        }
        app.add_systems(Startup, log_gpu_plugin_startup);
        app.add_systems(Update, (log_added_mesh_to_gaussian, setup_mesh_to_gaussian));
        app.add_systems(Update, build_tri_to_splat_cpu_inputs);
        app.add_systems(Update, update_tri_to_splat_params);
        app.add_plugins(TriToSplatPlugin);
    }
}

/// Backreference from a source entity with MeshToGaussian to the spawned cloud asset handle
#[derive(Component, Clone)]
pub struct MeshToGaussianCloud(pub Handle<bevy_gaussian_splatting::PlanarGaussian3d>);

/// Marker to prevent reprocessing
#[derive(Component)]
pub struct ConvertedOnce;

/// Link the spawned cloud entity to its source (MeshToGaussian) entity.
#[derive(Component, Clone, Copy, Debug)]
pub struct CloudOf(pub Entity);