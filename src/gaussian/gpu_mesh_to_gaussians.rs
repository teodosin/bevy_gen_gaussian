use bevy::{
    prelude::*,
        render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_graph::{Node, NodeRunError, RenderGraphApp, RenderGraphContext, RenderLabel},
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingType,
            Buffer, BufferBindingType, BufferInitDescriptor, BufferSize, BufferUsages,
            CachedComputePipelineId, CachedPipelineState, ComputePassDescriptor, ComputePipelineDescriptor,
            PipelineCache, ShaderDefVal, ShaderStages,
        },
        renderer::{RenderContext, RenderDevice},
        Render, RenderApp, RenderSet,
    },
};
use bevy_gaussian_splatting::{CloudSettings, Gaussian3d, PlanarGaussian3d, PlanarGaussian3dHandle, RasterizeMode};
use bevy_gaussian_splatting::render::{CloudPipeline, GaussianUniformBindGroups};
use bevy_gaussian_splatting::PlanarStorageBindGroup;
use bevy::render::extract_component::DynamicUniformIndex;
use bevy_gaussian_splatting::render::CloudUniform;

// Reuse canonical component/type from gaussian::mod.rs (avoid duplicate defs)
use super::{MeshToGaussian, MeshToGaussianMode};

pub struct GpuMeshToGaussiansPlugin;

impl Plugin for GpuMeshToGaussiansPlugin {
    fn build(&self, app: &mut App) {
    app.add_plugins(ExtractComponentPlugin::<MeshCpuBuffers>::default());
        app.add_systems(Update, (setup_mesh_to_gaussian_conversion, attach_cpu_buffers_when_ready));

        // Render-world setup
        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_systems(Render, (queue_tri_to_splat_bind_groups.in_set(RenderSet::Queue),));

        // Add a minimal compute node that runs before sorting/draw
        #[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
        struct TriToSplatLabel;
        render_app.add_render_graph_node::<TriToSplatNode>(bevy::core_pipeline::core_3d::graph::Core3d, TriToSplatLabel);
        // Ensure our compute runs after prepasses but before the main pass
        render_app.add_render_graph_edges(
            bevy::core_pipeline::core_3d::graph::Core3d,
            (
                bevy::core_pipeline::core_3d::graph::Node3d::LatePrepass,
                TriToSplatLabel,
                bevy::core_pipeline::core_3d::graph::Node3d::StartMainPass,
            ),
        );
    }

    fn finish(&self, app: &mut App) {
        // Initialize compute pipeline after renderer is fully set up
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.init_resource::<TriToSplatPipeline>();
        }
    }
}

// App-world temporary marker to track which entity we actually convert and its mesh handle
#[derive(Component)]
struct MeshConversionSource {
    _mesh_entity: Entity,
    mesh_handle: Handle<Mesh>,
}

#[derive(Component, Clone)]
struct MeshCpuBuffers {
    positions: Vec<[f32; 3]>,
    indices: Vec<u32>,
    thickness: f32,
    world_from_mesh: [[f32; 4]; 4],
}

impl ExtractComponent for MeshCpuBuffers {
    type QueryData = &'static MeshCpuBuffers;
    type QueryFilter = ();
    type Out = Self;
    fn extract_component(item: bevy::ecs::query::QueryItem<'_, Self::QueryData>) -> Option<Self::Out> {
        Some(item.clone())
    }
}

fn setup_mesh_to_gaussian_conversion(
    mut commands: Commands,
    meshes: Res<Assets<Mesh>>,
    mut planar_gaussians: ResMut<Assets<PlanarGaussian3d>>,
    children: Query<&Children>,
    mesh_on_entity: Query<&Mesh3d>,
    transforms: Query<&GlobalTransform>,
    query: Query<(Entity, &MeshToGaussian), (With<MeshToGaussian>, Without<PlanarGaussian3dHandle>)>,
) {
    // Iterate entities that want conversion but don't yet have a cloud
    for (root, config) in &query {
        if config.mode != MeshToGaussianMode::TrianglesOneToOne {
            continue;
        }

        // Find first descendant with Mesh3d
        let mut found: Option<(Entity, Handle<Mesh>)> = None;
        let mut stack = vec![root];
        while let Some(ent) = stack.pop() {
            if let Ok(mesh3d) = mesh_on_entity.get(ent) {
                found = Some((ent, mesh3d.0.clone()));
                break;
            }
            if let Ok(kids) = children.get(ent) {
                stack.extend_from_slice(&kids[..]);
            }
        }
        let Some((mesh_entity, mesh_handle)) = found else {
            // Scene not instantiated yet; try again next frame
            continue;
        };

        // Count triangles and create a cloud of that capacity if mesh asset is loaded
        let Some(mesh) = meshes.get(&mesh_handle) else {
            // Hide the source mesh early if requested, then wait for asset
            if config.hide_source_mesh {
                commands.entity(mesh_entity).insert(Visibility::Hidden);
            }
            // Track the source handle so we can complete setup later
            commands.entity(root).insert(MeshConversionSource { _mesh_entity: mesh_entity, mesh_handle: mesh_handle.clone() });
            continue;
        };
        let tri_count = match mesh.indices() { Some(ix) => ix.len() / 3, None => mesh.count_vertices() / 3 };
        let cloud = PlanarGaussian3d::from(vec![Gaussian3d::default(); tri_count]);
        let cloud_handle = planar_gaussians.add(cloud);

        // Attach cloud and settings on the root entity so renderer sees it; keep track of source mesh
        commands.entity(root).insert((
            PlanarGaussian3dHandle(cloud_handle.clone()),
            CloudSettings { aabb: true, rasterize_mode: RasterizeMode::Normal, ..default() },
            MeshConversionSource { _mesh_entity: mesh_entity, mesh_handle: mesh_handle.clone() },
        ));

        if config.hide_source_mesh {
            commands.entity(mesh_entity).insert(Visibility::Hidden);
        }

    // Prepare CPU compact buffers for upload once
    let positions: Vec<[f32; 3]> = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            Some(bevy::render::mesh::VertexAttributeValues::Float32x3(v)) => v.clone(),
            Some(bevy::render::mesh::VertexAttributeValues::Float32x4(v)) => v.iter().map(|p| [p[0], p[1], p[2]]).collect(),
            _ => Vec::new(),
        };
        let indices: Vec<u32> = match mesh.indices() {
            Some(bevy::render::mesh::Indices::U32(ix)) => ix.clone(),
            Some(bevy::render::mesh::Indices::U16(ix)) => ix.iter().map(|&x| x as u32).collect(),
            None => (0..positions.len() as u32).collect(),
        };
        if !positions.is_empty() && !indices.is_empty() {
            let world_from_mesh = transforms
                .get(mesh_entity)
                .map(|t| t.compute_matrix().to_cols_array_2d())
                .unwrap_or(Mat4::IDENTITY.to_cols_array_2d());
            commands.entity(root).insert(MeshCpuBuffers { positions, indices, thickness: config.surfel_thickness, world_from_mesh });
        }
    }
}

// Fallback: if mesh asset wasn't yet loaded at add-time, attach CPU buffers once it becomes available
fn attach_cpu_buffers_when_ready(
    mut commands: Commands,
    meshes: Res<Assets<Mesh>>,
    mut planar_gaussians: ResMut<Assets<PlanarGaussian3d>>,
    transforms: Query<&GlobalTransform>,
    pending: Query<(Entity, &MeshConversionSource, Option<&PlanarGaussian3dHandle>, Option<&MeshToGaussian>), Without<MeshCpuBuffers>>,
) {
    for (entity, src, maybe_cloud, maybe_cfg) in &pending {
        if let Some(mesh) = meshes.get(&src.mesh_handle) {
            let positions: Vec<[f32; 3]> = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
                Some(bevy::render::mesh::VertexAttributeValues::Float32x3(v)) => v.clone(),
                Some(bevy::render::mesh::VertexAttributeValues::Float32x4(v)) => v.iter().map(|p| [p[0], p[1], p[2]]).collect(),
                _ => Vec::new(),
            };
            let indices: Vec<u32> = match mesh.indices() {
                Some(bevy::render::mesh::Indices::U32(ix)) => ix.clone(),
                Some(bevy::render::mesh::Indices::U16(ix)) => ix.iter().map(|&x| x as u32).collect(),
                None => (0..positions.len() as u32).collect(),
            };
            if !positions.is_empty() && !indices.is_empty() {
                // Ensure a cloud exists; if not, create one sized to triangles
                if maybe_cloud.is_none() {
                    let tri_count = match mesh.indices() { Some(ix) => ix.len() / 3, None => mesh.count_vertices() / 3 };
                    let cloud = PlanarGaussian3d::from(vec![Gaussian3d::default(); tri_count]);
                    let handle = planar_gaussians.add(cloud);
                    commands.entity(entity).insert((
                        PlanarGaussian3dHandle(handle),
                        CloudSettings { aabb: true, rasterize_mode: RasterizeMode::Normal, ..default() },
                    ));
                }

                // Use the mesh entity's transform
                let world_from_mesh = transforms
                    .get(src._mesh_entity)
                    .map(|t| t.compute_matrix().to_cols_array_2d())
                    .unwrap_or(Mat4::IDENTITY.to_cols_array_2d());
                let thickness = maybe_cfg.map(|c| c.surfel_thickness).unwrap_or(0.01);
                commands.entity(entity).insert(MeshCpuBuffers { positions, indices, thickness, world_from_mesh });
            }
        }
    }
}

// Additional components for GPU conversion state
#[derive(Component)]
pub struct TriToSplatGpu {
    pub positions_ssbo: Buffer,
    pub indices_ssbo: Buffer,
    pub uniform_buf: Buffer,
    pub tri_count: u32,
    pub workgroups: u32,
    pub bind_group_inputs: BindGroup, // group(0): our inputs
}

#[derive(Resource)]
struct TriToSplatPipeline {
    // We reuse the renderer's group(0) view layout and group(1) gaussian uniform layout via CloudPipeline
    inputs_layout: Option<BindGroupLayout>, // our local group(0): positions, indices, uniforms
    pipeline: CachedComputePipelineId,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    world_from_mesh: [[f32; 4]; 4],
    thickness: f32,
    visibility: f32,
    opacity: f32,
    tri_count: u32,
    _pad: [u32; 3],
}

impl FromWorld for TriToSplatPipeline {
    fn from_world(world: &mut World) -> Self {
    let Some(render_device) = world.get_resource::<RenderDevice>() else { return Self { inputs_layout: None, pipeline: CachedComputePipelineId::INVALID }; };
        let Some(pipeline_cache) = world.get_resource::<PipelineCache>() else { return Self { inputs_layout: None, pipeline: CachedComputePipelineId::INVALID }; };
        let Some(asset_server) = world.get_resource::<AssetServer>() else { return Self { inputs_layout: None, pipeline: CachedComputePipelineId::INVALID }; };
        let Some(cloud_pipeline) = world.get_resource::<CloudPipeline<Gaussian3d>>() else { return Self { inputs_layout: None, pipeline: CachedComputePipelineId::INVALID }; };

        // Our local inputs group(0): positions, indices, uniforms
        let inputs_layout = render_device.create_bind_group_layout(
            Some("tri_to_splat_inputs"),
            &[
                BindGroupLayoutEntry { // positions
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                BindGroupLayoutEntry { // indices
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                       BindGroupLayoutEntry { // uniforms
                           binding: 2,
                           visibility: ShaderStages::COMPUTE,
                           ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: BufferSize::new(std::mem::size_of::<Uniforms>() as u64) },
                           count: None,
                       },
            ],
        );

        let shader = asset_server.load("shaders/tri_to_splat.wgsl");
        let shader_defs = vec![
            // enable compute writes to cloud buffers
            ShaderDefVal::from("READ_WRITE_POINTS"),
            // match renderer storage flavor
            ShaderDefVal::from("BUFFER_STORAGE"),
            ShaderDefVal::from("PLANAR_F32"),
            ShaderDefVal::from("GAUSSIAN_3D_STRUCTURE"),
            ShaderDefVal::from("F32"),
            // SH metadata used by imported bindings and helpers
            ShaderDefVal::UInt("SH_COEFF_COUNT".into(), bevy_gaussian_splatting::material::spherical_harmonics::SH_COEFF_COUNT as u32),
            ShaderDefVal::UInt("HALF_SH_COEFF_COUNT".into(), bevy_gaussian_splatting::material::spherical_harmonics::HALF_SH_COEFF_COUNT as u32),
            ShaderDefVal::UInt("SH_VEC4_PLANES".into(), bevy_gaussian_splatting::material::spherical_harmonics::SH_VEC4_PLANES as u32),
            ShaderDefVal::UInt("SH_DEGREE".into(), bevy_gaussian_splatting::material::spherical_harmonics::SH_DEGREE as u32),
            ShaderDefVal::UInt("SH_4D_COEFF_COUNT".into(), bevy_gaussian_splatting::material::spherindrical_harmonics::SH_4D_COEFF_COUNT as u32),
            ShaderDefVal::UInt("SH_DEGREE_TIME".into(), bevy_gaussian_splatting::material::spherindrical_harmonics::SH_4D_DEGREE_TIME as u32),
        ];
        // Include group(1) gaussian uniforms and group(2) cloud storage to satisfy pipeline layout indices 1 and 2.
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("tri_to_splat_pipeline".into()),
            layout: vec![
                inputs_layout.clone(),
                cloud_pipeline.gaussian_uniform_layout.clone(),
                cloud_pipeline.gaussian_cloud_layout.clone(),
            ],
            shader,
            shader_defs,
            entry_point: "main".into(),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: true,
        });

        Self { inputs_layout: Some(inputs_layout), pipeline }
    }
}

fn queue_tri_to_splat_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline: Option<Res<TriToSplatPipeline>>,
    query: Query<(Entity, &MeshCpuBuffers, &PlanarGaussian3dHandle), Without<TriToSplatGpu>>,
) {
    let Some(pipeline) = pipeline else { return; };
    for (entity, cpu, _cloud_handle) in &query {
        let tri_count = (cpu.indices.len() / 3) as u32;

        let positions_ssbo = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("tri_to_splat_positions"),
            contents: bytemuck::cast_slice(&cpu.positions),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let indices_ssbo = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("tri_to_splat_indices"),
            contents: bytemuck::cast_slice(&cpu.indices),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let uniforms = Uniforms {
            world_from_mesh: cpu.world_from_mesh,
            thickness: cpu.thickness,
            visibility: 1.0,
            opacity: 1.0,
            tri_count,
            _pad: [0; 3],
        };
        let uniform_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("tri_to_splat_uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        // Our inputs bind group (group 0)
        let Some(inputs_layout) = &pipeline.inputs_layout else { continue };
        let bind_group_inputs = render_device.create_bind_group(
            "tri_to_splat_inputs_bg",
            inputs_layout,
            &[
                BindGroupEntry { binding: 0, resource: positions_ssbo.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: indices_ssbo.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: uniform_buf.as_entire_binding() },
            ],
        );

        commands.entity(entity).insert(TriToSplatGpu {
            positions_ssbo,
            indices_ssbo,
            uniform_buf,
            tri_count,
            workgroups: (tri_count as f32 / 256.0).ceil() as u32,
            bind_group_inputs,
        });
    }
}

// Minimal compute node that dispatches once per entity with TriToSplatGpu
struct TriToSplatNode {
    conv_q: bevy::ecs::query::QueryState<(Entity, &'static TriToSplatGpu)>,
    cloud_q: bevy::ecs::query::QueryState<&'static PlanarStorageBindGroup<Gaussian3d>>,
    uniform_q: bevy::ecs::query::QueryState<&'static DynamicUniformIndex<CloudUniform>>,
    initialized: bool,
}

impl FromWorld for TriToSplatNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            conv_q: world.query(),
            cloud_q: world.query(),
            uniform_q: world.query(),
            initialized: false,
        }
    }
}

impl Node for TriToSplatNode {
    fn update(&mut self, world: &mut World) {
        // Ensure pipeline is ready
        let cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<TriToSplatPipeline>();
        if let CachedPipelineState::Ok(_) = cache.get_compute_pipeline_state(pipeline.pipeline) {
            self.initialized = true;
            self.conv_q.update_archetypes(world);
            self.cloud_q.update_archetypes(world);
            self.uniform_q.update_archetypes(world);
        }
    }

    fn run(&self, _graph: &mut RenderGraphContext, render_context: &mut RenderContext, world: &World) -> Result<(), NodeRunError> {
        if !self.initialized { return Ok(()); }
        let cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<TriToSplatPipeline>();
    if pipeline.pipeline == CachedComputePipelineId::INVALID { return Ok(()); }
    let Some(pso) = cache.get_compute_pipeline(pipeline.pipeline) else { return Ok(()); };
        let uniforms = world.resource::<GaussianUniformBindGroups>();
        let gaussian_bg = uniforms.base_bind_group.as_ref();
        if gaussian_bg.is_none() {
            // required group(1) not ready yet
            return Ok(());
        }

        let mut pass = render_context.command_encoder().begin_compute_pass(&ComputePassDescriptor { label: Some("tri_to_splat_compute"), timestamp_writes: None });
        pass.set_pipeline(pso);
        for (entity, conv) in self.conv_q.iter_manual(world) {
            pass.set_bind_group(0, &conv.bind_group_inputs, &[]);
            // safe to unwrap due to early return above
            if let Ok(uni_ix) = self.uniform_q.get_manual(world, entity) {
                pass.set_bind_group(1, gaussian_bg.unwrap(), &[uni_ix.index()]);
            } else {
                // dynamic uniform not ready for this entity yet
                continue;
            }
            if let Ok(cloud_bg) = self.cloud_q.get_manual(world, entity) {
                pass.set_bind_group(2, &cloud_bg.bind_group, &[]);
                pass.dispatch_workgroups(conv.workgroups, 1, 1);
            }
        }
        Ok(())
    }
}
