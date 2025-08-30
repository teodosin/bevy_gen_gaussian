//! Triangles → Gaussians (compute), writing **directly** into the planar 3D storage (RW)
//! used by bevy_gaussian_splatting. Bevy 0.16 / wgpu 0.24.
//!
//! - Separate **RW** bind group at @set(2) for compute (avoids RO/RW validation errors).
//! - Uses typed RenderGraph label + `ViewNodeRunner` and a `QueryState` inside the node
//!   to iterate with only `&World`.
//! - Pipeline layout = [ inputs_layout (set 0), params_layout (set 1), planar_rw_layout (set 2) ].
//!
//! Make sure you load the shader as "tri_to_splat.wgsl" in your assets.

use crate::MeshToGaussian;

use bevy::{
    core_pipeline::core_3d::graph::{Core3d, Node3d}, ecs::query::QueryItem, prelude::*, render::{
        extract_component::{
            ComponentUniforms, DynamicUniformIndex, ExtractComponent, ExtractComponentPlugin,
            UniformComponentPlugin,
        },
        render_asset::RenderAssets,
        render_graph::{NodeRunError, RenderGraphApp, RenderGraphContext, RenderLabel, ViewNode, ViewNodeRunner},
        render_resource::*,
        renderer::{RenderContext, RenderDevice},
        Render, RenderApp, RenderSet,
    }
};

// From your forked splatting crate
use bevy_gaussian_splatting::{
    PlanarGaussian3dHandle,
    gaussian::formats::planar_3d::{PlanarStorageGaussian3d, Gaussian3d},
    render::CloudPipeline,
};

// ------------------------ Per-view params (dynamic uniform @set(1)) ------------------------

/// Params visible to the compute shader as a **dynamic uniform**.
/// Keep it minimal; extend as needed (must remain `ShaderType`).
#[derive(Component, Clone, Copy, Default, ExtractComponent, ShaderType)]
pub struct TriToSplatParams {
    pub gaussian_count: u32,
}

/// Index into the dynamic uniform buffer for the current view.
pub type TriToSplatParamsIndex = DynamicUniformIndex<TriToSplatParams>;

// ---------------------- Inputs (set 0) and planar RW (set 2) ----------------------

/// Per-entity inputs bind group for @group(0).
/// You create this elsewhere to reflect your actual inputs (positions/indices/etc).
#[derive(Component)]
pub struct TriToSplatGpu {
    pub bind_group_inputs: BindGroup,
    pub workgroups: UVec3,
}

/// CPU-side inputs collected from a mesh, uploaded to GPU during prepare to back the inputs bind group.
#[derive(Component, Clone)]
pub struct TriToSplatCpuInput {
    pub positions: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
    pub tri_count: u32,
}

/// **RW** bind group for planar storage used by compute (**@group(2)**).
#[derive(Component)]
pub struct PlanarStorageBindGroupRw {
    pub bind_group: BindGroup,
}

/// Create RW bind groups using the same layout as the render pipeline.
/// This system runs after default bind group creation and ensures compatibility.
pub fn queue_compatible_rw_bind_groups(
    mut commands: Commands,
    rd: Res<RenderDevice>,
    pipeline: Res<CloudPipeline<Gaussian3d>>,
    gpu_clouds: Res<RenderAssets<PlanarStorageGaussian3d>>,
    q: Query<(Entity, &PlanarGaussian3dHandle), Without<PlanarStorageBindGroupRw>>,
) {
    bevy::log::info!("queue_compatible_rw_bind_groups: begin");
    let mut created = 0usize;
    
    for (entity, handle) in &q {
        let Some(storage) = gpu_clouds.get(&handle.0) else { continue };

        // Use the same layout as the render pipeline
        let bg = rd.create_bind_group(
            "compatible_storage_gaussian_3d_bind_group_rw",
            &pipeline.gaussian_cloud_layout,
            &[
                BindGroupEntry { binding: 0, resource: storage.position_visibility.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: storage.spherical_harmonic.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: storage.rotation.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: storage.scale_opacity.as_entire_binding() },
            ],
        );

        commands.entity(entity).insert(PlanarStorageBindGroupRw { bind_group: bg });
        created += 1;
    }
    
    bevy::log::info!("queue_compatible_rw_bind_groups: created {} bind groups", created);
}
pub fn queue_planar_cloud_rw_bind_group(
    mut commands: Commands,
    rd: Res<RenderDevice>,
    gpu_clouds: Res<RenderAssets<PlanarStorageGaussian3d>>,
    q: Query<(Entity, &PlanarGaussian3dHandle), Without<PlanarStorageBindGroupRw>>,
) {
    bevy::log::info!("queue_planar_cloud_rw_bind_group: begin");
    // Matches the compute pipeline's set(2) layout.
    let planar_rw_layout = rd.create_bind_group_layout(
        "storage_gaussian_3d_rw_layout",
        &[
            // 0: position_visibility (vec3 + f32) => 16B
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(16),
                },
                count: None,
            },
            // 1: spherical harmonic planes (~192B per gaussian typically)
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(192),
                },
                count: None,
            },
            // 2: rotation (quat) => 16B
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(16),
                },
                count: None,
            },
            // 3: scale_opacity => 16B
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(16),
                },
                count: None,
            },
        ],
    );

    let mut created = 0usize;
    for (entity, handle) in &q {
        let Some(storage) = gpu_clouds.get(&handle.0) else { continue };

        // Bevy 0.16: (label, &layout, entries).
        // See examples/docs for this exact API shape. :contentReference[oaicite:1]{index=1}
        let bg = rd.create_bind_group(
            "storage_gaussian_3d_bind_group_rw",
            &planar_rw_layout,
            &[
                BindGroupEntry { binding: 0, resource: storage.position_visibility.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: storage.spherical_harmonic.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: storage.rotation.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: storage.scale_opacity.as_entire_binding() },
            ],
        );

        commands.entity(entity).insert(PlanarStorageBindGroupRw { bind_group: bg });
        created += 1;
    }

    bevy::log::info!("queue_planar_cloud_rw_bind_group: created {} bind groups", created);
}

/// Create a trivial inputs bind group for each cloud so the compute node can dispatch.
/// This uses small dummy read-only storage buffers and a tiny uniform to satisfy layout set(0).
pub fn queue_tri_to_splat_inputs(
    mut commands: Commands,
    rd: Res<RenderDevice>,
    pipe: Res<TriToSplatPipeline>,
    q: Query<(Entity, &PlanarStorageBindGroupRw, &TriToSplatCpuInput), Without<TriToSplatGpu>>,
)
{
    let mut created = 0usize;
    for (entity, _rw, cpu) in &q {
        // Upload CPU arrays to GPU buffers
        let ro_flags = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let u_flags  = BufferUsages::UNIFORM | BufferUsages::COPY_DST;
        let pos_bytes = bytemuck::cast_slice::<[f32;4], u8>(&cpu.positions);
        let idx_bytes = bytemuck::cast_slice::<u32, u8>(&cpu.indices);

        let buf_positions = rd.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("tri_to_splat.positions"),
            contents: pos_bytes,
            usage: ro_flags,
        });
        let buf_indices = rd.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("tri_to_splat.indices"),
            contents: idx_bytes,
            usage: ro_flags,
        });
        // Uniform: pack counts (verts, indices, tris)
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Counts { verts: u32, indices: u32, tris: u32, _pad: u32 }
        let counts = Counts { verts: cpu.positions.len() as u32, indices: cpu.indices.len() as u32, tris: cpu.tri_count, _pad: 0 };
        let buf_counts = rd.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("tri_to_splat.counts"),
            contents: bytemuck::bytes_of(&counts),
            usage: u_flags,
        });

        let bind_group_inputs = rd.create_bind_group(
            "tri_to_splat.inputs_bg",
            &pipe.inputs_layout,
            &[
                BindGroupEntry { binding: 0, resource: buf_positions.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: buf_indices.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: buf_indices.as_entire_binding() }, // placeholder extra
                BindGroupEntry { binding: 3, resource: buf_counts.as_entire_binding() },
            ],
        );

        // Workgroup sizing: 256 threads per group for tris
        let x = (cpu.tri_count + 255) / 256;
        commands.entity(entity).insert(TriToSplatGpu { bind_group_inputs, workgroups: UVec3::new(x.max(1), 1, 1) });
        created += 1;
    }
    if created > 0 { bevy::log::info!("queue_tri_to_splat_inputs: created {} inputs bind groups", created); }
}

// --------------------------------- Pipeline ----------------------------------

#[derive(Resource)]
pub struct TriToSplatPipeline {
    pub pipeline: CachedComputePipelineId,
    pub inputs_layout: BindGroupLayout,   // @group(0)
    pub params_layout: BindGroupLayout,   // @group(1) dynamic uniform
    pub planar_rw_layout: BindGroupLayout // @group(2)
}

impl FromWorld for TriToSplatPipeline {
    fn from_world(world: &mut World) -> Self {
        let rd = world.resource::<RenderDevice>();
        let asset_server = world.resource::<AssetServer>();

        // @group(0): inputs (you can adjust entries to mirror your actual inputs bind group)
        let inputs_layout = rd.create_bind_group_layout(
            "tri_to_splat.inputs_layout",
            &[
                // 0 = RO storage buffer (e.g., positions table)
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                // 1 = RO storage buffer (e.g., triangle indices)
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                // 2 = RO storage buffer (optional extra)
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                // 3 = non-dynamic uniform (optional per-job constants)
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        );

        // @group(1): dynamic uniform (TriToSplatParams)
        let params_layout = rd.create_bind_group_layout(
            "tri_to_splat.params_layout",
            &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: BufferSize::new(std::mem::size_of::<TriToSplatParams>() as u64),
                },
                count: None,
            }],
        );

        // @group(2): planar RW layout (must match queue system + shader)
        let planar_rw_layout = rd.create_bind_group_layout(
            "storage_gaussian_3d_rw_layout",
            &[
                BindGroupLayoutEntry {
                    binding: 0, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: BufferSize::new(16) },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: BufferSize::new(192) },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: BufferSize::new(16) },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: BufferSize::new(16) },
                    count: None,
                },
            ],
        );

    // Load from our crate's assets folder (assets/shaders/tri_to_splat.wgsl)
    let shader: Handle<Shader> = asset_server.load("shaders/tri_to_splat.wgsl");

        // Bevy 0.16: layout is Vec<BindGroupLayout> and push_constant_ranges is required (can be empty). :contentReference[oaicite:2]{index=2}
        let pipeline = world
            .resource_mut::<PipelineCache>()
            .queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("tri_to_splat_pipeline".into()),
                layout: vec![
                    inputs_layout.clone(),
                    params_layout.clone(),
                    planar_rw_layout.clone(),
                ],
                push_constant_ranges: vec![],
                shader,
                shader_defs: vec![],
                entry_point: "cs_main".into(),
                zero_initialize_workgroup_memory: false,
            });

        Self { pipeline, inputs_layout, params_layout, planar_rw_layout }
    }
}

// ---------------------------------- Node -------------------------------------

/// The compute node; uses a `QueryState` so it can iterate with only `&World`.
pub struct TriToSplatNode {
    conv_q: QueryState<(&'static TriToSplatGpu, &'static PlanarStorageBindGroupRw)>,
}

impl FromWorld for TriToSplatNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            conv_q: QueryState::new(world),
        }
    }
}

impl ViewNode for TriToSplatNode {
    // Provide per-view dynamic uniform index
    type ViewQuery = (&'static TriToSplatParams, &'static TriToSplatParamsIndex);

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        rcx: &mut RenderContext,
        (_params, params_ix): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let cache = world.resource::<PipelineCache>();
        let pipe  = world.resource::<TriToSplatPipeline>();
        let Some(compute) = cache.get_compute_pipeline(pipe.pipeline) else {
            bevy::log::trace!("TriToSplatNode: compute pipeline not ready yet");
            return Ok(());
        };

        // Build @set(1) uniform bind group from ComponentUniforms (Bevy example pattern). :contentReference[oaicite:3]{index=3}
        let params_uniforms = world.resource::<ComponentUniforms<TriToSplatParams>>();
        let Some(params_binding) = params_uniforms.uniforms().binding() else {
            bevy::log::warn!("TriToSplatNode: TriToSplatParams uniform buffer not initialized yet");
            return Ok(());
        };
        let params_bg = rcx.render_device().create_bind_group(
            "tri_to_splat.params_bg",
            &pipe.params_layout,
            &[BindGroupEntry {
                binding: 0,
                resource: params_binding.clone(),
            }],
        );

        // Compute pass
        let mut pass = rcx.command_encoder().begin_compute_pass(&ComputePassDescriptor {
            label: Some("tri_to_splat.compute"),
            timestamp_writes: None,
        });
        pass.set_pipeline(compute);
        pass.set_bind_group(1, &params_bg, &[params_ix.index()]);

        // Iterate conversion jobs with a query directly on world.
        let mut job_count = 0usize;
        for (conv, planar_rw) in self.conv_q.iter_manual(world) {
            pass.set_bind_group(0, &conv.bind_group_inputs, &[]);
            pass.set_bind_group(2, &planar_rw.bind_group, &[]);
            pass.dispatch_workgroups(conv.workgroups.x, conv.workgroups.y, conv.workgroups.z);
            job_count += 1;
        }

        if job_count == 0 {
            bevy::log::trace!("TriToSplatNode: no jobs to dispatch this frame");
        } else {
            bevy::log::info!("TriToSplatNode: dispatched {} job(s)", job_count);
        }

        Ok(())
    }
}

// ------------------------------ Plugin wiring --------------------------------

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct TriToSplatNodeLabel;

pub struct TriToSplatPlugin;

impl Plugin for TriToSplatPlugin {
    fn build(&self, app: &mut App) {
        // Per-view params → extract & upload as a dynamic uniform
        app.add_plugins((
            ExtractComponentPlugin::<TriToSplatParams>::default(),
            UniformComponentPlugin::<TriToSplatParams>::default(),
        ));

        // Render app: systems and graph node registration are safe in build.
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else { return; };
        bevy::log::info!("TriToSplatPlugin.build: configuring render systems and graph node");
        render_app
            // Prepare RW bind groups (same schedule set as in Bevy examples). :contentReference[oaicite:4]{index=4}
            .add_systems(Render, queue_planar_cloud_rw_bind_group.in_set(RenderSet::PrepareBindGroups))
            // Create compatible bind groups using the same layout as the render pipeline
            .add_systems(Render, queue_compatible_rw_bind_groups.in_set(RenderSet::PrepareBindGroups).after(queue_planar_cloud_rw_bind_group))
            // Also prepare placeholder inputs so the compute pass has something to bind at set(0)
            .add_systems(Render, queue_tri_to_splat_inputs.in_set(RenderSet::PrepareBindGroups))
            // Typed label + ViewNodeRunner; edges insert the node between LatePrepass → StartMainPass.
            .add_render_graph_node::<ViewNodeRunner<TriToSplatNode>>(Core3d, TriToSplatNodeLabel)
            .add_render_graph_edges(Core3d, (Node3d::LatePrepass, TriToSplatNodeLabel, Node3d::StartMainPass));
    }

    // Defer pipeline creation until after the renderer has initialized the RenderDevice.
    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            bevy::log::info!("TriToSplatPlugin.finish: initializing TriToSplatPipeline resource");
            render_app.init_resource::<TriToSplatPipeline>();
        }
    }
}
