//! Triangles → Gaussians (compute), writing **directly** into the planar 3D storage (RW)
//! used by bevy_gaussian_splatting. Bevy 0.16 / wgpu 0.24.
//!
//! - Separate **RW** bind group at @set(2) for compute (avoids RO/RW validation errors).
//! - Uses typed RenderGraph label + `ViewNodeRunner` and a `QueryState` inside the node
//!   to iterate with only `&World`.
//! - Pipeline layout = [ inputs_layout (set 0), params_layout (set 1), planar_rw_layout (set 2) ].
//!
//! Make sure you load the shader as "tri_to_splat.wgsl" in your assets.

use bevy::{
    core_pipeline::core_3d::graph::Core3d,
    ecs::query::QueryItem,
    prelude::*,
    render::{
        extract_component::{
            ComponentUniforms, DynamicUniformIndex, ExtractComponent, ExtractComponentPlugin,
            UniformComponentPlugin,
        },
        render_asset::RenderAssets,
        render_graph::{
            NodeRunError, RenderGraphApp, RenderGraphContext, RenderLabel, ViewNode, ViewNodeRunner,
        },
        render_resource::*,
        renderer::{RenderContext, RenderDevice},
        Render, RenderApp, RenderSet,
    },
};

// From your forked splatting crate
use bevy_gaussian_splatting::{
    gaussian::formats::planar_3d::{
        PlanarStorageGaussian3d},
        sort::radix::RadixSortLabel,
        PlanarGaussian3dHandle
};







// ------------------------ Per-view params (dynamic uniform @set(1)) ------------------------

/// Params visible to the compute shader as a **dynamic uniform**.
/// Keep it minimal; extend as needed (must remain `ShaderType`).
#[derive(Component, Clone, Copy, Default, ExtractComponent, ShaderType)]
pub struct TriToSplatParams {
    // Number of gaussians to process (max across clouds); informative for shader-side bounds.
    pub gaussian_count:   u32,
    // Global elapsed time in seconds; used to drive hardcoded morph timelines.
    pub elapsed_seconds:  f32,
    // Hardcoded morph duration in seconds (can be overridden per-scene later).
    pub duration_seconds: f32,
    // Padding to keep std140-like 16-byte alignment for the uniform struct.
    pub _pad:             f32,
    // Starting sphere for spawn positions (center and radius)
    pub sphere_center:    Vec3,
    pub sphere_radius:    f32,
}

/// Index into the dynamic uniform buffer for the current view.
pub type TriToSplatParamsIndex = DynamicUniformIndex<TriToSplatParams>;







// ---------------------- Inputs (set 0) and planar RW (set 2) ----------------------

/// Per-entity inputs bind group for @group(0).
/// You create this elsewhere to reflect your actual inputs (positions/indices/etc).
#[derive(Component)]
pub struct TriToSplatGpu {
    pub bind_group_inputs:  BindGroup,
    pub workgroups:         UVec3,
}



/// CPU-side inputs collected from a mesh, uploaded to GPU during prepare to back the inputs bind group.
#[derive(Component, Clone, ExtractComponent)]
pub struct TriToSplatCpuInput {
    pub positions:  Vec<[f32; 4]>,
    pub indices:    Vec<u32>,
    pub tri_count:  u32,
}



/// **RW** bind group for planar storage used by compute (**@group(2)**).
#[derive(Component)]
pub struct PlanarStorageBindGroupRw {
    pub bind_group: BindGroup,
}







// ---------------- Job Queue (prepared -> consumed) -----------------

#[derive(Clone)]
struct TriToSplatJob {
    inputs_bg:      BindGroup,
    planar_rw_bg:   BindGroup,
    workgroups:     UVec3,
}

#[derive(Resource, Default)]
pub struct TriToSplatJobQueue {
    jobs: Vec<TriToSplatJob>,
}

/// Clear queued compute jobs at the start of the Render frame so we only dispatch once per frame
fn clear_tri_to_splat_jobs(
    mut job_queue: ResMut<TriToSplatJobQueue>
) {

    if !job_queue.jobs.is_empty() {
        bevy::log::info!(
            "clear_tri_to_splat_jobs: clearing {} queued job(s)",
            job_queue.jobs.len()
        );
        job_queue.jobs.clear();
    }
}







/// Creates a layout with read_only=false
pub fn queue_planar_cloud_rw_bind_group(
    mut commands:   Commands,
    rd:             Res<RenderDevice>,
    gpu_clouds:     Res<RenderAssets<PlanarStorageGaussian3d>>,
    pipeline:       Res<TriToSplatPipeline>,
    q:              Query<(Entity, &PlanarGaussian3dHandle)>,
) {

    bevy::log::info!("queue_planar_cloud_rw_bind_group: begin");
    
    let mut created = 0usize;

    for (entity, handle) in &q {
        let Some(storage) = gpu_clouds.get(&handle.0) else {
            continue;
        };

        let bg = rd.create_bind_group(
            "storage_gaussian_3d_bind_group_rw",
            &pipeline.planar_rw_layout, // Use the correct layout from our pipeline
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: storage.position_visibility.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: storage.spherical_harmonic.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: storage.rotation.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: storage.scale_opacity.as_entire_binding(),
                },
            ],
        );

        commands
            .entity(entity)
            .insert(PlanarStorageBindGroupRw { bind_group: bg });

        bevy::log::info!("queue_planar_cloud_rw_bind_group: added PlanarStorageBindGroupRw to entity {entity:?}");

        created += 1;
    }

    if created > 0 {
        bevy::log::info!(
            "queue_planar_cloud_rw_bind_group: created {} bind groups",
            created
        );
    }
}







/// Create a trivial inputs bind group for each cloud so the compute node can dispatch.
/// This uses small dummy read-only storage buffers and a tiny uniform to satisfy layout set(0).
pub fn queue_tri_to_splat_inputs(
    mut commands:   Commands,
    rd:             Res<RenderDevice>,
    pipe:           Res<TriToSplatPipeline>,
    mut job_queue:  ResMut<TriToSplatJobQueue>,
    q:              Query<(Entity, &PlanarStorageBindGroupRw, &TriToSplatCpuInput)>,
    existing_gpu:   Query<(), With<TriToSplatGpu>>, 
) {

    bevy::log::info!("queue_tri_to_splat_inputs: candidates={}", q.iter().len());

    let mut created = 0usize;

    for (entity, planar_rw, cpu) in &q {

        // Skip entities that already have TriToSplatGpu
        if existing_gpu.get(entity).is_ok() {
            bevy::log::info!("queue_tri_to_splat_inputs: skipping entity {entity:?} - already has TriToSplatGpu");
            continue;
        }

        bevy::log::info!("queue_tri_to_splat_inputs: processing entity {entity:?}");

        // Upload CPU arrays to GPU buffers
        let ro_flags    = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let u_flags     = BufferUsages::UNIFORM | BufferUsages::COPY_DST;
        let pos_bytes   = bytemuck::cast_slice::<[f32; 4], u8>(&cpu.positions);
        let idx_bytes   = bytemuck::cast_slice::<u32, u8>(&cpu.indices);

        let buf_positions = rd.create_buffer_with_data(&BufferInitDescriptor {
            label:      Some("tri_to_splat.positions"),
            contents:   pos_bytes,
            usage:      ro_flags,
        });
        let buf_indices = rd.create_buffer_with_data(&BufferInitDescriptor {
            label:      Some("tri_to_splat.indices"),
            contents:   idx_bytes,
            usage:      ro_flags,
        });

        // Uniform: pack counts (verts, indices, tris)
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Counts {
            verts: u32,
            indices: u32,
            tris: u32,
            _pad: u32,
        }

        let counts = Counts {
            verts:      cpu.positions.len() as u32,
            indices:    cpu.indices.len() as u32,
            tris:       cpu.tri_count,
            _pad:       0,
        };

        let buf_counts = rd.create_buffer_with_data(&BufferInitDescriptor {
            label:      Some("tri_to_splat.counts"),
            contents:   bytemuck::bytes_of(&counts),
            usage:      u_flags,
        });

        let bind_group_inputs = rd.create_bind_group(
            "tri_to_splat.inputs_bg",
            &pipe.inputs_layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: buf_positions.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: buf_indices.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: buf_indices.as_entire_binding(),
                }, // placeholder extra
                BindGroupEntry {
                    binding: 3,
                    resource: buf_counts.as_entire_binding(),
                },
            ],
        );

        // Workgroup sizing: match WGSL @workgroup_size(64, 1, 1)
        let x = (cpu.tri_count + 63) / 64;

        bevy::log::info!(
            "queue_tri_to_splat_inputs: uploading {} verts / {} tris; dispatch x={}",
            cpu.positions.len(),
            cpu.tri_count,
            x.max(1)
        );

        let workgroups = UVec3::new(x.max(1), 1, 1);

        // Enqueue a job for the compute node
        job_queue.jobs.push(TriToSplatJob {
            inputs_bg:      bind_group_inputs.clone(),
            planar_rw_bg:   planar_rw.bind_group.clone(),
            workgroups,
        });

        // Mark entity so we don't enqueue again
        commands.entity(entity).insert(TriToSplatGpu {
            bind_group_inputs: bind_group_inputs,
            workgroups,
        });

        bevy::log::info!("queue_tri_to_splat_inputs: added TriToSplatGpu to entity {entity:?}");

        created += 1;
    }

    if created > 0 {
        bevy::log::info!(
            "queue_tri_to_splat_inputs: created {} inputs bind groups",
            created
        );
    }
}
/// Re-enqueue compute jobs every frame for entities that already have GPU bind groups.
/// This makes the compute pass continuous without re-uploading buffers.
pub fn requeue_existing_tri_to_splat_jobs(
    mut job_queue:  ResMut<TriToSplatJobQueue>,
    q:              Query<(&TriToSplatGpu, &PlanarStorageBindGroupRw)>,
){
    let mut count = 0usize;
    for (gpu, planar_rw) in &q {
        job_queue.jobs.push(TriToSplatJob {
            inputs_bg:      gpu.bind_group_inputs.clone(),
            planar_rw_bg:   planar_rw.bind_group.clone(),
            workgroups:     gpu.workgroups,
        });
        count += 1;
    }

    if count > 0 {
        bevy::log::info!(
            "requeue_existing_tri_to_splat_jobs: queued {} job(s) for this frame",
            count
        );
    }
}







// --------------------------------- Pipeline ----------------------------------

#[derive(Resource)]
pub struct TriToSplatPipeline {
    pub pipeline: CachedComputePipelineId,
    pub inputs_layout: BindGroupLayout,    // @group(0)
    pub params_layout: BindGroupLayout,    // @group(1) dynamic uniform
    pub planar_rw_layout: BindGroupLayout, // @group(2) - THIS IS NOW CORRECT
}

impl FromWorld for TriToSplatPipeline {

    fn from_world(world: &mut World) -> Self {

        let rd           =  world.resource::<RenderDevice>();
        let asset_server =  world.resource::<AssetServer>();

        // @group(0): inputs (you can adjust entries to mirror your actual inputs bind group)
        let inputs_layout = rd.create_bind_group_layout(
            "tri_to_splat.inputs_layout",
            &[
                // 0 = RO storage buffer (e.g., positions table)
                BindGroupLayoutEntry {
                    binding:    0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },

                // 1 = RO storage buffer (e.g., triangle indices)
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },

                // 2 = RO storage buffer (optional extra)
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },

                // 3 = non-dynamic uniform (optional per-job constants)
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
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
                    min_binding_size: BufferSize::new(
                        std::mem::size_of::<TriToSplatParams>() as u64
                    ),
                },
                count: None,
            }],
        );

        // @group(2): planar RW layout (must match queue system + shader)
        let planar_rw_layout = rd.create_bind_group_layout(
            "storage_gaussian_3d_rw_layout",
            &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },

                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },

                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },

                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        );

        // Load from our crate's assets folder (assets/shaders/tri_to_splat.wgsl)
        let shader: Handle<Shader> = asset_server.load("shaders/tri_to_splat.wgsl");

        let pipeline = world
            .resource_mut::<PipelineCache>()
            .queue_compute_pipeline(ComputePipelineDescriptor {
                label:  Some("tri_to_splat_pipeline".into()),
                layout: vec![
                    inputs_layout.clone(),
                    params_layout.clone(),
                    planar_rw_layout.clone(), // Use our new, correct layout
                ],
                push_constant_ranges: vec![],
                shader,
                shader_defs: vec![],
                entry_point: "cs_main".into(),
                zero_initialize_workgroup_memory: false,
            });

        Self {
            pipeline,
            inputs_layout,
            params_layout,
            planar_rw_layout, // Store our correct layout
        }
    }
}







// ---------------------------------- Node -------------------------------------

/// The compute node; consumes jobs queued during PrepareBindGroups (like the Game of Life example).
pub struct TriToSplatNode;

impl FromWorld for TriToSplatNode {
    fn from_world(_world: &mut World) -> Self {
        Self
    }
}

impl ViewNode for TriToSplatNode {
    // Provide per-view dynamic uniform index
    type ViewQuery = (&'static TriToSplatParams, &'static TriToSplatParamsIndex);

    fn run(
        &self,
        _graph:                 &mut RenderGraphContext,
        rcx:                    &mut RenderContext,
        (_params, params_ix):   QueryItem<Self::ViewQuery>,
        world:                  &World,
    ) -> Result<(), NodeRunError> {

        bevy::log::info!("TriToSplatNode: run() called");
        
        let cache   = world.resource::<PipelineCache>();
        let pipe    = world.resource::<TriToSplatPipeline>();

        let Some(compute) = cache.get_compute_pipeline(pipe.pipeline) else {
            bevy::log::warn!("TriToSplatNode: compute pipeline not ready yet");
            return Ok(());
        };

        bevy::log::info!("TriToSplatNode: compute pipeline is ready");

        let params_uniforms = world.resource::<ComponentUniforms<TriToSplatParams>>();

        let Some(params_binding) = params_uniforms.uniforms().binding() else {
            bevy::log::warn!("TriToSplatNode: TriToSplatParams uniform buffer not initialized yet");
            return Ok(());
        };

        bevy::log::info!("TriToSplatNode: params uniform buffer is ready");
        
        let params_bg = rcx.render_device().create_bind_group(
            "tri_to_splat.params_bg",
            &pipe.params_layout,
            &[BindGroupEntry {
                binding: 0,
                resource: params_binding.clone(),
            }],
        );

        // Compute pass
        let mut pass = rcx
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("tri_to_splat.compute"),
                timestamp_writes: None,
            });

        pass.set_pipeline(compute);
        pass.set_bind_group(1, &params_bg, &[params_ix.index()]);

        bevy::log::info!("TriToSplatNode: bound params with index {}", params_ix.index());

        // Dispatch queued jobs
        let mut job_count = 0usize;

        if let Some(queue) = world.get_resource::<TriToSplatJobQueue>() {

            for job in &queue.jobs {
                bevy::log::info!(
                    "TriToSplatNode: dispatching workgroups({}, {}, {})",
                    job.workgroups.x, job.workgroups.y, job.workgroups.z
                );
                pass.set_bind_group(0, &job.inputs_bg, &[]);
                pass.set_bind_group(2, &job.planar_rw_bg, &[]);
                pass.dispatch_workgroups(job.workgroups.x, job.workgroups.y, job.workgroups.z);
                job_count += 1;
            }

        } else {
            bevy::log::warn!("TriToSplatNode: TriToSplatJobQueue resource missing");
        }

        if job_count == 0 {
            bevy::log::warn!("TriToSplatNode: no jobs to dispatch this frame - no entities found");
        } else {
            bevy::log::info!("TriToSplatNode: successfully dispatched {} job(s)", job_count);
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

        app.add_plugins((
            ExtractComponentPlugin::<TriToSplatParams>::default(),
            UniformComponentPlugin::<TriToSplatParams>::default(),
            ExtractComponentPlugin::<TriToSplatCpuInput>::default(),
        ));

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        bevy::log::info!("TriToSplatPlugin.build: configuring render systems and graph node");

        render_app
            .init_resource::<TriToSplatJobQueue>()
            .add_systems(
                Render,
                clear_tri_to_splat_jobs
                    .in_set(RenderSet::PrepareBindGroups)
                    .before(queue_planar_cloud_rw_bind_group),
            )
            .add_systems(
                Render,
                queue_planar_cloud_rw_bind_group.in_set(RenderSet::PrepareBindGroups),
            )
            .add_systems(
                Render,
                (
                    queue_tri_to_splat_inputs
                        .in_set(RenderSet::PrepareBindGroups)
                        .after(queue_planar_cloud_rw_bind_group),
                    // After we've created bind groups for any new entities, re-enqueue all existing jobs.
                    requeue_existing_tri_to_splat_jobs
                        .in_set(RenderSet::PrepareBindGroups)
                        .after(queue_tri_to_splat_inputs),
                ),
            )
            .add_render_graph_node::<ViewNodeRunner<TriToSplatNode>>(Core3d, TriToSplatNodeLabel)
            .add_render_graph_edges(
                Core3d,
                (
                    TriToSplatNodeLabel,
                    RadixSortLabel,
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            bevy::log::info!("TriToSplatPlugin.finish: initializing TriToSplatPipeline resource");
            render_app.init_resource::<TriToSplatPipeline>();
        }
    }
}