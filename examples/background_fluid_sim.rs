#![allow(dead_code)]
//! 2D Fluid-like particle simulation using Gaussian splats (compute-before-sort)
//!
//! Self-contained example that:
//! - Spawns a PlanarGaussian3d cloud with N particles
//! - Maintains a GPU velocities buffer
//! - Runs a compute pass each frame BEFORE sorting to update positions within camera bounds
//! - Renders with bevy_gaussian_splatting Gaussian pipeline

use bevy::prelude::*;
use bevy::render::{
    extract_component::{ExtractComponent, UniformComponentPlugin, DynamicUniformIndex},
    render_graph::{RenderGraphApp, RenderLabel, ViewNode, ViewNodeRunner},
    render_resource::*,
    renderer::RenderDevice,
    Render, RenderApp, RenderSet,
};
use bevy_gaussian_splatting::{
    gaussian::f32::{PositionVisibility, Rotation, ScaleOpacity},
    sort::radix::RadixSortLabel,
    PlanarGaussian3dHandle, SphericalHarmonicCoefficients, CloudSettings, GaussianCamera,
};

// GPU storage for the planar cloud we render; we'll create our own RW bind group locally
use bevy::render::render_asset::RenderAssets;
use bevy_gaussian_splatting::gaussian::formats::planar_3d::PlanarStorageGaussian3d;

// ------------------------------- Config ---------------------------------

const NUM_PARTICLES: u32 = 20_000;
// Make splats more visible for the demo
const BASE_SCALE: f32 = 0.12;
// Scale the compute-simulation bounds to better fit the camera view
const BOUNDS_SCALE_X: f32 = 1.0; // ~20x horizontally
const BOUNDS_SCALE_Y: f32 = 1.0; // ~12x vertically

// ------------------------------ App entry --------------------------------

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        // Bring in the gaussian renderer only; we'll hand-roll our compute and bind groups
        .add_plugins(bevy_gaussian_splatting::GaussianSplattingPlugin)
        // Uniform extraction for our params
        .add_plugins(UniformComponentPlugin::<FluidParams>::default())
    // Extract CPU init data so render-world prepare systems can see it
    .add_plugins(bevy::render::extract_component::ExtractComponentPlugin::<FluidCpuInit>::default())
        // Our local plugin that wires the compute before sorting
        .add_plugins(FluidComputePlugin)
        .add_systems(Startup, (setup_scene, setup_cloud, setup_ui))
    .add_systems(Update, update_params)
        .run();
}

// ------------------------------- Scene -----------------------------------

fn setup_scene(mut commands: Commands) {
    // 2D overlay camera
    commands.spawn((Camera2d, Camera { order: 10, ..default() }));

    // Top-down orthographic 3D camera for Gaussian rendering
    commands.spawn((
        GaussianCamera { warmup: true },
        Camera3d::default(),
        Camera {
            order: 0,
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
    // Keep units intuitive: 1.0 world unit ~= 1.0 ortho unit
    Projection::Orthographic(OrthographicProjection { scale: 1.0, ..OrthographicProjection::default_3d() }),
        Transform::from_translation(Vec3::new(0.0, 0.0, 10.0))
            .looking_at(Vec3::ZERO, Vec3::Y),
        // Initialize with some reasonable defaults; update_params will refresh each frame
        FluidParams {
            gaussian_count: NUM_PARTICLES,
            bounds_min: Vec2::splat(-5.0),
            bounds_max: Vec2::splat(5.0),
            damping: 0.995,
            speed_limit: 5.0,
            swirl_strength: 1.2,
            force: Vec2::new(0.0, 0.0),
            ..default()
        },
    ));
}

fn setup_ui(mut commands: Commands) {
    commands.spawn((
        Text::new("Fluid splats: WASD/Mouse not required – enjoy the flow"),
        TextFont { font_size: 18.0, ..default() },
        TextColor(Color::srgb(1.0, 1.0, 1.0)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        },
    ));
}

fn setup_cloud(mut commands: Commands, mut clouds: ResMut<Assets<bevy_gaussian_splatting::PlanarGaussian3d>>) {
    // Initialize the CPU-side asset with N particles
    let n = NUM_PARTICLES as usize;

    let mut position_visibility = Vec::with_capacity(n);
    let mut spherical_harmonic = Vec::with_capacity(n);
    let mut rotation = Vec::with_capacity(n);
    let mut scale_opacity = Vec::with_capacity(n);

    // Deterministic hashed distribution in [-12.0, 12.0]^2, z=0 (wider spread for larger bounds)
    for i in 0..n as u32 {
        let u = frac(hash11(i as f32));
        let v = frac(hash11((i as f32) * 1.37 + 7.11));
        let p = Vec2::new(u * 2.0 - 1.0, v * 2.0 - 1.0) * 12.0; // within [-12, 12]
        position_visibility.push(PositionVisibility { position: [p.x, p.y, 0.0], visibility: 1.0 });

        spherical_harmonic.push(SphericalHarmonicCoefficients { coefficients: solid_color_dc([0.9, 0.95, 1.0]) });
        rotation.push(Rotation { rotation: [1.0, 0.0, 0.0, 0.0] });
        scale_opacity.push(ScaleOpacity { scale: [BASE_SCALE, BASE_SCALE, BASE_SCALE], opacity: 1.0 });
    }

    let cloud_asset = bevy_gaussian_splatting::PlanarGaussian3d {
        position_visibility,
        spherical_harmonic,
        rotation,
        scale_opacity,
    };
    let handle = clouds.add(cloud_asset);

    // Create initial velocities (stronger random for visible motion)
    let mut velocities = Vec::with_capacity(n);
    for i in 0..n as u32 {
        let a = 6.2831853 * frac(hash11(i as f32));
        let r = 0.3 + 0.7 * frac(hash11((i as f32) * 3.1));
        velocities.push([a.cos() * r * 1.5, a.sin() * r * 1.5]);
    }

    commands.spawn((
        PlanarGaussian3dHandle(handle),
        CloudSettings { global_scale: 2.0, opacity_adaptive_radius: false, ..default() },
        Name::new("FluidGaussianCloud"),
        FluidCpuInit { count: NUM_PARTICLES, velocities },
        Visibility::Visible,
        Transform::IDENTITY,
    ));
}

// ------------------------------- CPU helpers ------------------------------

fn hammersley_1d(i: u32, n: u32) -> f32 { (i as f32 + 0.5) / n as f32 }
fn reverse_bits(x: u32) -> u32 { x.reverse_bits() }
fn frac(x: f32) -> f32 { x - x.floor() }
fn solid_color_dc(rgb: [f32; 3]) -> [f32; 48] {
    let mut c = [0.0_f32; 48];
    let inv_y00 = 1.0 / 0.2821_f32;
    c[0] = rgb[0] * inv_y00;
    c[1] = rgb[1] * inv_y00;
    c[2] = rgb[2] * inv_y00;
    c
}

fn hash11(n: f32) -> f32 { (n * 17.0 + 0.1).sin() * 43758.5453_f32 } // not truly random; good enough

// ------------------------------ Params (uniform) ---------------------------

#[derive(Component, Clone, Copy, Default, ExtractComponent, ShaderType)]
pub struct FluidParams {
    pub gaussian_count: u32,
    pub dt: f32,
    pub elapsed: f32,
    pub padding0: f32,
    pub bounds_min: Vec2,
    pub bounds_max: Vec2,
    pub damping: f32,
    pub speed_limit: f32,
    pub swirl_strength: f32,
    pub padding1: f32,
    pub force: Vec2,
}

pub type FluidParamsIndex = DynamicUniformIndex<FluidParams>;

#[derive(Component, Clone, ExtractComponent)]
pub struct FluidCpuInit {
    pub count: u32,
    pub velocities: Vec<[f32; 2]>,
}

#[derive(Component)]
pub struct FluidGpu {
    pub bind_group_vel: BindGroup,
    pub workgroups: UVec3,
}

#[derive(Resource, Default)]
pub struct FluidJobQueue { jobs: Vec<(BindGroup, BindGroup, UVec3)> } // (planar_rw, vel, wg)

/// Local RW bind group for planar cloud storage used by our compute pass
#[derive(Component)]
pub struct PlanarStorageBindGroupRw {
    pub bind_group: BindGroup,
}

// ------------------------- Compute pipeline resources ----------------------

#[derive(Resource)]
pub struct FluidPipeline {
    pub pipeline: CachedComputePipelineId,
    pub params_layout: BindGroupLayout, // @group(0)
    pub planar_rw_layout: BindGroupLayout, // reuse layout created in gaussian pipeline for set(1)
    pub vel_layout: BindGroupLayout, // @group(2)
}

impl FromWorld for FluidPipeline {
    fn from_world(world: &mut World) -> Self {
        let rd = world.resource::<RenderDevice>();
        let asset_server = world.resource::<AssetServer>();

        // Create a planar RW layout compatible with PlanarGaussian3d GPU storage
        let planar_rw_layout = rd.create_bind_group_layout(
            "fluid.planar_rw_layout",
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

        // @group(0) fluid params (dynamic uniform)
        let params_layout = rd.create_bind_group_layout(
            "fluid.params_layout",
            &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: Some(FluidParams::min_size()),
                },
                count: None,
            }],
        );

        // @group(2) velocities storage RW
        let vel_layout = rd.create_bind_group_layout(
            "fluid.vel_layout",
            &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        );

        // Load WGSL shader via AssetServer
        let shader: Handle<Shader> = asset_server.load("shaders/fluid_sim.wgsl");

        let pipeline = world
            .resource_mut::<PipelineCache>()
            .queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("fluid.compute".into()),
                layout: vec![
                    params_layout.clone(),
                    planar_rw_layout.clone(),
                    vel_layout.clone(),
                ],
                push_constant_ranges: vec![],
                shader,
                shader_defs: vec![],
                entry_point: "cs_main".into(),
                zero_initialize_workgroup_memory: false,
            });

        Self { pipeline, params_layout, planar_rw_layout, vel_layout }
    }
}

// ----------------------------- Render systems ------------------------------

fn fluid_clear_jobs(mut queue: ResMut<FluidJobQueue>) {
    queue.jobs.clear();
}

fn fluid_queue_new(
    mut commands: Commands,
    rd: Res<RenderDevice>,
    pipe: Res<FluidPipeline>,
    q: Query<(Entity, &PlanarStorageBindGroupRw, &FluidCpuInit), Without<FluidGpu>>,
) {
    if q.is_empty() { return; }

    for (e, _planar_rw, cpu) in &q {
        // Upload velocities as vec2<f32> storage
        let bytes = bytemuck::cast_slice::<[f32; 2], u8>(&cpu.velocities);
        let buf = rd.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("fluid.velocities"),
            contents: bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let vel_bg = rd.create_bind_group(
            "fluid.vel_bg",
            &pipe.vel_layout,
            &[BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
        );

        let x = (cpu.count + 255) / 256;
        let workgroups = UVec3::new(x.max(1), 1, 1);

        // Insert marker and store bind group + workgroups
        commands.entity(e).insert(FluidGpu { bind_group_vel: vel_bg.clone(), workgroups });
        bevy::log::info!("Fluid: created velocities BG and GPU tag for entity {:?}", e);
    }
}

fn fluid_enqueue_jobs(
    mut queue: ResMut<FluidJobQueue>,
    q: Query<(&PlanarStorageBindGroupRw, &FluidGpu)>,
) {
    for (planar, gpu) in &q {
        queue.jobs.push((planar.bind_group.clone(), gpu.bind_group_vel.clone(), gpu.workgroups));
    }
    if !queue.jobs.is_empty() {
        bevy::log::info!("Fluid: enqueued {} job(s)", queue.jobs.len());
    }
}

/// Create RW bind groups for each planar cloud once the GPU storage is ready
fn fluid_make_planar_rw_bind_group(
    mut commands: Commands,
    rd: Res<RenderDevice>,
    gpu_clouds: Res<RenderAssets<PlanarStorageGaussian3d>>,
    pipe: Res<FluidPipeline>,
    q: Query<(Entity, &PlanarGaussian3dHandle), Without<PlanarStorageBindGroupRw>>,
) {
    if q.is_empty() { return; }
    let mut created = 0usize;
    for (entity, handle) in &q {
        let Some(storage) = gpu_clouds.get(&handle.0) else { continue; };
        let bg = rd.create_bind_group(
            "fluid.planar_rw_bg",
            &pipe.planar_rw_layout,
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
    if created > 0 { bevy::log::info!("Fluid: created {created} planar RW bind group(s)"); }
}

// ------------------------------- Node -------------------------------------

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct FluidNodeLabel;

pub struct FluidNode;

impl FromWorld for FluidNode { fn from_world(_: &mut World) -> Self { Self } }

impl ViewNode for FluidNode {
    // Only the dynamic uniform index is present on the render-world view entity
    type ViewQuery = &'static FluidParamsIndex;
    fn run(
        &self,
        _graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut bevy::render::renderer::RenderContext,
        view: bevy::ecs::query::QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        let params_index = view;
        let queue = world.resource::<FluidJobQueue>();
        bevy::log::info!("FluidNode: run() — jobs={} (will dispatch if > 0)", queue.jobs.len());
        if queue.jobs.is_empty() { return Ok(()); }
        bevy::log::info!("FluidNode: dispatching {} job(s)", queue.jobs.len());
        let cache = world.resource::<PipelineCache>();
        let pipe = world.resource::<FluidPipeline>();
        let Some(pipeline) = cache.get_compute_pipeline(pipe.pipeline) else { return Ok(()); };

        // Create params bind group with dynamic offset using ComponentUniforms (must be before starting pass)
        let uniforms = world.resource::<bevy::render::extract_component::ComponentUniforms<FluidParams>>();
        let Some(binding) = uniforms.uniforms().binding() else { return Ok(()); };
        let params_bg = render_context.render_device().create_bind_group(
            "fluid.params_bg",
            &pipe.params_layout,
            &[BindGroupEntry { binding: 0, resource: binding }],
        );

        let mut pass = render_context.command_encoder().begin_compute_pass(&ComputePassDescriptor { label: Some("fluid.compute.pass"), timestamp_writes: None });
        pass.set_pipeline(pipeline);

        for (planar_bg, vel_bg, wg) in queue.jobs.iter() {
            pass.set_bind_group(0, &params_bg, &[params_index.index()]);
            pass.set_bind_group(1, planar_bg, &[]);
            pass.set_bind_group(2, vel_bg, &[]);
            pass.dispatch_workgroups(wg.x, wg.y, wg.z);
        }

        Ok(())
    }
}

// ------------------------------ Plugin wiring -----------------------------

pub struct FluidComputePlugin;

impl Plugin for FluidComputePlugin {
    fn build(&self, app: &mut App) {
    // UniformComponentPlugin for FluidParams is already enabled globally in main()
        // Insert a default resource on the main app so RenderApp init can copy as needed
        app.insert_resource(FluidJobQueue::default());

        // Hook into the render app
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<FluidJobQueue>()
            .add_systems(
                Render,
                fluid_clear_jobs
                    .in_set(RenderSet::PrepareBindGroups),
            )
            .add_systems(
                Render,
                fluid_make_planar_rw_bind_group
                    .in_set(RenderSet::PrepareBindGroups)
                    .after(fluid_clear_jobs),
            )
            .add_systems(
                Render,
                fluid_queue_new
                    .in_set(RenderSet::PrepareBindGroups)
                    .after(fluid_make_planar_rw_bind_group),
            )
            .add_systems(
                Render,
                fluid_enqueue_jobs
                    .in_set(RenderSet::PrepareBindGroups)
                    .after(fluid_queue_new),
            );

        // Add the compute node and wire before radix sort
        render_app
            .add_render_graph_node::<ViewNodeRunner<FluidNode>>(bevy::core_pipeline::core_3d::graph::Core3d, FluidNodeLabel)
            .add_render_graph_edges(
                bevy::core_pipeline::core_3d::graph::Core3d,
                (FluidNodeLabel, RadixSortLabel),
            );
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.init_resource::<FluidPipeline>();
        }
    }
}

// ----------------------------- Param updates ------------------------------

fn update_params(
    mut q_cam: Query<(&GlobalTransform, &mut FluidParams, &Projection), With<Camera3d>>, 
    time: Res<Time>,
    windows: Query<&Window>,
    mut gizmos: Gizmos,
) {
    let Ok((_xf, mut params, proj)) = q_cam.single_mut() else { return; };

    let dt = time.delta_secs().clamp(0.0, 1.0 / 30.0);
    params.dt = dt;
    params.elapsed += dt;

    // Compute view-aligned bounds in world space from ortho projection and current window aspect
    let half_w: f32;
    let half_h: f32;
    if let Projection::Orthographic(o) = proj {
        half_h = o.area.height() * 0.5 * o.scale * BOUNDS_SCALE_Y;
        half_w = o.area.width() * 0.5 * o.scale * BOUNDS_SCALE_X;
    } else {
        // Fallback based on window
        let Ok(win) = windows.single() else { return; };
        let aspect = (win.width() / win.height()).max(0.0001);
        half_h = 5.0 * BOUNDS_SCALE_Y;
        half_w = half_h * aspect as f32 * BOUNDS_SCALE_X / BOUNDS_SCALE_Y;
    }
    params.bounds_min = Vec2::new(-half_w, -half_h);
    params.bounds_max = Vec2::new(half_w, half_h);

    // Debug: draw bounds rectangle in world-space
    let min = params.bounds_min;
    let max = params.bounds_max;
    let z = 0.05; // slightly above the plane
    let a = Vec3::new(min.x, min.y, z);
    let b = Vec3::new(max.x, min.y, z);
    let c = Vec3::new(max.x, max.y, z);
    let d = Vec3::new(min.x, max.y, z);
    let col = Color::srgb(0.2, 1.0, 0.2);
    gizmos.line(a, b, col);
    gizmos.line(b, c, col);
    gizmos.line(c, d, col);
    gizmos.line(d, a, col);

    // Animated swirl around origin
    let t = params.elapsed;
    let swirl = 1.0_f32 + 0.6_f32 * (0.35_f32 * t).sin();
    params.swirl_strength = swirl as f32;
    params.force = Vec2::new(0.0, 0.0);

    // Slight damping scaled with dt
    params.damping = (1.0 - (1.0 - params.damping) * dt).clamp(0.95, 0.9999);
    params.speed_limit = 5.0;

    // Keep orientation top-down (no-op here; placeholder for future camera dynamics)
}

// No inline WGSL. Shader is loaded from assets/shaders/fluid_sim.wgsl
