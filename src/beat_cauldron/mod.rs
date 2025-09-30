use bevy::{
    input::mouse::AccumulatedMouseScroll,
    prelude::*,
    render::camera::{OrthographicProjection, Projection, ScalingMode},
    window::WindowPlugin,
};

use bevy_gaussian_splatting::{
    gaussian::f32::{PositionVisibility, Rotation, ScaleOpacity},
    CloudSettings,
    GaussianCamera,
    PlanarGaussian3d,
    PlanarGaussian3dHandle,
    SphericalHarmonicCoefficients,
};

use crate::GenGaussianPlugin;

#[derive(Component, Default, Reflect)]
#[reflect(Component)]
pub struct WorldView;

#[derive(Resource, Reflect)]
#[reflect(Resource)]
pub struct BeatCauldronSettings {
    pub grid_width: usize,
    pub grid_height: usize,
    pub cell_spacing: Vec2,
    pub grid_plane_z: f32,
    pub altitude_variation: f32,
    pub noise_base_frequency: f32,
    pub noise_lacunarity: f32,
    pub noise_persistence: f32,
    pub noise_octaves: u8,
    pub noise_offset: Vec2,
    pub color_hue_base: f32,
    pub color_hue_variation: f32,
    pub color_saturation_base: f32,
    pub color_saturation_variation: f32,
    pub color_lightness_base: f32,
    pub color_lightness_variation: f32,
    pub color_contrast_strength: f32,
    pub color_brightness_boost: f32,
    pub color_whiteness_strength: f32,
    pub color_gamma: f32,
    pub color_min_luminance: f32,
    pub color_pattern_exponent: f32,
    pub color_density_strength: f32,
    pub color_density_bias: f32,
    pub color_mask_threshold: f32,
    pub color_mask_sharpness: f32,
    pub color_final_contrast_strength: f32,
    pub min_scale: Vec3,
    pub max_scale: Vec3,
    pub scale_multiplier: f32,
    pub opacity_base: f32,
    pub opacity_variation: f32,
    pub camera_distance: f32,
    pub camera_vertical_padding: f32,
    pub zoom_speed: f32,
    pub min_zoom: f32,
    pub max_zoom: f32,
}

impl Default for BeatCauldronSettings {
    fn default() -> Self {
        let grid_width = 480;
        let grid_height = 360;
        let cell_spacing = Vec2::new(
            479.0 / ((grid_width - 1) as f32),
            359.0 / ((grid_height - 1) as f32),
        );
        let average_spacing = (cell_spacing.x + cell_spacing.y) * 0.5;

        Self {
            grid_width,
            grid_height,
            cell_spacing,
            grid_plane_z: 0.0,
            altitude_variation: 12.0,
            noise_base_frequency: 0.0075,
            noise_lacunarity: 2.15,
            noise_persistence: 0.55,
            noise_octaves: 5,
            noise_offset: Vec2::new(13.37, 42.0),
            color_hue_base: 205.0,
            color_hue_variation: 155.0,
            color_saturation_base: 0.68,
            color_saturation_variation: 0.28,
            color_lightness_base: 0.46,
            color_lightness_variation: 0.22,
            color_contrast_strength: 6.0,
            color_brightness_boost: -0.42,
            color_whiteness_strength: 0.0,
            color_gamma: 2.0,
            color_min_luminance: 0.0,
            color_pattern_exponent: 3.5,
            color_density_strength: 4.5,
            color_density_bias: 0.0,
            color_mask_threshold: 0.45,
            color_mask_sharpness: 3.25,
            color_final_contrast_strength: 14.0,
            min_scale: Vec3::new(
                0.32 * cell_spacing.x,
                0.32 * cell_spacing.y,
                0.32 * average_spacing,
            ),
            max_scale: Vec3::new(
                0.62 * cell_spacing.x,
                0.55 * cell_spacing.y,
                0.68 * average_spacing,
            ),
            scale_multiplier: 2000.0,
            opacity_base: 0.9,
            opacity_variation: 0.18,
            camera_distance: 600.0,
            camera_vertical_padding: 120.0,
            zoom_speed: 0.05,
            min_zoom: 0.2,
            max_zoom: 6.0,
        }
    }
}

impl BeatCauldronSettings {
    pub fn total_splats(&self) -> usize {
        self.grid_width.saturating_mul(self.grid_height)
    }

    pub fn grid_extent(&self) -> Vec2 {
        let width = if self.grid_width > 1 {
            (self.grid_width - 1) as f32 * self.cell_spacing.x
        } else {
            0.0
        };
        let height = if self.grid_height > 1 {
            (self.grid_height - 1) as f32 * self.cell_spacing.y
        } else {
            0.0
        };

        Vec2::new(width, height)
    }

    pub fn grid_half_extents(&self) -> Vec2 {
        self.grid_extent() * 0.5
    }

    pub fn viewport_height(&self) -> f32 {
        self.grid_extent().y + self.camera_vertical_padding
    }

    pub fn sample_noise(&self, grid_position: Vec2, offset: Vec2) -> f32 {
        if self.noise_octaves == 0 {
            return 0.5;
        }

        let base_position = grid_position * self.cell_spacing + self.noise_offset + offset;
        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = self.noise_base_frequency;
        let mut max_value = 0.0;

        for _ in 0..self.noise_octaves {
            value += amplitude * smooth_value_noise(base_position * frequency);
            max_value += amplitude;
            amplitude *= self.noise_persistence;
            frequency *= self.noise_lacunarity;
        }

        if max_value > 0.0 {
            (value / max_value).clamp(0.0, 1.0)
        } else {
            0.5
        }
    }
}

pub struct BeatCauldronPlugin;

impl Plugin for BeatCauldronPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<WorldView>();
        app.register_type::<BeatCauldronSettings>();

        app.init_resource::<BeatCauldronSettings>();

        app.add_systems(Startup, (spawn_world_view_camera, spawn_gaussian_grid));
        app.add_systems(Update, adjust_world_view_zoom);
    }
}

pub fn beat_cauldron() {
    info!("Launching Beat Cauldron sandbox");

    App::new()
        .add_plugins(
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Beat Cauldron".into(),
                    ..default()
                }),
                ..default()
            }),
        )
        .add_plugins((GenGaussianPlugin, BeatCauldronPlugin))
        .run();
}

fn spawn_world_view_camera(mut commands: Commands, settings: Res<BeatCauldronSettings>) {
    let projection = Projection::from(OrthographicProjection {
        scaling_mode: ScalingMode::FixedVertical {
            viewport_height: settings.viewport_height(),
        },
        ..OrthographicProjection::default_3d()
    });

    let focus = Vec3::new(0.0, 0.0, settings.grid_plane_z);
    let transform = Transform::from_xyz(0.0, 0.0, settings.camera_distance).looking_at(focus, Vec3::Y);

    commands.spawn((
        Camera3d::default(),
        projection,
        transform,
        GaussianCamera::default(),
        WorldView,
        Name::new("WorldViewCamera"),
    ));
}

fn adjust_world_view_zoom(
    mouse_scroll: Res<AccumulatedMouseScroll>,
    settings: Res<BeatCauldronSettings>,
    mut query: Query<&mut Projection, (With<Camera3d>, With<WorldView>)>,
) {
    let scroll_delta = mouse_scroll.delta.y;
    if scroll_delta.abs() <= f32::EPSILON {
        return;
    }

    let delta_zoom = -scroll_delta * settings.zoom_speed;
    let multiplier = 1.0 + delta_zoom;

    if !multiplier.is_finite() || multiplier <= f32::EPSILON {
        return;
    }

    for mut projection in query.iter_mut() {
        if let Projection::Orthographic(orthographic) = projection.as_mut() {
            let new_scale = (orthographic.scale * multiplier)
                .clamp(settings.min_zoom, settings.max_zoom);
            orthographic.scale = new_scale;
        }
    }
}

fn spawn_gaussian_grid(
    mut commands: Commands,
    mut clouds: ResMut<Assets<PlanarGaussian3d>>,
    settings: Res<BeatCauldronSettings>,
) {
    use noise::{NoiseFn, Perlin};

    // --- Noise crate config (inline; move to settings later if you want) ---
    let color_noise_seed: u32 = 1337;
    let color_noise_freq: Vec2 = Vec2::splat(0.02);
    let color_noise_offset: Vec2 = Vec2::new(13.7, -9.1);
    let noise_octaves: u32 = 4;
    let noise_lacunarity: f32 = 2.0;
    let noise_gain: f32 = 0.5;
    let brightness_gamma: f32 = 1.0;

    // Build a single Perlin generator once.
    let perlin = Perlin::new(color_noise_seed);

    // Simple Perlin fBm returning [0,1].
    #[inline]
    fn fbm2_perlin(perlin: &Perlin, p: Vec2, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
        let mut amp = 0.5;
        let mut sum = 0.0f32;
        let mut norm = 0.0f32;
        let mut freq = 1.0f32;
        for _ in 0..octaves {
            let n = perlin.get([ (p.x * freq) as f64, (p.y * freq) as f64 ]) as f32; // ~[-1,1]
            sum += amp * n;
            norm += amp;
            freq *= lacunarity;
            amp *= gain;
        }
        let v = if norm > 0.0 { sum / norm } else { 0.0 };
        ((v * 0.5) + 0.5).clamp(0.0, 1.0) // → [0,1]
    }

    let total_splats = settings.total_splats();

    let mut positions: Vec<PositionVisibility> = Vec::with_capacity(total_splats);
    let mut harmonics: Vec<SphericalHarmonicCoefficients> = Vec::with_capacity(total_splats);
    let mut rotations: Vec<Rotation> = Vec::with_capacity(total_splats);
    let mut scales: Vec<ScaleOpacity> = Vec::with_capacity(total_splats);

    let half_extents = settings.grid_half_extents();

    for y in 0..settings.grid_height {
        for x in 0..settings.grid_width {
            let grid_position = Vec2::new(x as f32, y as f32);

            let world_x = x as f32 * settings.cell_spacing.x - half_extents.x;
            let world_y = half_extents.y - y as f32 * settings.cell_spacing.y;

            // Keep your existing samples for other properties
            let base_noise = settings.sample_noise(grid_position, Vec2::ZERO);
            let color_noise = settings.sample_noise(grid_position, Vec2::new(37.0, 91.0));
            let secondary_noise = settings.sample_noise(grid_position, Vec2::new(-73.0, 19.0));
            let altitude_noise = settings.sample_noise(grid_position, Vec2::new(17.0, -53.0));

            // --- simplified color via Noise crate: grayscale brightness in [0,1] ---
            let p = grid_position * color_noise_freq + color_noise_offset;
            let mut brightness = fbm2_perlin(&perlin, p, noise_octaves, noise_lacunarity, noise_gain);
            brightness = brightness.powf(brightness_gamma.max(0.01));

            let mut sh = SphericalHarmonicCoefficients::default();
            sh.coefficients[0] = brightness;
            sh.coefficients[1] = brightness;
            sh.coefficients[2] = brightness;

            let altitude = settings.grid_plane_z
                + (altitude_noise * 2.0 - 1.0) * settings.altitude_variation;

            positions.push(PositionVisibility {
                position: [world_x, world_y, altitude],
                visibility: 1.0,
            });

            harmonics.push(sh);
            rotations.push(Rotation {
                rotation: [1.0, 0.0, 0.0, 0.0],
            });

            let scale_x = settings.min_scale.x
                + base_noise * (settings.max_scale.x - settings.min_scale.x);
            let scale_y = settings.min_scale.y
                + color_noise * (settings.max_scale.y - settings.min_scale.y);
            let scale_z = settings.min_scale.z
                + secondary_noise * (settings.max_scale.z - settings.min_scale.z);
            let scale = Vec3::new(scale_x, scale_y, scale_z) * settings.scale_multiplier;

            let opacity_noise = settings.sample_noise(grid_position, Vec2::new(89.0, -131.0));
            let opacity = (settings.opacity_base
                + (opacity_noise * 2.0 - 1.0) * settings.opacity_variation)
                .clamp(0.0, 1.0);

            scales.push(ScaleOpacity {
                scale: scale.to_array(),
                opacity,
            });
        }
    }

    let cloud_asset = PlanarGaussian3d {
        position_visibility: positions,
        spherical_harmonic: harmonics,
        rotation: rotations,
        scale_opacity: scales,
    };

    let handle = clouds.add(cloud_asset);

    commands.spawn((
        PlanarGaussian3dHandle(handle),
        CloudSettings::default(),
        Transform::default(),
        Visibility::Visible,
        WorldView,
        Name::new("WorldViewGaussianCloud"),
    ));
}



fn smooth_value_noise(point: Vec2) -> f32 {
    let cell = point.floor();
    let frac = point - cell;

    let c00 = lattice_value(cell);
    let c10 = lattice_value(cell + Vec2::new(1.0, 0.0));
    let c01 = lattice_value(cell + Vec2::new(0.0, 1.0));
    let c11 = lattice_value(cell + Vec2::new(1.0, 1.0));

    let fade = frac * frac * (Vec2::splat(3.0) - 2.0 * frac);

    let nx0 = c00 + (c10 - c00) * fade.x;
    let nx1 = c01 + (c11 - c01) * fade.x;

    let value = nx0 + (nx1 - nx0) * fade.y;
    value.clamp(0.0, 1.0)
}

fn lattice_value(point: Vec2) -> f32 {
    let dot = point.dot(Vec2::new(127.1, 311.7));
    (dot.sin() * 43758.5453).fract()
}