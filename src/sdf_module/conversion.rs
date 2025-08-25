use bevy::prelude::*;
use bevy_gaussian_splatting::Gaussian3d;
use super::primitives::{SDF, BoxedSDF};
use crate::gaussian::settings::PointCloudSettings;





/// Settings for SDF to Gaussian conversion
#[derive(Debug, Clone)]
pub struct SDFConversionSettings {
    /// Resolution of the sampling grid
    pub resolution: UVec3,
    /// Bounds of the sampling space
    pub bounds: (Vec3, Vec3), // (min, max)
    /// Settings for the generated Gaussians
    pub gaussian_settings: PointCloudSettings,
    /// Threshold for considering a point "inside" the SDF (usually 0.0)
    pub threshold: f32,
    /// Whether to place Gaussians on the surface (near threshold) or inside
    pub surface_only: bool,
    /// Maximum distance from surface to place gaussians (when surface_only is true)
    pub surface_thickness: f32,
}

impl Default for SDFConversionSettings {
    fn default() -> Self {
        Self {
            resolution: UVec3::new(32, 32, 32),
            bounds: (Vec3::splat(-5.0), Vec3::splat(5.0)),
            gaussian_settings: PointCloudSettings::default(),
            threshold: 0.0,
            surface_only: true,
            surface_thickness: 0.5,
        }
    }
}





/// Convert an SDF to a cloud of Gaussians by sampling it on a grid
pub fn sdf_to_gaussians(sdf: &BoxedSDF, settings: &SDFConversionSettings) -> Vec<Gaussian3d> {
    let (min_bound, max_bound) = settings.bounds;
    let size = max_bound - min_bound;
    let step = size / settings.resolution.as_vec3();
    
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    
    for x in 0..settings.resolution.x {
        for y in 0..settings.resolution.y {
            for z in 0..settings.resolution.z {
                let pos = min_bound + step * Vec3::new(x as f32, y as f32, z as f32);
                let distance = sdf.distance(pos);
                
                let should_place = if settings.surface_only {
                    // Place gaussians near the surface
                    distance.abs() <= settings.surface_thickness
                } else {
                    // Place gaussians inside the SDF
                    distance <= settings.threshold
                };
                
                if should_place {
                    positions.push(pos);
                    
                    // Compute normal by sampling nearby points
                    let epsilon = step.min_element() * 0.1;
                    let normal = compute_sdf_normal(sdf, pos, epsilon);
                    normals.push(normal);
                }
            }
        }
    }
    
    // Convert points to gaussians
    crate::gaussian::creation::points_to_gaussians(
        &positions,
        Some(&normals),
        Transform::IDENTITY,
        &settings.gaussian_settings,
    )
}

/// Compute the normal at a point on an SDF surface using finite differences
fn compute_sdf_normal(sdf: &BoxedSDF, pos: Vec3, epsilon: f32) -> Vec3 {
    let gradient = Vec3::new(
        sdf.distance(pos + Vec3::X * epsilon) - sdf.distance(pos - Vec3::X * epsilon),
        sdf.distance(pos + Vec3::Y * epsilon) - sdf.distance(pos - Vec3::Y * epsilon),
        sdf.distance(pos + Vec3::Z * epsilon) - sdf.distance(pos - Vec3::Z * epsilon),
    );
    
    gradient.normalize_or_zero()
}

/// Higher-level convenience function that creates an SDF and converts it to Gaussians
pub fn sdf_sphere_to_gaussians(
    center: Vec3,
    radius: f32,
    settings: &SDFConversionSettings,
) -> Vec<Gaussian3d> {
    let sdf = super::primitives::sdf_sphere(center, radius);
    sdf_to_gaussians(&sdf, settings)
}

/// Convert a box SDF to Gaussians
pub fn sdf_box_to_gaussians(
    center: Vec3,
    size: Vec3,
    settings: &SDFConversionSettings,
) -> Vec<Gaussian3d> {
    let sdf = super::primitives::sdf_box(center, size);
    sdf_to_gaussians(&sdf, settings)
}
