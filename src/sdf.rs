use bevy::prelude::*;
use crate::edit::{EditBatch, EditOp};

/// Different SDF brush modes for voxel editing
#[derive(Debug, Clone, Copy)]
pub enum BrushMode {
    Add,
    Subtract,
    Paint,
}

/// Settings for SDF brush operations
#[derive(Resource)]
pub struct BrushSettings {
    pub radius: f32,
    pub mode: BrushMode,
}

impl Default for BrushSettings {
    fn default() -> Self {
        Self {
            radius: 3.0,
            mode: BrushMode::Add,
        }
    }
}

/// Apply a spherical brush operation to the voxel world
pub fn apply_sphere_brush(
    batch: &mut EditBatch,
    center: Vec3,
    radius: f32,
    mode: BrushMode,
) {
    let min_bound = (center - Vec3::splat(radius)).as_ivec3().max(IVec3::ZERO);
    let max_bound = (center + Vec3::splat(radius)).as_ivec3().min(IVec3::splat(31));
    
    for x in min_bound.x..=max_bound.x {
        for y in min_bound.y..=max_bound.y {
            for z in min_bound.z..=max_bound.z {
                let pos = Vec3::new(x as f32, y as f32, z as f32);
                let distance = pos.distance(center);
                
                if distance <= radius {
                    let voxel_pos = IVec3::new(x, y, z);
                    match mode {
                        BrushMode::Add => {
                            batch.ops.push(EditOp::Set(voxel_pos));
                        }
                        BrushMode::Subtract => {
                            batch.ops.push(EditOp::Clear(voxel_pos));
                        }
                        BrushMode::Paint => {
                            // For now, paint is the same as add
                            // In the future, this could apply different materials
                            batch.ops.push(EditOp::Set(voxel_pos));
                        }
                    }
                }
            }
        }
    }
}

/// Apply a box brush operation to the voxel world
pub fn apply_box_brush(
    batch: &mut EditBatch,
    center: Vec3,
    half_extents: Vec3,
    mode: BrushMode,
) {
    let min_bound = (center - half_extents).as_ivec3().max(IVec3::ZERO);
    let max_bound = (center + half_extents).as_ivec3().min(IVec3::splat(31));
    
    for x in min_bound.x..=max_bound.x {
        for y in min_bound.y..=max_bound.y {
            for z in min_bound.z..=max_bound.z {
                let voxel_pos = IVec3::new(x, y, z);
                match mode {
                    BrushMode::Add => {
                        batch.ops.push(EditOp::Set(voxel_pos));
                    }
                    BrushMode::Subtract => {
                        batch.ops.push(EditOp::Clear(voxel_pos));
                    }
                    BrushMode::Paint => {
                        batch.ops.push(EditOp::Set(voxel_pos));
                    }
                }
            }
        }
    }
}

/// Cast a ray from camera to world position for voxel editing
pub fn cast_editing_ray(
    camera: &Camera,
    camera_transform: &GlobalTransform,
    cursor_pos: Vec2,
    mode: RaycastMode,
) -> Option<Vec3> {
    let ray = camera.viewport_to_world(camera_transform, cursor_pos).ok()?;
    
    match mode {
        RaycastMode::HorizontalPlane { y } => {
            // Find intersection with a horizontal plane
            let ray_dir_y = ray.direction.y;
            if ray_dir_y.abs() > 0.001 { // Avoid division by zero
                let t = (y - ray.origin.y) / ray_dir_y;
                if t > 0.0 { // Ray goes towards the plane
                    let intersection_point = ray.origin + ray.direction * t;
                    
                    // Only return if intersection is within voxel bounds
                    if intersection_point.x >= 0.0 && intersection_point.x < 32.0 &&
                       intersection_point.z >= 0.0 && intersection_point.z < 32.0 {
                        return Some(intersection_point);
                    }
                }
            }
        }
        RaycastMode::FixedDistance { distance } => {
            let brush_center = ray.origin + ray.direction * distance;
            
            // Clamp to voxel bounds
            let clamped_center = Vec3::new(
                brush_center.x.clamp(0.0, 31.0),
                brush_center.y.clamp(0.0, 31.0),
                brush_center.z.clamp(0.0, 31.0),
            );
            
            return Some(clamped_center);
        }
    }
    
    None
}

/// Different modes for raycasting in voxel editing
#[derive(Debug, Clone, Copy)]
pub enum RaycastMode {
    /// Cast to a horizontal plane at the given Y coordinate
    HorizontalPlane { y: f32 },
    /// Cast to a fixed distance from the camera
    FixedDistance { distance: f32 },
}

/// Generate procedural terrain using multiple noise octaves
pub fn generate_terrain(
    batch: &mut EditBatch,
    seed: f32,
    size: (u32, u32, u32), // (width, height, depth)
) {
    for x in 0..size.0 {
        for z in 0..size.2 {
            let fx = x as f32;
            let fz = z as f32;
            
            // Multiple noise octaves for more interesting terrain
            let noise1 = ((fx * 0.15 + seed * 0.001).sin() * (fz * 0.12 + seed * 0.002).cos()).abs();
            let noise2 = ((fx * 0.3 + seed * 0.003).sin() * (fz * 0.25 + seed * 0.004).cos()).abs() * 0.5;
            let noise3 = ((fx * 0.6 + seed * 0.005).sin() * (fz * 0.5 + seed * 0.006).cos()).abs() * 0.25;
            
            let height = ((noise1 + noise2 + noise3) * 8.0 + 4.0) as u32;
            
            for y in 0..height.min(size.1) {
                batch.ops.push(EditOp::Set(IVec3::new(x as i32, y as i32, z as i32)));
            }
        }
    }
}
