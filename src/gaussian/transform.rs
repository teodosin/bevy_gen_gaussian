use bevy::prelude::*;
use bevy_gaussian_splatting::Gaussian3d;

/// Transform a cloud of Gaussians by applying a Transform to all positions
pub fn transform_cloud(gaussians: &[Gaussian3d], transform: Transform) -> Vec<Gaussian3d> {
    gaussians.iter().map(|g| {
        let mut new_g = *g;
        let pos = Vec3::from_array(g.position_visibility.position);
        let new_pos = transform.transform_point(pos);
        new_g.position_visibility.position = new_pos.to_array();
        new_g
    }).collect()
}

/// Filter a cloud of Gaussians based on a predicate function
pub fn filter_cloud<F>(gaussians: &[Gaussian3d], predicate: F) -> Vec<Gaussian3d>
where
    F: Fn(&Gaussian3d) -> bool,
{
    gaussians.iter().filter(|g| predicate(g)).copied().collect()
}

/// Scale all Gaussians in a cloud by a uniform factor
pub fn scale_cloud(gaussians: &[Gaussian3d], scale_factor: f32) -> Vec<Gaussian3d> {
    gaussians.iter().map(|g| {
        let mut new_g = *g;
        let scale = Vec3::from_array(g.scale_opacity.scale);
        let new_scale = scale * scale_factor;
        new_g.scale_opacity.scale = new_scale.to_array();
        new_g
    }).collect()
}

/// Change the opacity of all Gaussians in a cloud
pub fn set_cloud_opacity(gaussians: &[Gaussian3d], opacity: f32) -> Vec<Gaussian3d> {
    gaussians.iter().map(|g| {
        let mut new_g = *g;
        new_g.scale_opacity.opacity = opacity.clamp(0.0, 1.0);
        new_g
    }).collect()
}

/// Linearly interpolate between two clouds of Gaussians
/// 
/// If the clouds have different sizes, the smaller cloud is repeated cyclically
/// to match the larger one.
pub fn interpolate_clouds(
    cloud_a: &[Gaussian3d], 
    cloud_b: &[Gaussian3d], 
    t: f32
) -> Vec<Gaussian3d> {
    let t = t.clamp(0.0, 1.0);
    let max_len = cloud_a.len().max(cloud_b.len());
    
    let mut result = Vec::with_capacity(max_len);
    
    for i in 0..max_len {
        let a = &cloud_a[i % cloud_a.len()];
        let b = &cloud_b[i % cloud_b.len()];
        
        let mut new_g = *a;
        
        // Interpolate position
        let pos_a = Vec3::from_array(a.position_visibility.position);
        let pos_b = Vec3::from_array(b.position_visibility.position);
        let new_pos = pos_a.lerp(pos_b, t);
        new_g.position_visibility.position = new_pos.to_array();
        
        // Interpolate scale
        let scale_a = Vec3::from_array(a.scale_opacity.scale);
        let scale_b = Vec3::from_array(b.scale_opacity.scale);
        let new_scale = scale_a.lerp(scale_b, t);
        new_g.scale_opacity.scale = new_scale.to_array();
        
        // Interpolate opacity
        let opacity_a = a.scale_opacity.opacity;
        let opacity_b = b.scale_opacity.opacity;
        new_g.scale_opacity.opacity = opacity_a * (1.0 - t) + opacity_b * t;
        
        // Interpolate rotation (slerp)
        let rot_a = Quat::from_array(a.rotation.rotation);
        let rot_b = Quat::from_array(b.rotation.rotation);
        let new_rot = rot_a.slerp(rot_b, t);
        new_g.rotation.rotation = new_rot.to_array();
        
        // For spherical harmonics, we'll do linear interpolation
        // This isn't physically accurate but provides smooth transitions
        for i in 0..bevy_gaussian_splatting::material::spherical_harmonics::SH_COEFF_COUNT {
            let sh_a = a.spherical_harmonic.coefficients.get(i).copied().unwrap_or(0.0);
            let sh_b = b.spherical_harmonic.coefficients.get(i).copied().unwrap_or(0.0);
            new_g.spherical_harmonic.set(i, sh_a * (1.0 - t) + sh_b * t);
        }
        
        result.push(new_g);
    }
    
    result
}

/// Animate a cloud of Gaussians using a time-based function
pub fn animate_cloud<F>(gaussians: &[Gaussian3d], time: f32, animation_fn: F) -> Vec<Gaussian3d>
where
    F: Fn(&Gaussian3d, usize, f32) -> Gaussian3d,
{
    gaussians.iter().enumerate().map(|(index, g)| {
        animation_fn(g, index, time)
    }).collect()
}

/// Simple wave animation that moves Gaussians up and down
pub fn wave_animation(gaussian: &Gaussian3d, index: usize, time: f32) -> Gaussian3d {
    let mut new_g = *gaussian;
    let pos = Vec3::from_array(gaussian.position_visibility.position);
    let wave_offset = (time * 2.0 + index as f32 * 0.1).sin() * 0.5;
    let new_pos = pos + Vec3::new(0.0, wave_offset, 0.0);
    new_g.position_visibility.position = new_pos.to_array();
    new_g
}

/// Rotation animation that spins Gaussians around their center
pub fn rotation_animation(gaussian: &Gaussian3d, _index: usize, time: f32) -> Gaussian3d {
    let mut new_g = *gaussian;
    let rotation = Quat::from_rotation_y(time);
    let current_rot = Quat::from_array(gaussian.rotation.rotation);
    let new_rot = current_rot * rotation;
    new_g.rotation.rotation = new_rot.to_array();
    new_g
}

/// Combine multiple clouds into a single cloud
pub fn combine_clouds(clouds: &[&[Gaussian3d]]) -> Vec<Gaussian3d> {
    let total_size: usize = clouds.iter().map(|c| c.len()).sum();
    let mut result = Vec::with_capacity(total_size);
    
    for cloud in clouds {
        result.extend_from_slice(cloud);
    }
    
    result
}
