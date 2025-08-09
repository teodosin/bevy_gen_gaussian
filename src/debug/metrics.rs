use bevy::prelude::*;

/// Simple metrics for tracking Gaussian cloud information
#[derive(Resource, Default, Debug)]
pub struct GaussianMetrics {
    pub total_gaussians: usize,
    pub last_frame_time: f32,
    pub fps: f32,
}

/// System to update FPS metrics
pub fn update_metrics(time: Res<Time>, mut metrics: ResMut<GaussianMetrics>) {
    metrics.last_frame_time = time.delta_secs();
    metrics.fps = 1.0 / time.delta_secs();
}

/// System to count gaussians in the scene (placeholder - would need actual implementation)
pub fn count_gaussians(mut metrics: ResMut<GaussianMetrics>) {
    // This would need to query for actual Gaussian entities in a real implementation
    // For now it's just a placeholder
    metrics.total_gaussians = 0;
}

/// Debug overlay system for displaying metrics
pub fn debug_overlay(
    mut gizmos: Gizmos,
    metrics: Res<GaussianMetrics>,
) {
    // Simple text overlay would go here
    // This is a placeholder for now
    info!("FPS: {:.1}, Gaussians: {}", metrics.fps, metrics.total_gaussians);
}
