/// Settings for controlling mesh-to-Gaussian conversion
#[derive(Debug, Clone)]
pub struct MeshConversionSettings {
    /// Default scale for vertex gaussians
    pub vertex_scale: f32,
    /// Scale for edge gaussians
    pub edge_scale: f32,
    /// Scale for face gaussians
    pub face_scale: f32,
    /// Default opacity for all gaussians
    pub opacity: f32,
    /// Whether to generate gaussians for vertices
    pub include_vertices: bool,
    /// Whether to generate gaussians for edges
    pub include_edges: bool,
    /// Whether to generate gaussians for faces
    pub include_faces: bool,
}

impl Default for MeshConversionSettings {
    fn default() -> Self {
        Self {
            vertex_scale: 0.02,
            edge_scale: 0.015,
            face_scale: 0.03,
            opacity: 0.8,
            include_vertices: false,
            include_edges: false,
            include_faces: true,
        }
    }
}

/// Settings for point cloud to Gaussian conversion
#[derive(Debug, Clone)]
pub struct PointCloudSettings {
    /// Scale for point gaussians
    pub scale: f32,
    /// Opacity for point gaussians
    pub opacity: f32,
    /// Whether to use provided normals for color (if false, uses position-based color)
    pub use_normals_for_color: bool,
}

impl Default for PointCloudSettings {
    fn default() -> Self {
        Self {
            scale: 0.02,
            opacity: 0.8,
            use_normals_for_color: true,
        }
    }
}

/// Color mode for Gaussian generation
#[derive(Debug, Clone, Copy)]
pub enum ColorMode {
    /// Use surface normals to derive color
    Normal,
    /// Use a solid color for all gaussians
    Solid([f32; 3]),
    /// Use position-based color gradient
    Gradient { from: [f32; 3], to: [f32; 3] },
    /// Use random colors
    Random,
}
