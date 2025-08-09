use bevy::prelude::*;
use super::primitives::{SDF, BoxedSDF};

/// SDF operation types
#[derive(Debug, Clone, Copy)]
pub enum SDFOperation {
    Union,
    Intersection,
    Subtraction,
    SmoothUnion(f32),
    SmoothIntersection(f32),
    SmoothSubtraction(f32),
}

/// Combined SDF that applies an operation to two SDFs
pub struct CombinedSDF {
    pub left: BoxedSDF,
    pub right: BoxedSDF,
    pub operation: SDFOperation,
}

impl SDF for CombinedSDF {
    fn distance(&self, point: Vec3) -> f32 {
        let d1 = self.left.distance(point);
        let d2 = self.right.distance(point);
        
        match self.operation {
            SDFOperation::Union => d1.min(d2),
            SDFOperation::Intersection => d1.max(d2),
            SDFOperation::Subtraction => d1.max(-d2),
            SDFOperation::SmoothUnion(k) => smooth_min(d1, d2, k),
            SDFOperation::SmoothIntersection(k) => smooth_max(d1, d2, k),
            SDFOperation::SmoothSubtraction(k) => smooth_max(d1, -d2, k),
        }
    }
}

/// Transformed SDF that applies a transform to the input coordinates
pub struct TransformedSDF {
    pub sdf: BoxedSDF,
    pub inverse_transform: Mat4,
}

impl SDF for TransformedSDF {
    fn distance(&self, point: Vec3) -> f32 {
        let local_point = self.inverse_transform.transform_point3(point);
        self.sdf.distance(local_point)
    }
}

/// Combine two SDFs with an operation
pub fn combine_sdfs(left: BoxedSDF, right: BoxedSDF, operation: SDFOperation) -> BoxedSDF {
    Box::new(CombinedSDF {
        left,
        right,
        operation,
    })
}

/// Transform an SDF
pub fn transform_sdf(sdf: BoxedSDF, transform: Transform) -> BoxedSDF {
    let inverse_transform = transform.compute_matrix().inverse();
    Box::new(TransformedSDF {
        sdf,
        inverse_transform,
    })
}

/// Smooth minimum function for smooth unions
fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    a * h + b * (1.0 - h) - k * h * (1.0 - h)
}

/// Smooth maximum function for smooth intersections
fn smooth_max(a: f32, b: f32, k: f32) -> f32 {
    -smooth_min(-a, -b, k)
}

/// Convenience methods for chaining operations
pub trait SDFExt {
    fn union(self, other: BoxedSDF) -> BoxedSDF;
    fn intersection(self, other: BoxedSDF) -> BoxedSDF;
    fn subtraction(self, other: BoxedSDF) -> BoxedSDF;
    fn smooth_union(self, other: BoxedSDF, smoothness: f32) -> BoxedSDF;
    fn transform(self, transform: Transform) -> BoxedSDF;
}

impl SDFExt for BoxedSDF {
    fn union(self, other: BoxedSDF) -> BoxedSDF {
        combine_sdfs(self, other, SDFOperation::Union)
    }
    
    fn intersection(self, other: BoxedSDF) -> BoxedSDF {
        combine_sdfs(self, other, SDFOperation::Intersection)
    }
    
    fn subtraction(self, other: BoxedSDF) -> BoxedSDF {
        combine_sdfs(self, other, SDFOperation::Subtraction)
    }
    
    fn smooth_union(self, other: BoxedSDF, smoothness: f32) -> BoxedSDF {
        combine_sdfs(self, other, SDFOperation::SmoothUnion(smoothness))
    }
    
    fn transform(self, transform: Transform) -> BoxedSDF {
        transform_sdf(self, transform)
    }
}
