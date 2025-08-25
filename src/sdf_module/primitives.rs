use bevy::prelude::*;





/// Trait for signed distance functions
pub trait SDF: Send + Sync {
    fn distance(&self, point: Vec3) -> f32;
}

/// A boxed SDF for dynamic dispatch
pub type BoxedSDF = Box<dyn SDF>;





/// Sphere SDF
#[derive(Debug, Clone)]
pub struct SphereSDF {
    pub center: Vec3,
    pub radius: f32,
}

impl SDF for SphereSDF {
    fn distance(&self, point: Vec3) -> f32 {
        (point - self.center).length() - self.radius
    }
}





/// Box SDF
#[derive(Debug, Clone)]
pub struct BoxSDF {
    pub center: Vec3,
    pub size: Vec3,
}

impl SDF for BoxSDF {
    fn distance(&self, point: Vec3) -> f32 {
        let d = (point - self.center).abs() - self.size * 0.5;
        d.max(Vec3::ZERO).length() + d.max_element().min(0.0)
    }
}





/// Plane SDF
#[derive(Debug, Clone)]
pub struct PlaneSDF {
    pub normal: Vec3,
    pub distance: f32,
}

impl SDF for PlaneSDF {
    fn distance(&self, point: Vec3) -> f32 {
        point.dot(self.normal) - self.distance
    }
}





/// Cylinder SDF
#[derive(Debug, Clone)]
pub struct CylinderSDF {
    pub center: Vec3,
    pub radius: f32,
    pub height: f32,
}

impl SDF for CylinderSDF {
    fn distance(&self, point: Vec3) -> f32 {
        let local_point = point - self.center;
        let xz_dist = Vec2::new(local_point.x, local_point.z).length() - self.radius;
        let y_dist = local_point.y.abs() - self.height * 0.5;
        
        if xz_dist < 0.0 && y_dist < 0.0 {
            xz_dist.max(y_dist)
        } else {
            Vec2::new(xz_dist.max(0.0), y_dist.max(0.0)).length()
        }
    }
}





/// Convenience functions for creating common SDFs

pub fn sdf_sphere(center: Vec3, radius: f32) -> BoxedSDF {
    Box::new(SphereSDF { center, radius })
}

pub fn sdf_box(center: Vec3, size: Vec3) -> BoxedSDF {
    Box::new(BoxSDF { center, size })
}

pub fn sdf_plane(normal: Vec3, distance: f32) -> BoxedSDF {
    Box::new(PlaneSDF { normal: normal.normalize(), distance })
}

pub fn sdf_cylinder(center: Vec3, radius: f32, height: f32) -> BoxedSDF {
    Box::new(CylinderSDF { center, radius, height })
}
