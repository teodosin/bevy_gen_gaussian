use std::collections::HashSet;
use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology, VertexAttributeValues};
use bevy_gaussian_splatting::Gaussian3d;

use crate::gaussian::settings::{MeshConversionSettings, PointCloudSettings, ColorMode};

/// Convert a mesh into Gaussian3d instances for vertices, edges, and faces
/// 
/// This is a pure function that takes a mesh and produces gaussians without side effects.
/// It can generate gaussians for vertices, edges (connecting vertices), and faces (triangle centers).
pub fn mesh_to_gaussians(
    mesh: &Mesh, 
    transform: Transform, 
    settings: &MeshConversionSettings
) -> Vec<Gaussian3d> {
    let topology = mesh.primitive_topology();
    let positions = match read_positions(mesh) {
        Some(v) => v,
        None => {
            warn!("mesh_to_gaussians: mesh missing positions");
            return Vec::new();
        }
    };
    let normals_opt = read_normals(mesh);

    // Build index buffer as u32
    let indices_u32: Option<Vec<u32>> = match mesh.indices() {
        Some(Indices::U32(ix)) => Some(ix.clone()),
        Some(Indices::U16(ix)) => Some(ix.iter().map(|&x| x as u32).collect()),
        None => None,
    };

    // Vertex normals: either from attribute or computed from faces
    let vertex_normals = normals_opt.unwrap_or_else(|| 
        compute_vertex_normals(topology, &positions, indices_u32.as_ref())
    );

    let mut out: Vec<Gaussian3d> = Vec::new();

    // 1) Vertices
    if settings.include_vertices {
        for (vpos, vnorm) in positions.iter().zip(vertex_normals.iter()) {
            let pos = transform.transform_point(*vpos);
            let rot = Quat::IDENTITY;
            let scale = Vec3::splat(settings.vertex_scale);
            out.push(gaussian_from_transform(pos, rot, scale, *vnorm, settings.opacity));
        }
    }

    // For edges and faces we need indices and triangles
    if let Some(indices) = indices_u32 {
        // 2) Faces: assumes triangle topology
        if settings.include_faces {
            let tri_iter = triangles_from(topology, &indices);
            let tris: Vec<[u32; 3]> = tri_iter.collect();
            for tri in &tris {
                let p0 = positions[tri[0] as usize];
                let p1 = positions[tri[1] as usize];
                let p2 = positions[tri[2] as usize];

                let centroid = (p0 + p1 + p2) / 3.0;

                let u = p1 - p0;
                let v = p2 - p0;

                let x_axis = u.normalize_or_zero();
                let z_axis = u.cross(v).normalize_or_zero();
                let y_axis = z_axis.cross(x_axis);

                let rot = Quat::from_mat3(&Mat3::from_cols(x_axis, y_axis, z_axis));

                let u_len = u.length();
                let v_on_y = v.dot(y_axis).abs();

                let scale = Vec3::new(u_len, v_on_y, settings.face_scale);
                let face_n = z_axis;

                out.push(gaussian_from_transform(
                    transform.transform_point(centroid),
                    rot,
                    scale,
                    face_n,
                    settings.opacity,
                ));
            }
        }

        // 3) Edges: dedupe undirected
        if settings.include_edges {
            let tri_iter = triangles_from(topology, &indices);
            let tris: Vec<[u32; 3]> = tri_iter.collect();
            let mut edge_set: HashSet<(u32, u32)> = HashSet::new();
            
            for tri in &tris {
                let edges = [
                    (tri[0], tri[1]),
                    (tri[1], tri[2]),
                    (tri[2], tri[0]),
                ];
                for (a, b) in edges {
                    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                    if edge_set.insert((lo, hi)) {
                        let pa = positions[lo as usize];
                        let pb = positions[hi as usize];
                        let mid = (pa + pb) * 0.5;
                        let na = vertex_normals[lo as usize];
                        let nb = vertex_normals[hi as usize];
                        let n = (na + nb).normalize_or_zero();

                        let edge_vec = pb - pa;
                        let rot = Quat::from_rotation_arc(Vec3::X, edge_vec.normalize_or_zero());
                        let scale = Vec3::new(edge_vec.length(), settings.edge_scale, settings.edge_scale);

                        out.push(gaussian_from_transform(
                            transform.transform_point(mid),
                            rot,
                            scale,
                            n,
                            settings.opacity,
                        ));
                    }
                }
            }
        }
    } else {
        // No indices; treat as point cloud of vertices only
        debug!("mesh_to_gaussians: mesh had no indices; produced only vertex splats");
    }

    out
}

/// Convert a point cloud (positions + optional normals) to Gaussians
pub fn points_to_gaussians(
    positions: &[Vec3],
    normals: Option<&[Vec3]>,
    transform: Transform,
    settings: &PointCloudSettings,
) -> Vec<Gaussian3d> {
    let mut out = Vec::new();
    
    for (i, &pos) in positions.iter().enumerate() {
        let world_pos = transform.transform_point(pos);
        let normal = if let Some(normals) = normals {
            normals.get(i).copied().unwrap_or(Vec3::Y)
        } else {
            // Use position as normal if no normals provided
            pos.normalize_or_zero()
        };
        
        let rot = Quat::IDENTITY;
        let scale = Vec3::splat(settings.scale);
        
        out.push(gaussian_from_transform(
            world_pos, 
            rot, 
            scale, 
            normal, 
            settings.opacity
        ));
    }
    
    out
}

// Helper function to get triangles from indices based on topology
fn triangles_from(topology: PrimitiveTopology, indices: &[u32]) -> impl Iterator<Item = [u32; 3]> + '_ {
    match topology {
        PrimitiveTopology::TriangleList => {
            Box::new(indices.chunks_exact(3).map(|c| [c[0], c[1], c[2]])) 
                as Box<dyn Iterator<Item = [u32; 3]> + '_>
        },
        _ => {
            warn!("mesh_to_gaussians: non-triangle topology {:?} not fully supported; attempting naive 3-chunking", topology);
            Box::new(indices.chunks(3).filter(|c| c.len() == 3).map(|c| [c[0], c[1], c[2]]))
        }
    }
}

// --- Mesh attribute readers ---

fn read_positions(mesh: &Mesh) -> Option<Vec<Vec3>> {
    let attr = Mesh::ATTRIBUTE_POSITION;
    mesh.attribute(attr).and_then(|a| {
        match a {
            VertexAttributeValues::Float32x3(v) => {
                Some(v.iter().map(|p| Vec3::from_slice(p)).collect())
            }
            VertexAttributeValues::Float32x2(v) => {
                Some(v.iter().map(|p| Vec3::new(p[0], p[1], 0.0)).collect())
            }
            VertexAttributeValues::Float32x4(v) => {
                Some(v.iter().map(|p| Vec3::new(p[0], p[1], p[2])).collect())
            }
            VertexAttributeValues::Uint32x3(v) => {
                Some(v.iter().map(|p| Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32)).collect())
            }
            _ => None,
        }
    })
}

fn read_normals(mesh: &Mesh) -> Option<Vec<Vec3>> {
    let attr = Mesh::ATTRIBUTE_NORMAL;
    mesh.attribute(attr).and_then(|a| {
        match a {
            VertexAttributeValues::Float32x3(v) => {
                Some(v.iter().map(|p| Vec3::from_slice(p)).collect())
            }
            VertexAttributeValues::Float32x4(v) => {
                Some(v.iter().map(|p| Vec3::new(p[0], p[1], p[2])).collect())
            }
            VertexAttributeValues::Uint32x3(v) => {
                Some(v.iter().map(|p| Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32)).collect())
            }
            _ => None,
        }
    })
}

// Compute per-vertex normals if missing
fn compute_vertex_normals(
    topology: PrimitiveTopology, 
    positions: &[Vec3], 
    indices: Option<&Vec<u32>>
) -> Vec<Vec3> {
    let mut normals = vec![Vec3::ZERO; positions.len()];

    if let Some(ix) = indices {
        for tri in triangles_from(topology, ix) {
            let p0 = positions[tri[0] as usize];
            let p1 = positions[tri[1] as usize];
            let p2 = positions[tri[2] as usize];
            let n = face_normal(p0, p1, p2);
            normals[tri[0] as usize] += n;
            normals[tri[1] as usize] += n;
            normals[tri[2] as usize] += n;
        }
    }

    for n in &mut normals {
        *n = n.normalize_or_zero();
    }
    normals
}

fn face_normal(p0: Vec3, p1: Vec3, p2: Vec3) -> Vec3 {
    (p1 - p0).cross(p2 - p0).normalize_or_zero()
}

fn normal_to_rgb(n: Vec3) -> [f32; 3] {
    let c = (n * 0.5) + Vec3::splat(0.5);
    [c.x, c.y, c.z]
}

// Construct a Gaussian3d from a transform, a normal for color, and an opacity.
fn gaussian_from_transform(
    pos: Vec3,
    rot: Quat,
    scale: Vec3,
    norm: Vec3,
    opacity: f32,
) -> Gaussian3d {
    let mut g = Gaussian3d::default();
    
    // position + visibility
    g.position_visibility.position = pos.to_array();
    g.position_visibility.visibility = 1.0;

    // rotation
    g.rotation.rotation = rot.to_array();

    // scale and opacity
    g.scale_opacity.scale = scale.to_array();
    g.scale_opacity.opacity = opacity;

    // Color via SH DC coefficients
    // With sh0 feature: sh = (rgb - 0.5) / 0.2821
    let rgb = normal_to_rgb(norm);
    g.spherical_harmonic.set(0, (rgb[0] - 0.5) / 0.2821);
    g.spherical_harmonic.set(1, (rgb[1] - 0.5) / 0.2821);
    g.spherical_harmonic.set(2, (rgb[2] - 0.5) / 0.2821);
    
    // zero the rest for determinism
    for i in 3..bevy_gaussian_splatting::material::spherical_harmonics::SH_COEFF_COUNT {
        g.spherical_harmonic.set(i, 0.0);
    }

    g
}
