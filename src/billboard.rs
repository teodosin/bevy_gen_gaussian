use bevy::prelude::*;
use crate::extraction::SurfaceBuffer;
use crate::metrics::Metrics;
use bevy::render::render_resource::{Buffer, BufferUsages};
use bevy::render::renderer::RenderDevice;
use bevy::render::render_resource::BufferInitDescriptor; // re-exported type
use bytemuck::{Pod, Zeroable};

#[derive(Component)]
pub struct BillboardTag;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuBillboard { pub pos: [f32;3], pub color: [u8;4] }

#[derive(Resource, Default)]
pub struct BillboardGpu { pub buffer: Option<Buffer>, pub count: u32 }

pub fn update_billboard_instances(
    mut gpu: ResMut<BillboardGpu>,
    mut surf: ResMut<SurfaceBuffer>,
    mut metrics: ResMut<Metrics>,
    render_device: Res<RenderDevice>,
) {
    if !surf.dirty { return; }
    
    let mut data = Vec::with_capacity(surf.instances.len());
    for inst in &surf.instances {
        data.push(GpuBillboard { pos: inst.pos.to_array(), color: inst.color });
    }
    let bytes = bytemuck::cast_slice(&data);
    let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("billboard_instances"),
        contents: bytes,
        usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
    });
    gpu.count = data.len() as u32;
    gpu.buffer = Some(buffer);
    metrics.instance_count = gpu.count as u64;
    
    // Reset dirty flag so we don't upload the same data every frame
    surf.dirty = false;
    
    println!("Billboard GPU: uploaded {} instances", gpu.count);
}
