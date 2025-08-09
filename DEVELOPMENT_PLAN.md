# Bevy Gen Gaussian - Development Plan

## Overview

This crate provides utility functions and components for creating, manipulating, and working with Gaussian clouds in the Bevy engine, specifically designed to complement the `bevy_gaussian_splatting` crate.

## Core Philosophy

- **Modular Design**: Each module should be self-contained and composable
- **Pure Functions**: Prefer pure functions over stateful systems where possible
- **Seamless Integration**: Easy to pipe outputs from one module to another
- **Performance**: Efficient algorithms suitable for real-time applications

## Module Structure

### 1. `gaussian` Module - Gaussian Cloud Operations
**Purpose**: Create, manipulate, and transform Gaussian clouds

**Key Functions**:
- `mesh_to_gaussians(mesh, transform, settings) -> Vec<Gaussian3d>`
- `points_to_gaussians(points, normals, settings) -> Vec<Gaussian3d>`
- `interpolate_clouds(cloud_a, cloud_b, t) -> Vec<Gaussian3d>`
- `morph_cloud(cloud, target_positions, t) -> Vec<Gaussian3d>`
- `animate_cloud(cloud, animation_fn, time) -> Vec<Gaussian3d>`
- `paint_cloud(cloud, brush, position, settings) -> Vec<Gaussian3d>`
- `filter_cloud(cloud, predicate) -> Vec<Gaussian3d>`
- `transform_cloud(cloud, transform) -> Vec<Gaussian3d>`

**Types**:
- `GaussianSettings` - controls appearance (scale, opacity, color modes)
- `MeshConversionSettings` - vertex/edge/face processing options
- `AnimationCurve` - time-based transformation definitions
- `PaintBrush` - brush settings for cloud modification

### 2. `voxel` Module - Voxel Operations (Currently Implemented)
**Purpose**: Voxel-based operations and conversions

**Current Status**: Implemented but needs refactoring into clean API
**Key Functions** (to be cleaned up):
- `sdf_to_voxels(sdf_fn, bounds, resolution) -> VoxelGrid`
- `voxels_to_gaussians(voxels, settings) -> Vec<Gaussian3d>`
- `edit_voxels(voxels, brush, position) -> VoxelGrid`

### 3. `sdf` Module - Signed Distance Function Operations
**Purpose**: Create and manipulate signed distance functions

**Key Functions**:
- `sdf_to_gaussians(sdf_fn, bounds, resolution, settings) -> Vec<Gaussian3d>`
- `combine_sdfs(sdf_a, sdf_b, operation) -> SDF`
- `transform_sdf(sdf, transform) -> SDF`
- `animate_sdf(sdf, animation_fn) -> SDF`
- `noise_sdf(noise_fn, amplitude) -> SDF`

**Types**:
- `SDF` - trait for signed distance functions
- `SDFOperation` - union, intersection, subtraction
- `NoiseFunction` - various noise implementations

### 4. `debug` Module - Debug and Visualization Utilities
**Purpose**: Reusable debug UI components and visualization tools

**Key Components**:
- `GaussianCountDisplay` - shows current gaussian count
- `PerformanceMetrics` - FPS, render time, memory usage
- `CloudInspector` - inspect individual gaussians
- `SDFVisualizer` - visualize SDF fields
- `VoxelGridDisplay` - show voxel grid information

## Implementation Phases

### Phase 1: Core Gaussian Module (Current Priority)
1. Extract mesh-to-gaussian conversion from the mesh_to_cloud example
2. Create clean, pure function API
3. Add basic cloud transformation functions
4. Implement interpolation and morphing

### Phase 2: SDF Integration
1. Create SDF trait and basic implementations
2. Direct SDF-to-gaussian conversion
3. SDF composition and transformation functions
4. Integration with noise functions

### Phase 3: Voxel Module Refactoring
1. Clean up existing voxel code into pure functions
2. Remove ECS dependencies from core algorithms
3. Create clean voxel-to-gaussian pipeline
4. Optimize for performance

### Phase 4: Advanced Features
1. Animation system for cloud morphing
2. Paint/brush system for cloud editing
3. Advanced debug tools and visualizers
4. Performance optimizations and GPU compute

## API Design Principles

### Composability
```rust
// Functions should be easily chainable
let cloud = sdf_sphere(1.0)
    .combine(sdf_cube(0.8), SDFOperation::Union)
    .to_gaussians(bounds, resolution, settings)
    .transform(Transform::from_translation(Vec3::Y))
    .filter(|g| g.position.y > 0.0);
```

### Pure Functions
```rust
// Prefer pure functions over systems
fn mesh_to_gaussians(
    mesh: &Mesh, 
    transform: Transform, 
    settings: &MeshConversionSettings
) -> Vec<Gaussian3d> {
    // Pure function - no side effects
}
```

### Type Safety
```rust
// Use strong types for clarity
struct GaussianCloud(Vec<Gaussian3d>);
struct VoxelGrid { /* ... */ }
struct SDF(Box<dyn Fn(Vec3) -> f32>);
```

## File Structure After Refactoring

```
src/
├── lib.rs              # Main crate interface, re-exports
├── gaussian/           # Gaussian cloud operations
│   ├── mod.rs
│   ├── creation.rs     # mesh_to_gaussians, points_to_gaussians
│   ├── transform.rs    # transform, filter, animate
│   ├── morph.rs        # interpolate, morph between clouds
│   └── paint.rs        # brush-based editing
├── sdf/                # Signed distance functions
│   ├── mod.rs
│   ├── primitives.rs   # sphere, cube, etc.
│   ├── operations.rs   # union, intersection, etc.
│   ├── conversion.rs   # sdf_to_gaussians
│   └── noise.rs        # noise-based SDFs
├── voxel/              # Voxel operations (refactored)
│   ├── mod.rs
│   ├── grid.rs         # VoxelGrid type and operations
│   ├── conversion.rs   # voxels_to_gaussians, sdf_to_voxels
│   └── editing.rs      # voxel editing functions
└── debug/              # Debug utilities
    ├── mod.rs
    ├── metrics.rs      # performance metrics
    ├── ui.rs           # debug UI components
    └── visualizer.rs   # visualization helpers
```

## Next Steps

1. Create the new modular structure
2. Extract and clean up mesh_to_gaussians from the example
3. Implement basic gaussian transformation functions
4. Add comprehensive documentation and examples
5. Create integration tests showing the composability

This plan prioritizes modularity, composability, and clean APIs while building on the existing work in a more organized way.
