# bevy_gen_gaussian (Prototype)

Foundational voxel + surface (Gaussian / billboard) generation & editing crate.

## Goals (MVP)
- Fixed-size test volume (32³) voxel edits
- Sphere SDF fill/carve
- Surface extraction -> billboard instances
- Free-fly camera example
- Metrics logging

## Roadmap (Short)
1. constants / types
2. simple voxel store + edit ops
3. extraction -> billboard instancing
4. example: basic_view
5. SDF sphere op
6. dirty region refinement
7. tree backend swap (later)

## Feature Flags
- `billboard_mode` (default): instanced quads
- `gaussian_mode`: (placeholder for splat cluster path)

## Examples
Run:
```bash
cargo run -p bevy_gen_gaussian --example basic_view
```
While running press `F` to populate a patterned test fill.
