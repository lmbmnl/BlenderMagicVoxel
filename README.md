# BlenderMagicVoxel

MagicaVoxel-style voxel editor inside Blender 5.x: draw coloured voxels directly in the viewport, then bake them into a clean mesh.

## Features

- 255-colour palette (MagicaVoxel default), rename / add / remove colours, eyedropper
- Brushes: point, square, triangle, hexagon, circle, sphere (filled or outline), box, line, face extrude, fill
- Live Mirror toggle (M) for every brush, with the mirror plane drawn in the viewport: across the
  first voxel (spawned on the 3D cursor), or across the model centre when the size limit is on
- Placement preview with a thick outline, translucent fill and see-through hint
- Select, move with a snapped gizmo, rotate, flip, mirror, duplicate, copy / paste
- `.vox` import / export with palette
- Greedy meshing on Confirm, voxel size in scene units
- Own undo / redo, shortcut legend in the viewport

## Install

Blender → Edit → Preferences → Get Extensions → ⌄ → **Install from Disk…** and pick the zip
built with:

```
blender --command extension build --source-dir . --output-dir ..
```

Then open the **VoxelDraw** tab in the 3D View sidebar (N) and press **Start Voxel**.

## Tests

```
blender -b --factory-startup --python test_voxeldraw.py
```

## Inspiration

Inspired by [VoxelDraw](https://github.com/theunnecessarythings/VoxelDraw) by Sreeraj R (MIT, 2020)
and by [MagicaVoxel](https://ephtracy.github.io/) by ephtracy.

## License

MIT
