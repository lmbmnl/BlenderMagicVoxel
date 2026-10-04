# BlenderMagicVoxel

MagicaVoxel-style voxel editor inside Blender 5.x: draw coloured voxels directly in the viewport, then bake them into a clean mesh.

## Features

- 255-colour palette (MagicaVoxel default), rename / add / remove colours, eyedropper
- Brushes: point, square, triangle, hexagon, circle, sphere (filled or outline), box, line, face extrude, fill
- Live Mirror toggle (M) for every brush, with the mirror plane drawn in the viewport: across the
  first voxel (spawned on the 3D cursor), or across the model centre when the size limit is on
- Placement preview with a thick outline, translucent fill and see-through hint
- Select, move with a snapped gizmo, rotate, flip, mirror, duplicate, copy / paste
- Rotate gizmo: drag a ring around the selection in 45° steps (90° steps exact, 45° approximated)
- `.vox` import / export with palette
- Greedy meshing on Confirm, voxel size in scene units
- Own undo / redo, shortcut legend in the viewport
- Viewport panels (RetopoFlow style) while drawing: Tools, Brush, Session, Symmetry, Palette
  (resizable), Selection, Scene, Shortcuts. Drag a header to move a panel, ▾ to collapse it;
  places and sizes are saved in the .blend. U hides them, the sidebar panels stay available

## Install

Blender → Edit → Preferences → Get Extensions → ⌄ → **Install from Disk…** and pick the zip
built with:

```
blender --command extension build --source-dir . --output-dir ..
```

Then open the **VoxelDraw** tab in the 3D View sidebar (N) and press **Start Voxel**.
Esc (or Exit in the Session panel) stops the tool; Start Voxel resumes the session.

## Tests

```
blender -b --factory-startup --python test_voxeldraw.py
blender -b --factory-startup --python test_vd_ui.py
```

## Inspiration

Inspired by [VoxelDraw](https://github.com/theunnecessarythings/VoxelDraw) by Sreeraj R (MIT, 2020)
and by [MagicaVoxel](https://ephtracy.github.io/) by ephtracy.

## License

MIT
