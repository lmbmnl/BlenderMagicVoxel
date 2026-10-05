# VoxelDraw extension repository

This branch is a Blender extensions repository: Blender reads `index.json` and downloads
the zip next to it. It holds no source code (that is on the development branch).

## Add it to Blender (4.2 or newer)

1. Edit → Preferences → System → Network: enable **Allow Online Access**.
2. Get Extensions → Repositories (the ⌄ menu at the top right) → **+** → **Add Remote Repository**, URL:

   ```
   https://raw.githubusercontent.com/lmbmnl/BlenderMagicVoxel/extensions/index.json
   ```

3. Install **VoxelDraw** from Get Extensions. New versions then show up there with an
   update button (or "Check for Updates"): no uninstall needed.

## Publish a new version

```
blender --command extension build --source-dir <source checkout> --output-dir <this branch>
blender --command extension server-generate --repo-dir <this branch>
```

Remove the old zip before `server-generate` if only the latest version should be offered,
then commit `index.json` and the zip to this branch.
