# Sim Studio · React edition

This is the source of the local simulator UI, built into `static/studio/` and served
by `python -m sim_engine.joint_lab`. Open `http://127.0.0.1:8765/lab/`.
The previous viewer is available at `/legacy`.

```powershell
cd web/joint-lab
npm ci
npm run build
npm test
```

For frontend development, keep the Python lab on port 8765 and run `npm run dev`.
Vite proxies the simulation endpoints to Python. Production use needs only the
Python command after building. All JS/CSS dependencies are bundled locally.

## Architecture and technology review

| Choice | Purpose and assessment |
| --- | --- |
| React 19 + TypeScript | Typed scene definitions and reusable panels; editor and replay share the same 3D renderer. React Three Fiber 9 matches React 19. |
| Vite | Small local SPA build; no Node server needed during normal simulator use. Build outputs are committed under `static/studio`. |
| Mantine 8 | Actual UI library for accessible inputs, sliders, drawers, switches, color selection, and responsive controls. No runtime CSS CDN. |
| Three.js + React Three Fiber + Drei | WebGL, original robot mesh buffers, lighting, orbit and transform controls. Transforms come from MuJoCo; there is no second JavaScript IK implementation. |
| Python + MuJoCo | Authoritative geometry, IK, gravity, contact, friction, and trajectories. Three.js displays states; switching rendering libraries does not improve the underlying robot calibration. |

The `/model?exact=1` endpoint preserves source mesh triangles (about 401k), at
seven decimal places in body coordinates. Editor and replay both apply body
rotation matrices and world positions from MuJoCo. This costs roughly 14.6 MB
of initial JSON before HTTP compression; geometry is memoized in each viewport
and rendering runs on demand. The appearance uses studio lighting and simple
materials, so it is geometrically faithful to the simulator rather than a
pixel-for-pixel copy of MuJoCo's renderer.

The XY editor has keyboard arrow alternatives and numeric inputs. Bowl dragging
and numeric entry project to the configured `PICK_REGION` boundary. Python
independently checks the bowl and selected target. An out-of-region request
produces a stationary, one-frame failed replay; it does not command a return
home or settle the arm first. This is a workspace guard, not a promise that
every permitted pose will successfully grasp or place.

Up to five 3 cm boxes/balls have IDs, shape, color, XY pose and yaw. They become
real MuJoCo free bodies with mass and collision geometry. Every object's position
and orientation is recorded per replay frame. The existing scripted policy
operates on the selected target; the other objects participate in physics as
obstacles. Automatic multi-object sequencing and color sorting are future work.
The existing controller is tuned for boxes, so sphere grasps can fail naturally.

Only one fixed `lab` seed is run; results replace `results/studio/replay_lab.json` and the
compatibility HTML. Batch seed lists, batch replay and batch step history are
never regenerated. `lab_case.json` stores the last submitted scene. Visibility
settings and the editable draft are versioned in browser localStorage. Both
replay renderers support all recorded objects, shapes and colors.

## Screens and extension points

- **Replay** defaults to the full original analysis UI, generated from the same
  replay template/CSS/JS. Its 3D viewport and wrist sensor now use React portals
  with the same Three.js/R3F renderer and exact mesh as Lab. Studio explicitly
  skips the legacy CPU Canvas 3D/camera draw routines; XY/YZ remain SVG. The old
  standalone HTML keeps its offline renderer for compatibility. A same-origin
  message adapter passes recorded body poses (or the original replay FK for old
  episodes), camera calibration, contact highlighting and trajectory overlays.
  Orbit/orthographic presets, follow, zoom and sensor tilt remain functional.
  GPU resources are reused. Camera and scene use one timestamped, speed-aware
  interpolation sample; the sensor is composed from the interpolated wrist and
  its calibrated local mount, so it stays rigidly attached during playback.
  Pausing and manual seeking snap to the recorded state. Hidden analysis panels
  do no playback work in focused 3D mode; SVG projections update at most 15 Hz
  during playback and exactly on pause/seek. A same-origin adapter synchronizes episode
  and step with Studio's URL without reloading during scrubbing. Existing batch
  replay data is read from `results/replay.json` (same data as the canvas
  `results/replay.html`); it is never regenerated here.
  Lab recordings also have a Three.js playback mode. Step bookmarks are local
  to this browser; exported replay JSON contains the episode, not bookmarks.
  Studio owns the single source/episode selector, result summary and fullscreen
  action; embedded legacy duplicates are hidden. The viewport owns transport,
  timeline and view presets. Sensor tilt/legend/shortcuts live in the Options
  disclosure; the bookmark editor opens from its toolbar button.
  Playback position reports update the URL without being echoed back as seek
  commands. Browser back/forward and explicit selections still seek. Changing
  view mode preserves the playback speed selected in the transport.
- **Lab** edits the shared scene and starts exactly one trial. Navigation and
  browser reload preserve the draft. Import/export uses validated JSON.
- **History** stores each completed result and initial scene at
  `results/studio/studio_runs/<run_id>/{case,payload,meta}.json`. Metadata can change while
  recorded states stay immutable. Favorites, notes, filtering, sorting,
  pagination and comparison operate on real runs. Rerun uses only that case.
- **Settings** controls theme, default route/playback, post-run navigation and
  Lab card visibility. Engine information comes from `/studio/health`; physical
  simulation settings remain owned by Python/config, rather than cosmetic toggles.

To add a screen, create `src/screens/YourScreen.tsx` and register its lazy
component, title and icon in `src/studio/screens.ts`. The shell automatically
supplies menu, active state, header, mobile drawer, loading/error boundary and
hash routing. Use `useStudio()` for scene, catalog, run status and settings;
use `route.navigate(id, params)` for navigation. Share UI through `components/`
and common layout classes in `studio/studio.css`. New backend capabilities can
be added to `tools/studio_api.py` without growing the original Lab handler.
Features with new data still need their own API and screen implementation.

## Coordinates

Lab and Studio Replay display XY relative to the pan-axis projection on the
tabletop (`world [0, -0.15, 0.42]`); Z=0 is the tabletop. A fixed XYZ marker at
this base origin is separate from the selected object's translation handles.
Lab numeric fields, map labels and end-effector measurements use this frame.
Replay position tables/logs use metres in this frame; physical condition panels
explicitly retain the engine's world-Z thresholds. JSON import/export, stored
cases, MuJoCo poses and the Python API always remain in world metres, so no
existing scene or simulation is shifted by this display change.

## Verification

- Python integration tests cover isolated repeat runs, duplicate-click handling,
  failure recovery, five physical objects, target switching, and zero-motion guards.
- Python tests also cover immutable run archives, source validation, existing
  replay import and full analysis rendering.
- Vitest checks bowl constraints and the scene/settings import contracts.
- `npm run build` performs strict TypeScript checking before bundling.
- Review checklist: labeled controls, visible keyboard focus, async status text,
  a skip link, keyboard alternatives for drag, undo for removal, responsive columns,
  reduced-motion styling, cleanup of requests/timers/Three.js buffers.

References: [Mantine Vite setup](https://mantine.dev/guides/vite/),
[React Three Fiber installation](https://r3f.docs.pmnd.rs/getting-started/installation),
[MuJoCo Python API](https://mujoco.readthedocs.io/en/stable/python.html).
