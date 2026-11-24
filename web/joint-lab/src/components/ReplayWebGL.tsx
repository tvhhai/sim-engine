import { useCallback, useMemo } from "react";
import { createPortal } from "react-dom";
import { SceneView } from "./LazySceneView";
import { TopView } from "./TopView";
import type { Model, ReplayViewState, ViewCamera } from "../types";

export function ReplayWebGL({
  document,
  scene,
  model,
}: {
  document: Document;
  scene: ReplayViewState;
  model: Model;
}) {
  const slots = useMemo(
    () => ({
      main: document.getElementById("studio-three-main"),
      sensor: document.getElementById("studio-three-camera"),
      top: document.getElementById("top"),
    }),
    [document],
  );
  const orbit = useCallback(
    (view: ViewCamera) =>
      document.defaultView?.postMessage(
        { type: "studio-replay-command", orbit: view },
        location.origin,
      ),
    [document],
  );
  const common = {
    model,
    poses: scene.poses,
    objects: scene.objects,
    bowl: scene.bowl,
    bowlGeoms: scene.bowlGeoms,
    bowlColor: scene.bowlColor,
    bowlOrigin: scene.bowl,
    ee: scene.ee,
    origin: scene.origin,
    sample: scene.sample,
    animate: scene.playing,
  };
  return (
    <>
      {slots.top && scene.xy && createPortal(<TopView data={scene.xy} showRegion={scene.showRegion} />, slots.top)}
      {slots.main &&
        createPortal(
          <SceneView
            {...common}
            view={scene.camera}
            onOrbit={orbit}
            lines={scene.lines}
            target={scene.target}
            contacts={scene.contacts}
            region={scene.region}
            zones={scene.zones}
            showRegion={scene.showRegion}
          />,
          slots.main,
        )}
      {slots.sensor &&
        scene.sensor &&
        createPortal(
          <SceneView {...common} sensor={scene.sensor} />,
          slots.sensor,
        )}
    </>
  );
}
