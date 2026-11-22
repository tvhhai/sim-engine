import { useEffect, useMemo, useLayoutEffect, useCallback } from "react";
import { useThree, useFrame } from "@react-three/fiber";
import { Line } from "@react-three/drei";
import {
  Matrix4,
  Object3D,
  Quaternion,
  OrthographicCamera,
  PerspectiveCamera,
  TextureLoader,
  Vector3,
} from "three";
import type {
  SensorCamera,
  Vec3,
  ViewCamera,
  Region,
  PlaybackSample,
} from "../types";
import { useReplayTransform } from "./useReplayTransform";
import { composeMount } from "../studio/playback";
import { sector } from "../workspace";

function AxisLabel({
  label,
  position,
  color = "#e7edf4",
  wide = false,
}: {
  label: string;
  position: Vec3;
  color?: string;
  wide?: boolean;
}) {
  const invalidate = useThree((s) => s.invalidate);
  const texture = useMemo(() => {
    const width = wide ? 128 : 40;
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="40"><rect width="${width}" height="40" rx="6" fill="#172129" fill-opacity=".9"/><text x="${width / 2}" y="28" text-anchor="middle" fill="${color}" font-family="sans-serif" font-size="26">${label}</text></svg>`;
    return new TextureLoader().load(
      "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg),
      () => invalidate(),
    );
  }, [label, color, wide, invalidate]);
  useEffect(() => () => texture.dispose(), [texture]);
  return (
    <sprite
      position={position}
      scale={[wide ? 0.068 : 0.022, 0.021, 1]}
      renderOrder={1001}
    >
      <spriteMaterial
        map={texture}
        transparent
        depthTest={false}
        toneMapped={false}
      />
    </sprite>
  );
}
export function BaseAxes({ origin }: { origin: Vec3 }) {
  return (
    <group position={origin}>
      <axesHelper args={[0.09]} renderOrder={1000}>
        <lineBasicMaterial attach="material" vertexColors depthTest={false} />
      </axesHelper>
      <AxisLabel position={[0.11, 0, 0.005]} label="X" color="#ed8f87" />
      <AxisLabel position={[0, 0.11, 0.005]} label="Y" color="#75d7b4" />
      <AxisLabel position={[0, 0, 0.115]} label="Z" color="#8daff7" />
      <AxisLabel position={[0, -0.06, 0.005]} label="O · đế" wide />
    </group>
  );
}
export function ReachBoundary({ region }: { region: Region }) {
  const points = useMemo(() => {
    const p = sector(region).map(([x, y]) => [x, y, 0.421] as Vec3);
    return [...p, p[0]];
  }, [region]);
  return (
    <Line
      points={points}
      color="#62d5b0"
      opacity={0.5}
      transparent
      lineWidth={1}
    />
  );
}
// External controls and the wrist sensor use the same camera conventions as MuJoCo (+Z back, +Y up).
export function SceneRig({
  view,
  ee,
  onOrbit,
}: {
  view?: ViewCamera;
  ee: Vec3;
  onOrbit?: (view: ViewCamera) => void;
}) {
  const { camera, set, size, controls, invalidate } = useThree();
  const perspective = useMemo(
    () => new PerspectiveCamera(38, 1, 0.005, 20),
    [],
  );
  const ortho = useMemo(
    () => new OrthographicCamera(-1, 1, 1, -1, 0.005, 20),
    [],
  );
  const orbit = controls as unknown as {
    target: Vector3;
    update: () => void;
    addEventListener: (name: string, fn: () => void) => void;
    removeEventListener: (name: string, fn: () => void) => void;
  } | null;
  const signature = JSON.stringify(view);
  const follow = view?.follow ? ee.join("/") : "";
  useEffect(() => {
    if (!view) return;
    const active = !view.orthographic ? perspective : ortho;
    if (camera !== active) set({ camera: active });
    const aspect = size.width / Math.max(1, size.height);
    {
      const target = new Vector3(...(view.follow ? ee : view.t));
      const cp = Math.cos(view.pitch),
        sp = Math.sin(view.pitch);
      active.position.set(
        target.x + view.dist * cp * Math.cos(view.yaw),
        target.y + view.dist * cp * Math.sin(view.yaw),
        target.z + view.dist * sp,
      );
      active.up.set(0, 0, 1);
      active.lookAt(target);
      if (orbit) {
        orbit.target.copy(target);
        orbit.update();
      }
      const h = view.dist * Math.tan((38 * Math.PI) / 360);
      ortho.left = -h * aspect;
      ortho.right = h * aspect;
      ortho.top = h;
      ortho.bottom = -h;
    }
    perspective.aspect = aspect;
    active.updateProjectionMatrix();
    invalidate();
  }, [
    signature,
    follow,
    size.width,
    size.height,
    camera,
    orbit,
    set,
    invalidate,
    perspective,
    ortho,
  ]);
  useEffect(() => {
    if (!orbit || !view || !onOrbit) return;
    const done = () => {
      const delta = camera.position.clone().sub(orbit.target),
        dist = delta.length();
      onOrbit({
        ...view,
        t: orbit.target.toArray() as Vec3,
        dist,
        yaw: Math.atan2(delta.y, delta.x),
        pitch: Math.asin(delta.z / dist),
        follow: false,
      });
    };
    orbit.addEventListener("end", done);
    return () => orbit.removeEventListener("end", done);
  }, [camera, orbit, onOrbit, signature]);
  return null;
}

export function SensorRig({
  sensor,
  sample,
  animate,
}: {
  sensor: SensorCamera;
  sample?: PlaybackSample;
  animate: boolean;
}) {
  const { camera, set, size, invalidate } = useThree();
  const lens = useMemo(
    () => new PerspectiveCamera(sensor.fovy, 1, 0.005, 20),
    [],
  );
  useLayoutEffect(() => {
    if (camera !== lens) set({ camera: lens });
    lens.fov = sensor.fovy;
    lens.aspect = size.width / Math.max(1, size.height);
    lens.updateProjectionMatrix();
    invalidate();
  }, [camera, lens, sensor.fovy, set, size.width, size.height, invalidate]);
  const mount = useMemo(() => new Object3D(), []);
  const local = useMemo(() => {
    const pose = sensor.mount?.local;
    if (!pose) return null;
    const r = pose.R;
    return {
      p: new Vector3(...pose.p),
      q: new Quaternion()
        .setFromRotationMatrix(
          new Matrix4().set(
            r[0],
            r[1],
            r[2],
            0,
            r[3],
            r[4],
            r[5],
            0,
            r[6],
            r[7],
            r[8],
            0,
            0,
            0,
            0,
            1,
          ),
        )
        .normalize(),
    };
  }, [JSON.stringify(sensor.mount?.local)]);
  useReplayTransform(mount, sensor.mount?.bodyPose || sensor, sample, animate);
  const apply = useCallback(() => {
    if (local)
      composeMount(
        mount.position,
        mount.quaternion,
        local.p,
        local.q,
        lens.position,
        lens.quaternion,
      );
    else {
      lens.position.copy(mount.position);
      lens.quaternion.copy(mount.quaternion);
    }
    lens.updateMatrixWorld();
  }, [local, mount, lens]);
  useLayoutEffect(() => {
    apply();
    invalidate();
  }, [apply, sensor, sample, invalidate]);
  useFrame(() => apply());
  return null;
}
