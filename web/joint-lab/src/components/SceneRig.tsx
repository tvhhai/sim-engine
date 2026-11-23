import { useEffect, useMemo, useLayoutEffect, useCallback } from "react";
import { useThree, useFrame } from "@react-three/fiber";
import { Line } from "@react-three/drei";
import {
  BufferGeometry,
  Float32BufferAttribute,
  Matrix4,
  Object3D,
  Path,
  Shape,
  ShapeGeometry,
  Quaternion,
  OrthographicCamera,
  PerspectiveCamera,
  TextureLoader,
  Vector2,
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
import { ZONE_COLORS, tableBounds, visibleScan, zoneShapes } from "../zones";
import type { Bounds } from "../zones";
import type { PickZones } from "../zones";

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
// Lưới mặt bàn: chỉ vẽ trên phần bàn còn hiển thị (bounds), các vạch cách nhau 5 cm tính từ gốc (trục khớp 1)
export function TableGrid({ origin, bounds }: { origin: Vec3; bounds: Bounds }) {
  const geometry = useMemo(() => {
    const pts: number[] = [];
    const z = 0.4201;
    const first = (o: number, lo: number) => o + Math.ceil((lo - o) / 0.05 - 1e-9) * 0.05;
    for (let x = first(origin[0], bounds.x0); x <= bounds.x1 + 1e-9; x += 0.05) pts.push(x, bounds.y0, z, x, bounds.y1, z);
    for (let y = first(origin[1], bounds.y0); y <= bounds.y1 + 1e-9; y += 0.05) pts.push(bounds.x0, y, z, bounds.x1, y, z);
    const g = new BufferGeometry();
    g.setAttribute("position", new Float32BufferAttribute(pts, 3));
    return g;
  }, [origin[0], origin[1], bounds.x0, bounds.x1, bounds.y0, bounds.y1]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color="#5b6a73" />
    </lineSegments>
  );
}
const ZONE_Z = 0.4203; // ngay trên mặt bàn (0.42), dưới lưới toạ độ
// Vùng gắp xanh / hỗn hợp / đỏ trên mặt bàn và các điểm quét (✕ hỏng; chấm chỉ ở vùng hỗn hợp), xem zones.ts.
export function PickZones3D({ zones, bounds }: { zones: PickZones; bounds: Bounds }) {
  const scanKey = zones.scan?.length ?? 0;
  const { meshes, cross, dots } = useMemo(() => {
    const { pieces, outsideHole } = zoneShapes(zones, bounds);
    const meshes = pieces.map(([zone, pts]) => {
      const shape = new Shape(pts.map(([x, y]) => new Vector2(x, y)));
      return { zone, geometry: new ShapeGeometry(shape) };
    });
    if (outsideHole) {
      const table = new Shape();
      table.moveTo(bounds.x0, bounds.y0);
      table.lineTo(bounds.x1, bounds.y0);
      table.lineTo(bounds.x1, bounds.y1);
      table.lineTo(bounds.x0, bounds.y1);
      table.closePath();
      const hole = new Path();
      outsideHole.forEach(([x, y], i) => (i ? hole.lineTo(x, y) : hole.moveTo(x, y)));
      table.holes.push(hole);
      meshes.push({ zone: "red", geometry: new ShapeGeometry(table, 4) });
    }
    const pts = visibleScan(zones), d = 0.0035, z = ZONE_Z + 0.0004;
    const cross = new BufferGeometry();
    cross.setAttribute(
      "position",
      new Float32BufferAttribute(
        pts.filter((p) => !p.ok).flatMap(({ x, y }) => [
          x - d, y - d, z, x + d, y + d, z, x - d, y + d, z, x + d, y - d, z,
        ]),
        3,
      ),
    );
    const dots = new BufferGeometry();
    dots.setAttribute(
      "position",
      new Float32BufferAttribute(pts.filter((p) => p.ok).flatMap(({ x, y }) => [x, y, z]), 3),
    );
    return { meshes, cross, dots };
    // scan được gửi lại mỗi khung hình từ replay: so theo nội dung thay vì tham chiếu mảng
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(zones.region), JSON.stringify(zones.stop), scanKey, bounds.y0]);
  useEffect(
    () => () => {
      meshes.forEach((m) => m.geometry.dispose());
      cross.dispose();
      dots.dispose();
    },
    [meshes, cross, dots],
  );
  return (
    <group>
      {meshes.map((m, i) => (
        <mesh key={i} geometry={m.geometry} position={[0, 0, ZONE_Z]} renderOrder={1}>
          <meshBasicMaterial
            color={ZONE_COLORS[m.zone]}
            transparent
            opacity={m.zone === "green" ? 0.16 : 0.2}
            depthWrite={false}
          />
        </mesh>
      ))}
      <lineSegments geometry={cross} renderOrder={2}>
        <lineBasicMaterial color={ZONE_COLORS.red} transparent opacity={0.95} depthWrite={false} />
      </lineSegments>
      <points geometry={dots} renderOrder={2}>
        <pointsMaterial color="#3fcf8e" size={0.004} sizeAttenuation transparent opacity={0.85} depthWrite={false} />
      </points>
    </group>
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
