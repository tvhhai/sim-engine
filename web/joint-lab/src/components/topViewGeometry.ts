import type { BowlGeom, Item, Model, Scene, Snapshot, Vec3 } from "../types";
import type { PickZones } from "../zones";
import { bowlGeoms } from "../studio/scene";

export type Point2 = [number, number];
export type XYPath = { points: Vec3[]; color: string; width: number; opacity?: number; dash?: string; dots?: boolean };
export type XYGhost = { joints: Vec3[]; color: string; opacity: number; label: string };
export type XYViewData = {
  origin: Point2;
  table: Point2;
  objects: Item[];
  targetId?: string;
  bowl: { pos: Vec3; color?: string; radius: number; geoms: BowlGeom[] };
  ee?: Vec3;
  target?: Vec3 | null;
  zones?: PickZones | null;
  arm?: {
    joints: Vec3[];
    colors: string[];
    hulls: { points: Point2[]; color: string }[];
  };
  initialObject?: Item;
  paths?: XYPath[];
  ghosts?: XYGhost[];
};

// Match Replay's orthographic viewport, scale, and metre labels.
export const XY_WIDTH = 520, XY_HEIGHT = 340;
export const XY_BOUNDS = { x0: -0.44, x1: 0.44, y0: -0.34, y1: 0.31 };
export const XY_SCALE = Math.min(500 / 0.88, 320 / 0.65);
export const topX = (x: number) => XY_WIDTH / 2 + x * XY_SCALE;
export const topY = (y: number) => XY_HEIGHT / 2 - (y + 0.015) * XY_SCALE;
export const fromTop = (x: number, y: number): Point2 => [(x - XY_WIDTH / 2) / XY_SCALE, (XY_HEIGHT / 2 - y) / XY_SCALE - 0.015];
export const xyPoints = (points: number[][]) => points.map(([x, y]) => `${topX(x)},${topY(y)}`).join(" ");

export function hull2(points: Point2[]): Point2[] {
  const sorted = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o: Point2, a: Point2, b: Point2) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const half = (pts: Point2[]) => {
    const out: Point2[] = [];
    for (const p of pts) {
      while (out.length > 1 && cross(out[out.length - 2], out[out.length - 1], p) <= 0) out.pop();
      out.push(p);
    }
    return out;
  };
  if (sorted.length < 3) return sorted;
  return [...half(sorted).slice(0, -1), ...half(sorted.reverse()).slice(0, -1)];
}

// Project all eight corners: Replay boxes can tilt or roll, not just turn in XY.
export function boxOutline(obj: Item): Point2[] {
  const half = (obj.size ?? 0.03) / 2;
  const angle = obj.yaw * Math.PI / 360;
  const [w, x, y, z] = obj.quat ?? [Math.cos(angle), 0, 0, Math.sin(angle)];
  const points: Point2[] = [];
  for (const a of [-half, half]) for (const b of [-half, half]) for (const c of [-half, half]) {
    const tx = 2 * (y * c - z * b), ty = 2 * (z * a - x * c), tz = 2 * (x * b - y * a);
    points.push([obj.x + a + w * tx + y * tz - z * ty, obj.y + b + w * ty + z * tx - x * tz]);
  }
  return hull2(points);
}

export function wallOutline(g: BowlGeom): Point2[] {
  const [a, b] = g.size, c = Math.cos(g.yaw), s = Math.sin(g.yaw);
  return [[-a, -b], [a, -b], [a, b], [-a, b]].map(([x, y]) => [g.pos[0] + x * c - y * s, g.pos[1] + x * s + y * c]);
}

export function dragPosition(point: Point2, offset: Point2, table: Point2): Point2 {
  return point.map((v, k) => Math.max(-table[k] + 0.015, Math.min(table[k] - 0.015, v - offset[k]))) as Point2;
}

const ARM_BODIES = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper"];
export function labTopView(scene: Scene, snapshot: Snapshot | null, model: Model, zones: PickZones): XYViewData {
  const pos: Vec3 = [scene.pose.wx, scene.pose.wy, 0.42];
  const hulls = Object.entries(model.mesh).flatMap(([name, mesh]) => {
    const pose = snapshot?.pose[name];
    if (!pose) return [];
    const points: Point2[] = [];
    for (let k = 0; k < mesh.v.length; k += 6) {
      const [x, y, z] = mesh.v.slice(k, k + 3), R = pose.R;
      points.push([R[0] * x + R[1] * y + R[2] * z + pose.p[0], R[3] * x + R[4] * y + R[5] * z + pose.p[1]]);
    }
    const contact = snapshot?.contacts.some((c) => c.a === name || c.b === name);
    return [{ points: hull2(points), color: contact ? "#e5484d" : "#e8b923" }];
  });
  const joints = ARM_BODIES.flatMap((name) => snapshot?.pose[name] ? [snapshot.pose[name].p] : []);
  if (snapshot) joints.push(snapshot.ee);
  return {
    origin: zones.region.pan,
    table: [0.45, 0.4],
    zones,
    objects: scene.objects,
    targetId: scene.target_id,
    bowl: { pos, color: scene.bowl.color, radius: scene.bowl.radius, geoms: bowlGeoms(scene.bowl).map((g) => ({ ...g, pos: g.pos.map((v, k) => v + pos[k]) as Vec3 })) },
    ee: snapshot?.ee,
    arm: { joints, hulls, colors: joints.map(() => "var(--arm)") },
  };
}
