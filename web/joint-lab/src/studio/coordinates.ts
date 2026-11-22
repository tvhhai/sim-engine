import type { Vec3 } from "../types";
// UI origin: pan-axis projection onto the tabletop. Physics and JSON stay in world metres.
export const BASE_ORIGIN: Vec3 = [0, -0.15, 0.42];
export const toBase = (point: Vec3, origin: Vec3 = BASE_ORIGIN): Vec3 =>
  point.map((v, i) => v - origin[i]) as Vec3;
export const toWorld = (point: Vec3, origin: Vec3 = BASE_ORIGIN): Vec3 =>
  point.map((v, i) => v + origin[i]) as Vec3;
