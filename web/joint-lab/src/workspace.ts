import type { Region } from "./types";
const deg = Math.PI / 180;
export function reachable(x: number, y: number, r: Region) {
  const dx = x - r.pan[0],
    dy = y - r.pan[1],
    radius = Math.hypot(dx, dy),
    angle = Math.atan2(dx, dy) / deg;
  return (
    radius >= r.r_min - 1e-8 &&
    radius <= r.r_max + 1e-8 &&
    angle >= r.angle_min_deg - 1e-8 &&
    angle <= r.angle_max_deg + 1e-8
  );
}
export function constrainBowl(
  x: number,
  y: number,
  r: Region,
): [number, number] {
  const dx = x - r.pan[0],
    dy = y - r.pan[1];
  // Project to the nearest valid polar boundary; the same bounds gate Python runs.
  const angle =
    Math.max(
      r.angle_min_deg,
      Math.min(r.angle_max_deg, Math.atan2(dx, dy) / deg),
    ) * deg;
  const radius = Math.max(r.r_min, Math.min(r.r_max, Math.hypot(dx, dy)));
  return [
    r.pan[0] + Math.sin(angle) * radius,
    r.pan[1] + Math.cos(angle) * radius,
  ];
}
export function sector(r: Region) {
  const points: number[][] = [];
  for (let i = 0; i <= 100; i++) {
    const a =
      (r.angle_min_deg + ((r.angle_max_deg - r.angle_min_deg) * i) / 100) * deg;
    points.push([
      r.pan[0] + Math.sin(a) * r.r_max,
      r.pan[1] + Math.cos(a) * r.r_max,
    ]);
  }
  for (let i = 100; i >= 0; i--) {
    const a =
      (r.angle_min_deg + ((r.angle_max_deg - r.angle_min_deg) * i) / 100) * deg;
    points.push([
      r.pan[0] + Math.sin(a) * r.r_min,
      r.pan[1] + Math.cos(a) * r.r_min,
    ]);
  }
  return points;
}
