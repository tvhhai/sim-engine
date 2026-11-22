import type { Region } from "./types";

// Ba vùng gắp trên mặt bàn, cùng quy ước với replay canvas (static/replay/replay.js: pickZone, pickZonePolygons):
//   green: trong hình quạt gắp được đã đo (config.PICK_REGION)
//   red:   ngoài giới hạn dừng của policy (khớp 1 quá ±(110+5)°, quá xa, quá sát đế): về home ngay
//   mixed: phần còn lại, có chỗ gắp được có chỗ không
// Điểm quét (tools/scan_pick_region.py): vùng đỏ không vẽ; vùng xanh chỉ vẽ ✕ (điểm hỏng), vì chấm xanh là điều vùng xanh
// đã nói; vùng hỗn hợp vẽ cả chấm và ✕.

export type StopRegion = {
  pan: [number, number];
  angle_deg: number;
  radius_m: number;
  inner_radius_m?: number;
};
export type ScanPoint = [number, number, number]; // x, y, 1 nếu gắp được
export type PickZones = {
  region: Region;
  stop?: StopRegion | null;
  scan?: ScanPoint[];
};
export type Zone = "green" | "mixed" | "red";
export const ZONE_COLORS: Record<Zone, string> = {
  green: "#16834f",
  red: "#d14049",
  mixed: "#8b55bd",
};
export const TABLE_HALF: [number, number] = [0.45, 0.4];

const deg = Math.PI / 180;

export function pickZone(x: number, y: number, z: PickZones): Zone {
  const { region, stop } = z;
  const dx = x - region.pan[0],
    dy = y - region.pan[1];
  const r = Math.hypot(dx, dy),
    angle = Math.atan2(dx, dy) / deg;
  if (
    stop &&
    (r > stop.radius_m ||
      r < (stop.inner_radius_m || 0) ||
      Math.abs(angle) > stop.angle_deg)
  )
    return "red";
  if (
    r < region.r_min - 1e-9 ||
    r > region.r_max + 1e-9 ||
    angle < region.angle_min_deg - 1e-9 ||
    angle > region.angle_max_deg + 1e-9
  )
    return "mixed";
  return "green";
}

// Điểm quét cần vẽ theo quy tắc trên
export function visibleScan(z: PickZones) {
  return (z.scan || []).flatMap(([x, y, ok]) => {
    const zone = pickZone(x, y, z);
    if (zone === "red" || (zone === "green" && ok)) return [];
    return [{ x, y, ok: !!ok, zone }];
  });
}

function sectorPts(
  pan: [number, number],
  inner: number,
  outer: number,
  start: number,
  end: number,
  n = 72,
) {
  const point = (r: number, q: number): [number, number] => {
    const a = (start + ((end - start) * q) / n) * deg;
    return [pan[0] + r * Math.sin(a), pan[1] + r * Math.cos(a)];
  };
  const pts = Array.from({ length: n + 1 }, (_, q) => point(outer, q));
  if (!inner) return [...pts, pan];
  return [...pts, ...Array.from({ length: n + 1 }, (_, q) => point(inner, n - q))];
}

// Sutherland–Hodgman: cắt đa giác theo hình chữ nhật mặt bàn (cửa sổ lồi nên đúng cả với đa giác lõm)
export function clipToRect(
  poly: [number, number][],
  [hx, hy]: [number, number] = TABLE_HALF,
) {
  const edges: [(p: [number, number]) => boolean, (a: [number, number], b: [number, number]) => [number, number]][] = [
    [(p) => p[0] >= -hx, (a, b) => cut(a, b, 0, -hx)],
    [(p) => p[0] <= hx, (a, b) => cut(a, b, 0, hx)],
    [(p) => p[1] >= -hy, (a, b) => cut(a, b, 1, -hy)],
    [(p) => p[1] <= hy, (a, b) => cut(a, b, 1, hy)],
  ];
  function cut(a: [number, number], b: [number, number], k: number, v: number): [number, number] {
    const t = (v - a[k]) / (b[k] - a[k]);
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  }
  let out = poly;
  for (const [inside, inter] of edges) {
    const input = out;
    out = [];
    input.forEach((cur, i) => {
      const prev = input[(i + input.length - 1) % input.length];
      if (inside(cur)) {
        if (!inside(prev)) out.push(inter(prev, cur));
        out.push(cur);
      } else if (inside(prev)) out.push(inter(prev, cur));
    });
    if (!out.length) break;
  }
  return out;
}

// Các mảnh vùng đã cắt theo mặt bàn. "Ngoài bán kính dừng" trả riêng (outsideHole) vì là mặt bàn trừ một hình tròn.
export function zoneShapes(z: PickZones) {
  const R = z.region,
    U = z.stop;
  if (!U) return { pieces: [["green", clipToRect(sectorPts(R.pan, R.r_min, R.r_max, R.angle_min_deg, R.angle_max_deg))]] as [Zone, [number, number][]][], outsideHole: null };
  const { pan, r_min: lo, r_max: hi, angle_min_deg: left, angle_max_deg: right } = R;
  const edge = U.angle_deg,
    limit = U.radius_m,
    inner = U.inner_radius_m || 0;
  const raw: [Zone, [number, number][]][] = [
    ["green", sectorPts(pan, lo, hi, left, right)],
    ["mixed", sectorPts(pan, inner, lo, -edge, edge)],
    ["mixed", sectorPts(pan, hi, limit, -edge, edge)],
    ["mixed", sectorPts(pan, lo, hi, -edge, left)],
    ["mixed", sectorPts(pan, lo, hi, right, edge)],
    ["red", sectorPts(pan, 0, limit, edge, 360 - edge)],
  ];
  if (inner) raw.push(["red", sectorPts(pan, 0, inner, -180, 180, 144).slice(0, -1)]);
  const pieces = raw
    .map(([zone, pts]) => [zone, clipToRect(pts)] as [Zone, [number, number][]])
    .filter(([, pts]) => pts.length >= 3);
  const circle = Array.from({ length: 145 }, (_, q) => {
    const a = (q / 144) * 360 * deg;
    return [pan[0] + limit * Math.sin(a), pan[1] + limit * Math.cos(a)] as [number, number];
  });
  return { pieces, outsideHole: circle };
}
