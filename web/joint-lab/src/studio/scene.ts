import type { Bowl, BowlGeom, Item, Meta, Pose, Scene } from "../types";
import { JOINTS } from "../types";
export const DEFAULT_SIZE = 0.03; // cỡ của vật lưu từ trước chưa có trường size (giữ 3 cm để mở lại đúng cảnh cũ)
export const NEW_OBJECT_SIZE = 0.04; // cỡ mặc định cho vật mới thêm trong Lab
// 5 cm là cỡ thử nghiệm; hộp 5 cm đạt 24/45 trong phép quét trước.
export const SIZE_RANGE = [0.02, 0.05] as const;
// Bát phải chứa được vật: đường kính lòng bát >= cỡ vật + BOWL_FIT_MARGIN (chừa chỗ cho hai hàm kẹp khi thả)
export const BOWL_FIT_MARGIN = 0.02;
export const DEFAULT_BOWL: Bowl = { color: "#3366d9", radius: 0.045, height: 0.04 };
export const BOWL_WALL = 0.003;
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
export function parseBowl(input: unknown): Bowl {
  const b = (input && typeof input === "object" ? input : {}) as Partial<Bowl>;
  return {
    color: /^#[0-9a-f]{6}$/i.test(String(b.color)) ? String(b.color) : DEFAULT_BOWL.color,
    radius: clamp(Number(b.radius) || DEFAULT_BOWL.radius, 0.03, 0.09),
    height: clamp(Number(b.height) || DEFAULT_BOWL.height, 0.02, 0.08),
  };
}
// Hình học bát tương đối so với tâm bát (cùng công thức với studio/scene.py: đáy trụ + 24 tấm tường hẹp), để Lab vẽ ngay khi chỉnh cỡ bát
export function bowlGeoms(bowl: Bowl): BowlGeom[] {
  const n = 24, r = bowl.radius, h = bowl.height, half = (1.34 * Math.PI * r) / n;
  const walls = Array.from({ length: n }, (_, k) => {
    const a = (2 * Math.PI * k) / n;
    return { type: "box", pos: [r * Math.cos(a), r * Math.sin(a), h / 2] as [number, number, number], size: [BOWL_WALL, half, h / 2] as [number, number, number], yaw: a };
  });
  return [{ type: "cylinder", pos: [0, 0, 0], size: [r, 0.003, 0], yaw: 0 }, ...walls];
}
// Cỡ vật (m) kẹp về SIZE_RANGE; thiếu hoặc không hợp lệ thì lấy DEFAULT_SIZE
export function clampSize(size: unknown): number {
  const v = Number(size);
  return Number.isFinite(v) && v > 0 ? clamp(v, SIZE_RANGE[0], SIZE_RANGE[1]) : DEFAULT_SIZE;
}
// Cỡ vật lớn nhất mà bát hiện tại chứa được (m), trong SIZE_RANGE
export function maxObjectSize(bowl: Bowl): number {
  return clamp(2 * (bowl.radius - BOWL_WALL) - BOWL_FIT_MARGIN, SIZE_RANGE[0], SIZE_RANGE[1]);
}
// Bán kính bát nhỏ nhất chứa được vật lớn nhất trong cảnh (m)
export function minBowlRadius(objects: Item[]): number {
  const biggest = Math.max(DEFAULT_SIZE, ...objects.map((o) => o.size ?? DEFAULT_SIZE));
  const start = Math.max(6, Math.ceil(((biggest + BOWL_FIT_MARGIN) / 2 + BOWL_WALL) * 200 - 1e-9));
  for (let step = start; step <= 18; step++) {
    const radius = step / 200;
    if (fitsBowl(objects, { ...DEFAULT_BOWL, radius })) return radius;
  }
  return 0.09;
}
// Cùng phép kiểm tra xếp một lớp với studio/scene.py.
export function fitsBowl(objects: Pick<Item, "size" | "type">[], bowl: Bowl): boolean {
  const inner = bowl.radius - BOWL_WALL;
  if (objects.some((o) => (o.size ?? DEFAULT_SIZE) + BOWL_FIT_MARGIN > 2 * inner + 1e-9)) return false;
  if (objects.length < 2) return true;
  const radii = objects.map((o) => (o.size ?? DEFAULT_SIZE) * (o.type === "box" ? Math.SQRT2 : 1) / 2).sort((a, b) => b - a);
  const placed: { x: number; y: number; radius: number }[] = [];
  for (const radius of radii) {
    const edge = inner - radius;
    const candidates = [1, .75, .5, .25, 0].flatMap((fraction) => Array.from({ length: 24 }, (_, i) => {
      const angle = i * 2 * Math.PI / 24;
      return { x: edge * fraction * Math.cos(angle), y: edge * fraction * Math.sin(angle), radius };
    }));
    const point = candidates.find(({ x, y }) => placed.every((p) => Math.hypot(x - p.x, y - p.y) + 1e-9 >= radius + p.radius));
    if (!point) return false;
    placed.push(point);
  }
  return true;
}
export function parseScene(input: unknown, meta: Meta): Scene {
  if (!input || typeof input !== "object")
    throw new Error("Tệp cảnh không hợp lệ");
  const value = input as Record<string, unknown>;
  const raw = (
    value.pose && typeof value.pose === "object" ? value.pose : value
  ) as Record<string, unknown>;
  const objects = (value.objects || raw.objects) as Item[];
  if (!Array.isArray(objects) || !objects.length || objects.length > 5)
    throw new Error("Cảnh phải có 1 đến 5 vật thể");
  const ids = new Set<string>();
  const items = objects.map((item) => {
    if (
      !item ||
      !["box", "sphere"].includes(item.type) ||
      !/^#[0-9a-f]{6}$/i.test(item.color) ||
      !/^[a-z0-9_-]{1,40}$/i.test(item.id) ||
      ids.has(item.id)
    )
      throw new Error("ID, loại hoặc màu vật thể không hợp lệ");
    ids.add(item.id);
    if (
      ![item.x, item.y, item.yaw].every(Number.isFinite) ||
      Math.abs(item.x) > 0.45 ||
      Math.abs(item.y) > 0.4
    )
      throw new Error("Vật thể phải nằm trên mặt bàn");
    return {
      id: item.id,
      type: item.type,
      color: item.color,
      x: item.x,
      y: item.y,
      yaw: item.yaw,
      size: clampSize(item.size),
    };
  });
  const pose = {} as Pose;
  for (const key of [
    ...JOINTS.map(([key]) => key),
    "grip",
    "wx",
    "wy",
  ] as const) {
    const number = Number(raw[key]);
    if (raw[key] === undefined || raw[key] === null || !Number.isFinite(number))
      throw new Error("Góc khớp hoặc tọa độ không hợp lệ");
    pose[key] = number;
  }
  if (Math.abs(pose.wx) > 0.45 || Math.abs(pose.wy) > 0.45)
    throw new Error("Tọa độ bát không hợp lệ");
  JOINTS.forEach(([key], i) => {
    pose[key] = Math.max(
      meta.limits[i][0],
      Math.min(meta.limits[i][1], pose[key]),
    );
  });
  pose.grip = Math.max(meta.gclose, Math.min(meta.gopen, pose.grip));
  const target_id = String(value.target_id || raw.target_id || items[0].id);
  const target = items.find((item) => item.id === target_id);
  if (!target) throw new Error("Vật cần gắp không tồn tại");
  Object.assign(pose, { bx: target.x, by: target.y, byaw: target.yaw });
  const mode = (value.mode ?? raw.mode) === "all" ? "all" : undefined;
  return { pose, objects: items, target_id, bowl: parseBowl(value.bowl ?? raw.bowl), ...(mode && { mode }) };
}
export function request(scene: Scene) {
  const target = scene.objects.find((item) => item.id === scene.target_id)!;
  return {
    ...scene.pose,
    bx: target.x,
    by: target.y,
    byaw: target.yaw,
    objects: scene.objects,
    target_id: scene.target_id,
    bowl: scene.bowl,
  };
}
export function downloadJson(name: string, data: unknown) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
