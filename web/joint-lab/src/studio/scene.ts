import type { Item, Meta, Pose, Scene } from "../types";
import { JOINTS } from "../types";
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
  return { pose, objects: items, target_id };
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
