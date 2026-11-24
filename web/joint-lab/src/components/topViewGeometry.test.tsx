import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { TopView } from "./TopView";
import { boxOutline, dragPosition, fromTop, labTopView, topX, topY } from "./topViewGeometry";
import type { XYViewData } from "./topViewGeometry";
import type { Item, Scene } from "../types";

const box: Item = { id: "box", type: "box", color: "#e5484d", x: 0.1, y: 0.2, yaw: 0, size: 0.04 };
const view: XYViewData = { origin: [0, -0.15], table: [0.45, 0.4], objects: [box], bowl: { pos: [-0.1, 0.1, 0.42], radius: 0.045, geoms: [] } };
const width = (obj: Item) => {
  const points = boxOutline(obj);
  return Math.max(...points.map((p) => p[0])) - Math.min(...points.map((p) => p[0]));
};

describe("shared XY viewport", () => {
  it("round-trips world coordinates through the Replay viewport including negative Y", () => {
    for (const [x, y] of [[-0.4, -0.3], [0, -0.15], [0.35, 0.28]]) {
      const point = fromTop(topX(x), topY(y));
      expect(point[0]).toBeCloseTo(x, 10);
      expect(point[1]).toBeCloseTo(y, 10);
    }
  });
  it("preserves the grab offset and limits dragging to the table", () => {
    const moved = dragPosition([0.12, 0.23], [0.02, 0.03], [0.45, 0.4]);
    expect(moved[0]).toBeCloseTo(0.1, 10);
    expect(moved[1]).toBeCloseTo(0.2, 10);
    expect(dragPosition([2, -2], [0, 0], [0.45, 0.4])).toEqual([0.435, -0.385]);
  });
  it("projects both yaw and quaternion tilt instead of flattening a replay box", () => {
    expect(width(box)).toBeCloseTo(0.04);
    expect(width({ ...box, yaw: 45 })).toBeCloseTo(0.04 * Math.SQRT2);
    expect(width({ ...box, quat: [Math.cos(Math.PI / 8), 0, Math.sin(Math.PI / 8), 0] })).toBeCloseTo(0.04 * Math.SQRT2);
  });
  it("renders the same shapes and metre axes in viewing and editing modes", () => {
    const replay = renderToStaticMarkup(<TopView data={view} />);
    const lab = renderToStaticMarkup(<TopView data={view} onMove={() => {}} />);
    for (const html of [replay, lab]) {
      expect(html).toContain('viewBox="0 0 520 340"');
      expect(html).toContain("x (m) →");
      expect(html).toContain("tâm khớp 1");
      expect(html).toMatch(/data-object="box"[^>]*>[\s\S]*?<polygon/);
    }
    expect(replay).not.toContain('role="button"');
    expect(lab).toContain('role="button" tabindex="0"');
  });
  it("translates the Lab bowl's local geometry exactly once", () => {
    const scene = { objects: [box], target_id: box.id, pose: { wx: -0.2, wy: 0.1 }, bowl: { color: "#3366d9", radius: 0.045, height: 0.04 } } as Scene;
    const data = labTopView(scene, null, { mesh: {} }, { region: { pan: [0, -0.15], r_min: 0.08, r_max: 0.42, angle_min_deg: -110, angle_max_deg: 110 } });
    expect(data.bowl.geoms[0].pos).toEqual([-0.2, 0.1, 0.42]);
    expect(data.bowl.geoms[1].pos[0]).toBeCloseTo(-0.155);
  });
});
