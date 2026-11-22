import { describe, it, expect } from "vitest";
import { parseSettings, DEFAULT_SETTINGS } from "./settings";
import { parseScene, request } from "./scene";
import type { Meta } from "../types";
const meta = {
  limits: [
    [-100, 100],
    [-100, 100],
    [-100, 100],
    [-100, 100],
    [-100, 100],
  ],
  gclose: -0.1,
  gopen: 1.2,
} as Meta;
const scene = {
  pose: {
    pan: 150,
    lift: 0,
    elbow: 0,
    wflex: 0,
    wroll: 0,
    grip: 2,
    wx: -0.15,
    wy: 0,
  },
  objects: [
    { id: "a", type: "sphere", color: "#abcdef", x: 0.15, y: 0.03, yaw: 10 },
  ],
  target_id: "a",
};
describe("Studio import contracts", () => {
  it("validates imported settings and fills missing fields", () => {
    const settings = parseSettings({
      theme: "light",
      defaultSpeed: "2",
      panels: { map: false },
      unknown: "ignored",
    });
    expect(settings.theme).toBe("light");
    expect(settings.panels.map).toBe(false);
    expect(settings.panels.objects).toBe(true);
    expect(settings.autoOpenReplay).toBe(DEFAULT_SETTINGS.autoOpenReplay);
  });
  it("rejects unsupported setting versions", () => {
    expect(() => parseSettings({ schema_version: 99 })).toThrow();
  });
  it("imports archive cases and clamps joints to the real widget limits", () => {
    const next = parseScene(
      { pose: { ...scene.pose, objects: scene.objects, target_id: "a" } },
      meta,
    );
    expect(next.pose.pan).toBe(100);
    expect(next.pose.grip).toBe(1.2);
    expect(request(next).bx).toBe(0.15);
  });
  it("rejects unknown targets, invalid colors and more than five objects", () => {
    expect(() => parseScene({ ...scene, target_id: "bad" }, meta)).toThrow();
    expect(() =>
      parseScene(
        { ...scene, objects: [{ ...scene.objects[0], color: "red" }] },
        meta,
      ),
    ).toThrow();
    expect(() =>
      parseScene({ ...scene, objects: Array(6).fill(scene.objects[0]) }, meta),
    ).toThrow();
  });
  it("keeps outside-workspace object positions for stop-condition tests", () => {
    const next = parseScene(
      { ...scene, objects: [{ ...scene.objects[0], x: -0.43, y: -0.38 }] },
      meta,
    );
    expect(next.objects[0].x).toBe(-0.43);
  });
});
