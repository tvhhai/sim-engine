import { describe, expect, it } from "vitest";
import { clipToRect, pickZone, visibleScan, zoneShapes } from "./zones";
import type { PickZones } from "./zones";

const zones: PickZones = {
  region: { pan: [0, -0.15], r_min: 0.12, r_max: 0.42, angle_min_deg: -110, angle_max_deg: 110 },
  stop: { pan: [0, -0.15], angle_deg: 115, radius_m: 0.44, inner_radius_m: 0.04 },
  scan: [
    [0, 0.1, 1], // green, ok -> hidden
    [0, 0.12, 0], // green, fail -> ✕
    [0, 0.28, 1], // r 0.43: mixed, ok -> shown
    [0, 0.28, 0], // mixed, fail -> shown
    [0, 0.35, 0], // r 0.5: red -> hidden
  ],
};

describe("pick zones", () => {
  it("classifies points like the canvas replay", () => {
    expect(pickZone(0, 0.1, zones)).toBe("green");
    expect(pickZone(0, 0.28, zones)).toBe("mixed");
    expect(pickZone(0, 0.35, zones)).toBe("red");
    expect(pickZone(0, -0.13, zones)).toBe("red"); // r 2 cm < inner radius
    expect(pickZone(-0.05, -0.3, zones)).toBe("red"); // behind the base, past ±115°
  });

  it("shows only failures in green, both in mixed, nothing in red", () => {
    expect(visibleScan(zones).map((p) => [p.y, p.ok, p.zone])).toEqual([
      [0.12, false, "green"],
      [0.28, true, "mixed"],
      [0.28, false, "mixed"],
    ]);
  });

  it("clips polygons to the table", () => {
    const out = clipToRect([[-1, -1], [1, -1], [1, 1], [-1, 1]]);
    expect(out.map(([x, y]) => [+Math.abs(x).toFixed(9), +Math.abs(y).toFixed(9)])).toEqual([
      [0.45, 0.4], [0.45, 0.4], [0.45, 0.4], [0.45, 0.4],
    ]);
  });

  it("builds green, mixed and red pieces inside the table", () => {
    const { pieces, outsideHole } = zoneShapes(zones);
    expect(new Set(pieces.map(([z]) => z))).toEqual(new Set(["green", "mixed", "red"]));
    for (const [, pts] of pieces)
      for (const [x, y] of pts) {
        expect(Math.abs(x)).toBeLessThanOrEqual(0.45 + 1e-9);
        expect(Math.abs(y)).toBeLessThanOrEqual(0.4 + 1e-9);
      }
    expect(outsideHole?.length).toBeGreaterThan(100);
  });
});
