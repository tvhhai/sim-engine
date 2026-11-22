import { describe, expect, it } from "vitest";
import { constrainBowl, reachable } from "./workspace";
const region = {
  pan: [0, -0.15] as [number, number],
  r_min: 0.12,
  r_max: 0.42,
  angle_min_deg: -110,
  angle_max_deg: 110,
};
describe("bowl workspace", () => {
  it("keeps every dragged coordinate inside the shared region", () => {
    for (let x = -0.45; x <= 0.45; x += 0.025)
      for (let y = -0.4; y <= 0.4; y += 0.025) {
        const p = constrainBowl(x, y, region);
        expect(reachable(...p, region)).toBe(true);
      }
  });
  it("does not move an already-valid bowl", () => {
    const p = constrainBowl(-0.15, 0, region);
    expect(p[0]).toBeCloseTo(-0.15);
    expect(p[1]).toBeCloseTo(0);
  });
  it("classifies unreachable object test positions without clamping them", () => {
    expect(reachable(-0.44, -0.38, region)).toBe(false);
  });
});
