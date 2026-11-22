import { describe, it, expect } from "vitest";
import { BASE_ORIGIN, toBase, toWorld } from "./coordinates";
describe("Base coordinate display", () => {
  it("anchors zero at the fixed pan projection on the table", () =>
    expect(toBase(BASE_ORIGIN)).toEqual([0, 0, 0]));
  it("shows the default bowl 15cm left and 15cm forward of the base", () =>
    expect(toBase([-0.15, 0, 0.42])).toEqual([-0.15, 0.15, 0]));
  it("round trips edited display coordinates without changing world poses", () => {
    const points: [[number, number, number], [number, number, number]] = [
      [0.123, 0.084, 0.435],
      [-0.43, -0.38, 0.6],
    ];
    points.forEach((point) =>
      toWorld(toBase(point)).forEach((v, i) =>
        expect(v).toBeCloseTo(point[i], 12),
      ),
    );
  });
});
