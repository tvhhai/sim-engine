import { describe, it, expect } from "vitest";
import { Matrix4, Quaternion, Vector3 } from "three";
import { PoseTween, composeMount, sampleProgress } from "./playback";
import type { BodyPose, PlaybackSample } from "../types";
const sample = (
  step: number,
  time = 1000,
  durationMs = 32,
  continuous = true,
): PlaybackSample => ({ stream: "lab/0", step, time, durationMs, continuous });
function pose(x: number, angle = 0): BodyPose {
  const m = new Matrix4().makeRotationZ(angle).elements;
  return {
    p: [x, 0, 0],
    R: [m[0], m[4], m[8], m[1], m[5], m[9], m[2], m[6], m[10]],
  };
}
function read(tween: PoseTween, now: number) {
  const p = new Vector3(),
    q = new Quaternion();
  tween.apply(now, p, q);
  return { p, q };
}
function moving() {
  const t = new PoseTween();
  t.set(pose(0), sample(0, 968), false);
  t.set(pose(1, Math.PI / 2), sample(1), true);
  return t;
}
describe("Synchronized replay interpolation", () => {
  it("uses speed-aware time rather than a fixed 32ms for every playback speed", () => {
    for (const speed of [0.25, 0.5, 1, 2, 4]) {
      const s = sample(1, 1000, 32 / speed);
      expect(sampleProgress(s, 1000 + s.durationMs / 2)).toBeCloseTo(0.5);
    }
  });
  it("two viewports show the same body pose at the same time, including late consumers", () => {
    const a = moving(),
      b = moving();
    read(a, 1007);
    read(b, 1011);
    const at = read(a, 1016),
      bt = read(b, 1016);
    expect(at.p.x).toBeCloseTo(0.5);
    expect(at.p.distanceTo(bt.p)).toBe(0);
    expect(at.q.angleTo(bt.q)).toBeCloseTo(0);
    expect(at.q.angleTo(new Quaternion())).toBeCloseTo(Math.PI / 4);
  });
  it("keeps the sensor rigidly attached to the interpolated wrist", () => {
    const wrist = moving(),
      local = new Vector3(0.03, 0.01, 0.02),
      localRotation = new Quaternion().setFromAxisAngle(
        new Vector3(0, 1, 0),
        0.2,
      );
    for (const now of [1000, 1008, 1016, 1024, 1032]) {
      const parent = read(wrist, now),
        p = new Vector3(),
        q = new Quaternion();
      composeMount(parent.p, parent.q, local, localRotation, p, q);
      const recovered = p
        .clone()
        .sub(parent.p)
        .applyQuaternion(parent.q.clone().invert());
      expect(recovered.distanceTo(local)).toBeCloseTo(0, 12);
      expect(
        parent.q.clone().invert().multiply(q).angleTo(localRotation),
      ).toBeCloseTo(0, 6);
    }
  });
  it("duplicate messages for one sample do not restart interpolation", () => {
    const t = moving();
    expect(read(t, 1008).p.x).toBeCloseTo(0.25);
    t.set(pose(1, Math.PI / 2), sample(1), true);
    expect(read(t, 1016).p.x).toBeCloseTo(0.5);
  });
  it("pausing and manual seeks snap exactly to the recorded state", () => {
    const t = moving();
    t.set(pose(1), sample(1), false);
    expect(read(t, 1008).p.x).toBe(1);
    t.set(pose(7), sample(7, 1010, 32, false), true);
    expect(read(t, 1011).p.x).toBe(7);
  });
  it("never invents interpolation across dropped steps or a different episode", () => {
    const t = moving();
    t.set(pose(9), sample(9), true);
    expect(read(t, 1001).p.x).toBe(9);
    t.set(pose(-4), { ...sample(10), stream: "baseline/10" }, true);
    expect(read(t, 1002).p.x).toBe(-4);
  });
  it("clamps outside the sample interval to avoid overshoot and extrapolation", () => {
    expect(sampleProgress(sample(1), 900)).toBe(0);
    expect(sampleProgress(sample(1), 1040)).toBe(1);
    expect(read(moving(), 1050).p.x).toBe(1);
  });
});
