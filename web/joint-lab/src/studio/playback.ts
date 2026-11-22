import { Matrix4, Quaternion, Vector3 } from "three";
import type { BodyPose, PlaybackSample } from "../types";
export const playbackNow = () => performance.timeOrigin + performance.now();
export const sampleProgress = (sample: PlaybackSample, now: number) =>
  Math.max(
    0,
    Math.min(1, (now - sample.time) / Math.max(1, sample.durationMs)),
  );
export function composeMount(
  parentPosition: Vector3,
  parentRotation: Quaternion,
  localPosition: Vector3,
  localRotation: Quaternion,
  position: Vector3,
  rotation: Quaternion,
) {
  position
    .copy(localPosition)
    .applyQuaternion(parentRotation)
    .add(parentPosition);
  rotation.copy(parentRotation).multiply(localRotation);
}
/** Camera and geometry interpolate recorded endpoints on the same absolute clock. */
export class PoseTween {
  from = new Vector3();
  fromQ = new Quaternion();
  target = new Vector3();
  targetQ = new Quaternion();
  sample?: PlaybackSample;
  active = false;
  initialized = false;
  private matrix = new Matrix4();
  set(pose: BodyPose, sample: PlaybackSample | undefined, animate: boolean) {
    const previous = this.sample;
    const next =
      !!sample &&
      (!previous ||
        sample.stream !== previous.stream ||
        sample.step !== previous.step);
    if (next) {
      this.from.copy(this.target);
      this.fromQ.copy(this.targetQ);
    }
    const r = pose.R;
    this.matrix.set(
      r[0],
      r[1],
      r[2],
      0,
      r[3],
      r[4],
      r[5],
      0,
      r[6],
      r[7],
      r[8],
      0,
      0,
      0,
      0,
      1,
    );
    this.target.set(...pose.p);
    this.targetQ.setFromRotationMatrix(this.matrix).normalize();
    if (next)
      this.active = !!(
        animate &&
        sample?.continuous &&
        this.initialized &&
        previous?.stream === sample.stream &&
        sample.step === previous.step + 1
      );
    if (!animate || !sample) this.active = false;
    this.sample = sample;
    this.initialized = true;
  }
  apply(now: number, position: Vector3, quaternion: Quaternion) {
    const t = this.active && this.sample ? sampleProgress(this.sample, now) : 1;
    position.lerpVectors(this.from, this.target, t);
    quaternion.slerpQuaternions(this.fromQ, this.targetQ, t);
    return t < 1;
  }
}
