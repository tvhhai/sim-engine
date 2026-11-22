import { useMemo, useState } from "react";
import type { ReactNode } from "react";
import type { Group } from "three";
import { Matrix4, Quaternion } from "three";
import type { BodyPose, Vec3, PlaybackSample } from "../types";
import { useReplayTransform } from "./useReplayTransform";
export function SmoothPose({
  pose,
  position,
  orientation,
  animate = false,
  sample,
  children,
}: {
  pose?: BodyPose;
  position?: Vec3;
  orientation?: [number, number, number, number];
  animate?: boolean;
  sample?: PlaybackSample;
  children: ReactNode;
}) {
  const [group, setGroup] = useState<Group | null>(null);
  const target = useMemo(() => {
    if (pose) return pose;
    const q = orientation
      ? new Quaternion(
          orientation[1],
          orientation[2],
          orientation[3],
          orientation[0],
        )
      : new Quaternion();
    const m = new Matrix4().makeRotationFromQuaternion(q).elements;
    return {
      p: position || ([0, 0, 0] as Vec3),
      R: [m[0], m[4], m[8], m[1], m[5], m[9], m[2], m[6], m[10]],
    };
  }, [pose, position, orientation]);
  useReplayTransform(group, target, sample, animate);
  return <group ref={setGroup}>{children}</group>;
}
