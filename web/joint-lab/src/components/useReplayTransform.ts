import { useLayoutEffect, useMemo } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import type { Object3D } from "three";
import type { BodyPose, PlaybackSample } from "../types";
import { PoseTween, playbackNow } from "../studio/playback";
export function useReplayTransform(
  object: Object3D | null,
  pose: BodyPose | undefined,
  sample: PlaybackSample | undefined,
  animate: boolean,
) {
  const tween = useMemo(() => new PoseTween(), []),
    invalidate = useThree((s) => s.invalidate);
  const signature = pose ? `${pose.p.join("/")}/${pose.R.join("/")}` : "";
  const clock = sample
    ? `${sample.stream}/${sample.step}/${sample.time}/${sample.durationMs}/${sample.continuous}`
    : "";
  useLayoutEffect(() => {
    if (!object || !pose) return;
    tween.set(pose, sample, animate);
    tween.apply(playbackNow(), object.position, object.quaternion);
    invalidate();
  }, [object, signature, clock, animate, tween, invalidate]);
  useFrame(() => {
    if (
      object &&
      pose &&
      tween.apply(playbackNow(), object.position, object.quaternion)
    )
      invalidate();
  });
}
