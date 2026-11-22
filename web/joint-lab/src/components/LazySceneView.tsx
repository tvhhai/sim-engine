import { lazy, Suspense } from "react";
import type { ComponentProps } from "react";
import { Loader } from "@mantine/core";

const WebGLScene = lazy(() =>
  import("./SceneView").then((module) => ({ default: module.SceneView })),
);
type Props = ComponentProps<typeof WebGLScene>;
export function SceneView(props: Props) {
  return (
    <Suspense
      fallback={
        <div className="scene-loading">
          <Loader color="teal" aria-label="Đang nạp cảnh 3D" />
        </div>
      }
    >
      <WebGLScene {...props} />
    </Suspense>
  );
}
