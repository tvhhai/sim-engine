import { memo, useEffect, useMemo, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import {
  GizmoHelper,
  GizmoViewport,
  OrbitControls,
  TransformControls,
  Line,
} from "@react-three/drei";
import {
  BufferGeometry,
  Float32BufferAttribute,
  MeshStandardMaterial,
  Color,
  DoubleSide,
} from "three";
import type { Group } from "three";
import type {
  BodyPose,
  BowlGeom,
  Item,
  Model,
  Vec3,
  ReplayViewState,
  SensorCamera,
  ViewCamera,
  Region,
  PlaybackSample,
} from "../types";
import { BaseAxes, PickZones3D, ReachBoundary, SceneRig, SensorRig } from "./SceneRig";
import type { PickZones } from "../zones";
import { SmoothPose } from "./SmoothPose";
import { BASE_ORIGIN } from "../studio/coordinates";

const Robot = memo(function Robot({
  model,
  poses,
  contacts,
  animate,
  sample,
}: {
  model: Model;
  poses: Record<string, BodyPose>;
  contacts?: ReplayViewState["contacts"];
  animate?: boolean;
  sample?: PlaybackSample;
}) {
  const parts = useMemo(
    () =>
      Object.entries(model.mesh).map(([name, data]) => {
        const geometry = new BufferGeometry();
        geometry.setAttribute(
          "position",
          new Float32BufferAttribute(data.v, 3),
        );
        geometry.setIndex(data.f);
        geometry.computeVertexNormals();
        let offset = 0;
        const materials = data.colors.map(([count, rgb], i) => {
          geometry.addGroup(offset, count * 3, i);
          offset += count * 3;
          return new MeshStandardMaterial({
            color: new Color(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255),
            roughness: 0.65,
            metalness: 0.12,
            side: DoubleSide,
          });
        });
        return { name, geometry, materials };
      }),
    [model],
  );
  useEffect(
    () => () => {
      parts.forEach((p) => {
        p.geometry.dispose();
        p.materials.forEach((m) => m.dispose());
      });
    },
    [parts],
  );
  useEffect(() => {
    parts.forEach((part) =>
      part.materials.forEach((material) => {
        material.emissive.set(
          contacts?.box.includes(part.name)
            ? "#e84545"
            : contacts?.table.includes(part.name)
              ? "#dd9d22"
              : contacts?.bowl.includes(part.name)
                ? "#a45bd7"
                : "#000000",
        );
        material.emissiveIntensity = 0.45;
      }),
    );
  }, [parts, contacts]);
  return (
    <>
      {parts.map(({ name, geometry, materials }) => {
        const pose = poses[name];
        if (!pose) return null;
        return (
          <SmoothPose key={name} pose={pose} animate={animate} sample={sample}>
            <mesh
              geometry={geometry}
              material={materials}
              castShadow
              receiveShadow
              dispose={null}
            />
          </SmoothPose>
        );
      })}
    </>
  );
});

function Movable({
  id,
  position,
  selected,
  onSelect,
  onMove,
  children,
  vertical = false,
  animate = false,
  sample,
  orientation,
}: {
  id: string;
  position: Vec3;
  selected: string;
  onSelect?: (id: string) => void;
  onMove?: (id: string, p: Vec3) => void;
  children: React.ReactNode;
  vertical?: boolean;
  animate?: boolean;
  sample?: PlaybackSample;
  orientation?: [number, number, number, number];
}) {
  const group = useRef<Group>(null);
  const body = (
    <group
      ref={group}
      position={position}
      onClick={
        onSelect
          ? (e) => {
              e.stopPropagation();
              onSelect(id);
            }
          : undefined
      }
    >
      {children}
    </group>
  );
  if (!onMove)
    return (
      <SmoothPose
        position={position}
        orientation={orientation}
        animate={animate}
        sample={sample}
      >
        {children}
      </SmoothPose>
    );
  return (
    <>
      {body}
      {selected === id && onMove && (
        <TransformControls
          object={group as React.RefObject<Group>}
          mode="translate"
          showZ={vertical}
          size={0.65}
          onMouseUp={() => {
            if (group.current)
              onMove(id, group.current.position.toArray() as Vec3);
          }}
        />
      )}
    </>
  );
}
export function SceneView({
  model,
  poses,
  objects,
  bowl,
  bowlGeoms,
  bowlOrigin,
  ee,
  selected = "",
  onSelect,
  onMove,
  origin = BASE_ORIGIN,
  view,
  sensor,
  onOrbit,
  lines,
  target,
  contacts,
  region,
  zones,
  showRegion = true,
  animate = false,
  sample,
}: {
  model: Model;
  poses: Record<string, BodyPose>;
  objects: Item[];
  bowl: Vec3;
  bowlGeoms: BowlGeom[];
  bowlOrigin: Vec3;
  ee: Vec3;
  selected?: string;
  onSelect?: (id: string) => void;
  onMove?: (id: string, p: Vec3) => void;
  origin?: Vec3;
  view?: ViewCamera;
  sensor?: SensorCamera;
  onOrbit?: (view: ViewCamera) => void;
  lines?: ReplayViewState["lines"];
  target?: Vec3 | null;
  contacts?: ReplayViewState["contacts"];
  region?: Region;
  zones?: PickZones | null;
  showRegion?: boolean;
  animate?: boolean;
  sample?: PlaybackSample;
}) {
  return (
    <div
      className="scene-canvas"
      aria-label={
        sensor ? "Góc nhìn camera cổ tay" : "Cảnh 3D của mô hình MuJoCo"
      }
    >
      <Canvas
        shadows
        onCreated={({ gl }) => {
          gl.domElement.dataset.renderer = "three-webgl";
        }}
        dpr={[1, 1.5]}
        camera={{
          position: [0.68, -0.9, 1.0],
          up: [0, 0, 1],
          fov: 38,
          near: 0.01,
          far: 20,
        }}
        frameloop="demand"
        fallback={
          <p>
            Trình duyệt cần WebGL để hiển thị 3D. Bạn vẫn có thể chỉnh cảnh bằng
            tọa độ XY.
          </p>
        }
      >
        <color attach="background" args={["#151b20"]} />
        <ambientLight intensity={1.4} />
        <directionalLight
          position={[1, -0.5, 2]}
          intensity={2}
          castShadow
          shadow-mapSize={[1024, 1024]}
          shadow-camera-left={-1}
          shadow-camera-right={1}
          shadow-camera-top={1}
          shadow-camera-bottom={-1}
          shadow-bias={-0.0001}
        />
        <directionalLight position={[-1, 1, 1]} intensity={1} />
        <mesh position={[0, 0, 0.4]} receiveShadow>
          <boxGeometry args={[0.9, 0.8, 0.04]} />
          <meshStandardMaterial color="#354149" roughness={0.9} />
        </mesh>
        {!sensor && (
          <gridHelper
            args={[0.9, 18, "#718089", "#48565e"]}
            rotation={[Math.PI / 2, 0, 0]}
            position={[origin[0], origin[1], 0.4201]}
          />
        )}
        {!sensor && <BaseAxes origin={origin} />}
        {!sensor && showRegion && zones && <PickZones3D zones={zones} />}
        {!sensor && region && showRegion && <ReachBoundary region={region} />}
        {!sensor &&
          lines
            ?.filter((line) => line.points.length > 1)
            .map((line, k) => (
              <Line
                key={k}
                points={line.points}
                color={line.color}
                opacity={line.opacity}
                transparent
                lineWidth={1.2}
              />
            ))}
        {!sensor && target && (
          <mesh position={target}>
            <sphereGeometry args={[0.006, 12, 12]} />
            <meshBasicMaterial color="#6dc6ff" wireframe />
          </mesh>
        )}
        <Robot
          model={model}
          poses={poses}
          contacts={sensor ? undefined : contacts}
          animate={animate}
          sample={sample}
        />
        {objects.map((obj) => (
          <Movable
            key={obj.id}
            id={obj.id}
            orientation={
              obj.quat || [
                Math.cos((obj.yaw * Math.PI) / 360),
                0,
                0,
                Math.sin((obj.yaw * Math.PI) / 360),
              ]
            }
            animate={animate}
            sample={sample}
            position={obj.pos || [obj.x, obj.y, 0.435]}
            selected={selected}
            onSelect={onSelect}
            onMove={onMove}
          >
            <mesh
              castShadow
              quaternion={
                onMove && obj.quat
                  ? [obj.quat[1], obj.quat[2], obj.quat[3], obj.quat[0]]
                  : undefined
              }
              rotation={
                onMove && !obj.quat
                  ? [0, 0, (obj.yaw * Math.PI) / 180]
                  : undefined
              }
            >
              {obj.type === "sphere" ? (
                <sphereGeometry args={[0.015, 32, 24]} />
              ) : (
                <boxGeometry args={[0.03, 0.03, 0.03]} />
              )}
              <meshStandardMaterial color={obj.color} roughness={0.45} />
            </mesh>
          </Movable>
        ))}
        <Movable
          id="bowl"
          position={bowl}
          selected={selected}
          onSelect={onSelect}
          onMove={onMove}
        >
          {bowlGeoms.map((g, i) => (
            <mesh
              key={i}
              position={[
                g.pos[0] - bowlOrigin[0],
                g.pos[1] - bowlOrigin[1],
                g.pos[2] - bowlOrigin[2],
              ]}
              rotation={
                g.type === "cylinder" ? [Math.PI / 2, 0, 0] : [0, 0, g.yaw]
              }
              castShadow
              receiveShadow
            >
              {g.type === "cylinder" ? (
                <cylinderGeometry
                  args={[g.size[0], g.size[0], g.size[1] * 2, 64]}
                />
              ) : (
                <boxGeometry
                  args={[g.size[0] * 2, g.size[1] * 2, g.size[2] * 2]}
                />
              )}
              <meshStandardMaterial color="#729df0" roughness={0.4} />
            </mesh>
          ))}
        </Movable>
        {onMove && (
          <Movable
            id="arm"
            position={ee}
            selected={selected}
            onSelect={onSelect}
            onMove={onMove}
            vertical
          >
            <mesh>
              <sphereGeometry args={[0.007, 16, 16]} />
              <meshBasicMaterial color="#cf9dff" />
            </mesh>
          </Movable>
        )}
        {!sensor && (
          <OrbitControls
            makeDefault
            target={view ? undefined : [origin[0], origin[1], 0.48]}
            minDistance={0.25}
            maxDistance={2.5}
            maxPolarAngle={view ? Math.PI : Math.PI * 0.49}
          />
        )}
        {sensor ? (
          <SensorRig sensor={sensor} animate={animate} sample={sample} />
        ) : (
          <SceneRig view={view} ee={ee} onOrbit={onOrbit} />
        )}
        {!sensor && (
          <GizmoHelper alignment="bottom-right" margin={[65, 65]}>
            <GizmoViewport
              axisColors={["#ed8f87", "#75d7b4", "#8daff7"]}
              labelColor="#10171b"
            />
          </GizmoHelper>
        )}
      </Canvas>
      <div className="canvas-caption">
        {sensor ? (
          "Camera cổ tay"
        ) : (
          <>
            SO-101 <span>·</span>{" "}
            {onMove
              ? "Chọn vật để kéo trục · chuột phải dịch · lăn để zoom"
              : "Phát lại trạng thái đã ghi từ MuJoCo"}
          </>
        )}
      </div>
    </div>
  );
}
