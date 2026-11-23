import { useMemo, useRef } from "react";
import type { PointerEvent } from "react";
import type { Region, Scene, Snapshot } from "../types";
import { reachable, sector } from "../workspace";
import { TABLE_HALF, ZONE_COLORS, visibleScan, zoneShapes } from "../zones";
import type { PickZones } from "../zones";
const X = (v: number) => (v + 0.45) * 1000,
  Y = (v: number) => (0.4 - v) * 1000;
export function TopView({
  scene,
  region,
  zones,
  snapshot,
  selected,
  onSelect,
  onMove,
}: {
  scene: Scene;
  region: Region;
  zones?: PickZones;
  snapshot: Snapshot | null;
  selected: string;
  onSelect: (id: string) => void;
  onMove: (id: string, x: number, y: number) => void;
}) {
  const drag = useRef<string | null>(null);
  // Vùng xanh / hỗn hợp / đỏ và điểm quét (✕ hỏng; chấm chỉ ở vùng hỗn hợp), cùng quy tắc với replay
  const zoneLayer = useMemo(() => {
    if (!zones) return null;
    const { pieces, outsideHole } = zoneShapes(zones);
    const poly = (pts: [number, number][]) => pts.map(([x, y]) => `${X(x)},${Y(y)}`).join(" ");
    const [hx, hy] = TABLE_HALF;
    const outside = outsideHole
      ? `M${X(-hx)},${Y(hy)}H${X(hx)}V${Y(-hy)}H${X(-hx)}Z M` +
        outsideHole.map(([x, y]) => `${X(x)},${Y(y)}`).join(" L") + "Z"
      : null;
    return (
      <g aria-hidden="true">
        {pieces.map(([zone, pts], i) => (
          <polygon key={i} points={poly(pts)} fill={ZONE_COLORS[zone]} fillOpacity={zone === "green" ? 0.12 : 0.15} />
        ))}
        {outside && <path d={outside} fill={ZONE_COLORS.red} fillOpacity={0.15} fillRule="evenodd" />}
        {visibleScan(zones).map(({ x, y, ok }, i) =>
          ok ? (
            <circle key={i} cx={X(x)} cy={Y(y)} r={3} fill="#3fcf8e" opacity={0.8} />
          ) : (
            <path
              key={i}
              d={`M${X(x) - 4},${Y(y) - 4}l8,8m0,-8l-8,8`}
              stroke={ZONE_COLORS.red}
              strokeWidth={2.2}
              strokeLinecap="round"
            />
          ),
        )}
      </g>
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(zones?.region), JSON.stringify(zones?.stop), zones?.scan?.length]);
  function move(e: PointerEvent<SVGSVGElement>) {
    if (!drag.current) return;
    const matrix = e.currentTarget.getScreenCTM();
    if (!matrix) return;
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(
      matrix.inverse(),
    );
    onMove(
      drag.current,
      Math.max(-0.435, Math.min(0.435, p.x / 1000 - 0.45)),
      Math.max(-0.385, Math.min(0.385, 0.4 - p.y / 1000)),
    );
  }
  const handle = (
    id: string,
    x: number,
    y: number,
    color: string,
    label: string,
    radiusPx = id === "bowl" ? 44 : 17,
  ) => (
    <g
      key={id}
      role="button"
      tabIndex={0}
      aria-label={label}
      data-object={id}
      style={{ cursor: "grab" }}
      onPointerDown={(e) => {
        e.stopPropagation();
        drag.current = id;
        onSelect(id);
        e.currentTarget.ownerSVGElement?.setPointerCapture(e.pointerId);
      }}
      onKeyDown={(e) => {
        const offsets: Record<string, number[]> = {
          ArrowLeft: [-0.005, 0],
          ArrowRight: [0.005, 0],
          ArrowUp: [0, 0.005],
          ArrowDown: [0, -0.005],
        };
        if (offsets[e.key]) {
          e.preventDefault();
          onSelect(id);
          onMove(id, x + offsets[e.key][0], y + offsets[e.key][1]);
        }
      }}
    >
      <circle
        cx={X(x)}
        cy={Y(y)}
        r={radiusPx}
        fill={color}
        fillOpacity={id === "bowl" ? 0.15 : 0.9}
        stroke={selected === id ? "#f3f4f6" : color}
        strokeWidth={selected === id ? 3 : 2}
      />
      {id === "arm" && (
        <path
          d={`M${X(x) - 10} ${Y(y)}h20 M${X(x)} ${Y(y) - 10}v20`}
          stroke="#fff"
        />
      )}
      <text
        x={X(x)}
        y={Y(y) - (id === "bowl" ? 54 : 27)}
        textAnchor="middle"
        fill="#d1d7df"
        fontSize="16"
      >
        {label}
      </text>
      <title>
        {label} · x {((x - region.pan[0]) * 100).toFixed(1)}, y{" "}
        {((y - region.pan[1]) * 100).toFixed(1)} cm từ đế
      </title>
    </g>
  );
  return (
    <div className="top-view">
      <svg
        viewBox="-35 -15 955 845"
        aria-label="Tọa độ XY: kéo vật thể và bát"
        onPointerMove={move}
        onPointerUp={(e) => {
          move(e);
          drag.current = null;
        }}
        onPointerCancel={() => {
          drag.current = null;
        }}
      >
        <rect width="900" height="800" rx="8" fill="#171d22" />
        {Array.from({ length: 19 }, (_, i) => {
          const v = i * 0.05 - 0.45;
          return (
            <g key={i}>
              <line x1={X(v)} y1={0} x2={X(v)} y2={800} stroke="#2a333b" />
              <text
                x={X(v)}
                y={825}
                textAnchor="middle"
                fill="#8b96a3"
                fontSize="15"
              >
                {Math.round((v - region.pan[0]) * 100)}
              </text>
            </g>
          );
        })}
        {Array.from({ length: 17 }, (_, i) => {
          const v = i * 0.05 - 0.4;
          return (
            <g key={i}>
              <line x1={0} y1={Y(v)} x2={900} y2={Y(v)} stroke="#2a333b" />
              <text
                x={-12}
                y={Y(v) + 5}
                textAnchor="end"
                fill="#8b96a3"
                fontSize="15"
              >
                {Math.round((v - region.pan[1]) * 100)}
              </text>
            </g>
          );
        })}
        {zoneLayer}
        <polygon
          points={sector(region)
            .map(([x, y]) => `${X(x)},${Y(y)}`)
            .join(" ")}
          fill="#4ad8a8"
          fillOpacity=".065"
          stroke="#4ad8a8"
          strokeOpacity=".6"
          strokeDasharray="7 6"
        />
        <line
          x1={X(region.pan[0])}
          y1={0}
          x2={X(region.pan[0])}
          y2={800}
          stroke="#6b7884"
        />
        <line
          x1={0}
          y1={Y(region.pan[1])}
          x2={900}
          y2={Y(region.pan[1])}
          stroke="#6b7884"
        />
        <rect
          x={X(region.pan[0]) - 48}
          y={Y(region.pan[1]) - 48}
          width="96"
          height="96"
          rx="12"
          fill="#49565e"
        />
        <circle
          cx={X(region.pan[0])}
          cy={Y(region.pan[1])}
          r={5}
          fill="#e8edf2"
        />
        <text
          x={X(region.pan[0]) + 54}
          y={Y(region.pan[1]) + 6}
          fill="#e8edf2"
          fontSize={15}
        >
          O (0, 0) · đế
        </text>
        {snapshot && (
          <polyline
            points={[
              [region.pan[0], region.pan[1]],
              ...Object.values(snapshot.pose)
                .filter((p) => p.p[2] > 0.45)
                .map((p) => p.p),
            ]
              .map((p) => `${X(p[0])},${Y(p[1])}`)
              .join(" ")}
            fill="none"
            stroke="#e7bb54"
            strokeWidth="8"
            opacity=".3"
          />
        )}
        {handle("bowl", scene.pose.wx, scene.pose.wy, scene.bowl.color, "Bát", 44 * (scene.bowl.radius / 0.045))}
        {scene.objects.map((obj, i) => (
          <g key={obj.id}>
            {handle(
              obj.id,
              obj.x,
              obj.y,
              obj.color,
              `${obj.type === "sphere" ? "Bóng" : "Hộp"} ${i + 1}${scene.target_id === obj.id ? " · đích" : ""}`,
              17 * ((obj.size ?? 0.03) / 0.03),
            )}
            {!reachable(obj.x, obj.y, region) && (
              <text
                x={X(obj.x) + 22}
                y={Y(obj.y) + 6}
                fill="#f59e8b"
                fontSize="22"
              >
                !
              </text>
            )}
          </g>
        ))}
        {snapshot &&
          handle("arm", snapshot.ee[0], snapshot.ee[1], "#c298f5", "Đầu tay")}
      </svg>
      <div className="map-legend">
        <span>
          <i className="dot mint" />
          Vùng thao tác
        </span>
        <span>X / Y · cm từ đế</span>
        <span>Kéo hoặc dùng phím mũi tên</span>
      </div>
    </div>
  );
}
