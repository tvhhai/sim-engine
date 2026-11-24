import { useId, useMemo, useRef } from "react";
import type { PointerEvent, ReactNode } from "react";
import { visibleScan, zoneShapes, ZONE_COLORS } from "../zones";
import { boxOutline, dragPosition, fromTop, topX, topY, wallOutline, xyPoints, XY_BOUNDS, XY_SCALE } from "./topViewGeometry";
import type { Point2, XYViewData } from "./topViewGeometry";

const INK = "var(--ink,var(--mantine-color-text,#e6e9ee))";
const MUTED = "var(--muted,var(--mantine-color-dimmed,#9aa3af))";
const CARD = "var(--card,var(--studio-surface,#1d2128))";
const HAND = "var(--hand,#e879f9)", BOX = "var(--box,#ff6b69)";

// Shared by Lab and Replay. Editing is enabled only when onMove is supplied.
export function TopView({ data, showRegion = true, selected = "", onSelect, onMove }: {
  data: XYViewData;
  showRegion?: boolean;
  selected?: string;
  onSelect?: (id: string) => void;
  onMove?: (id: string, x: number, y: number) => void;
}) {
  const clipId = useId();
  const drag = useRef<{ id: string; offset: Point2; pointer: number } | null>(null);
  const zoneLayer = useMemo(() => data.zones ? zoneShapes(data.zones, { x0: -data.table[0], x1: data.table[0], y0: -data.table[1], y1: data.table[1] }) : null, [data.zones, data.table]);
  const scan = useMemo(() => data.zones ? visibleScan(data.zones) : [], [data.zones]);
  function point(event: PointerEvent<SVGElement>): Point2 | null {
    const svg = event.currentTarget.tagName.toLowerCase() === "svg" ? event.currentTarget as SVGSVGElement : event.currentTarget.ownerSVGElement;
    const matrix = svg?.getScreenCTM();
    if (!svg || !matrix) return null;
    const p = svg.createSVGPoint();
    p.x = event.clientX; p.y = event.clientY;
    const local = p.matrixTransform(matrix.inverse());
    return fromTop(local.x, local.y);
  }
  function move(event: PointerEvent<SVGSVGElement>) {
    const active = drag.current, p = point(event);
    if (!active || active.pointer !== event.pointerId || !p || !onMove) return;
    onMove(active.id, ...dragPosition(p, active.offset, data.table));
  }
  const editable = (id: string, x: number, y: number, label: string, children: ReactNode) => (
    <g key={id} data-object={id} aria-label={label} role={onMove ? "button" : "img"} tabIndex={onMove ? 0 : undefined}
      style={{ cursor: onMove ? "grab" : undefined }}
      onPointerDown={onMove ? (event) => {
        if (event.button !== 0) return;
        const p = point(event);
        if (!p) return;
        event.preventDefault(); event.stopPropagation();
        drag.current = { id, offset: [p[0] - x, p[1] - y], pointer: event.pointerId };
        onSelect?.(id);
        event.currentTarget.ownerSVGElement?.setPointerCapture(event.pointerId);
      } : undefined}
      onKeyDown={onMove ? (event) => {
        const offsets: Record<string, Point2> = { ArrowLeft: [-0.005, 0], ArrowRight: [0.005, 0], ArrowUp: [0, 0.005], ArrowDown: [0, -0.005] };
        const offset = offsets[event.key];
        if (!offset) return;
        event.preventDefault(); event.stopPropagation();
        onSelect?.(id);
        onMove(id, ...dragPosition([x + offset[0], y + offset[1]], [0, 0], data.table));
      } : undefined}>
      {children}
      <title>{label} · x {(x - data.origin[0]).toFixed(3)}, y {(y - data.origin[1]).toFixed(3)} m từ đế</title>
    </g>
  );
  const axes: ReactNode[] = [];
  const num = (v: number) => (Math.abs(v) < 1e-9 ? 0 : v).toFixed(2);
  for (let axis = 0; axis < 2; axis++) {
    const low = axis ? XY_BOUNDS.y0 : XY_BOUNDS.x0, high = axis ? XY_BOUNDS.y1 : XY_BOUNDS.x1;
    for (let tick = Math.ceil((low - data.origin[axis]) / 0.05 - 1e-9); tick * 0.05 <= high - data.origin[axis] + 1e-9; tick++) {
      const value = tick * 0.05, pos = axis ? topY(value + data.origin[1]) : topX(value + data.origin[0]);
      const zero = tick === 0, major = tick % 2 === 0;
      axes.push(<g key={`${axis}/${tick}`}>
        <line x1={axis ? 10 : pos} y1={axis ? pos : 10} x2={axis ? 510 : pos} y2={axis ? pos : 330} stroke={MUTED} strokeWidth={zero ? 1.4 : 0.5} opacity={zero ? 0.7 : major ? 0.35 : 0.18} />
        {major && (!axis || pos < 316) && <text x={axis ? 13 : pos} y={axis ? pos - 2 : 327} fontSize={9} textAnchor={axis ? "start" : "middle"} fill={MUTED}>{num(value)}</text>}
      </g>);
    }
  }
  const { bowl, arm, ee } = data;
  const bowlColor = bowl.color || "var(--bowl,#6ea0ff)";
  const initial = data.initialObject, current = initial && data.objects.find((o) => o.id === initial.id);
  return (
    <svg className="xy-view" viewBox="0 0 520 340" aria-label="Nhìn từ trên · XY (m)"
      style={{ width: "100%", height: "100%", minHeight: 0, display: "block", touchAction: onMove ? "none" : undefined, userSelect: "none" }}
      onPointerMove={onMove ? move : undefined}
      onPointerUp={(event) => { move(event); drag.current = null; if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId); }}
      onPointerCancel={() => { drag.current = null; }} onLostPointerCapture={() => { drag.current = null; }}>
      <defs><clipPath id={clipId}><rect x={topX(-data.table[0])} y={topY(data.table[1])} width={data.table[0] * 2 * XY_SCALE} height={data.table[1] * 2 * XY_SCALE} /></clipPath></defs>
      <rect x={topX(-data.table[0])} y={topY(data.table[1])} width={data.table[0] * 2 * XY_SCALE} height={data.table[1] * 2 * XY_SCALE} fill="var(--table,var(--xy-table,#4a4130))" opacity={0.5} />
      {axes}
      <text x={508} y={314} fontSize={11} fontWeight={700} textAnchor="end" fill={INK}>x (m) →</text>
      <text x={13} y={22} fontSize={11} fontWeight={700} fill={INK}>↑ y (m)</text>
      {showRegion && zoneLayer && <g clipPath={`url(#${clipId})`} pointerEvents="none">
        {zoneLayer.pieces.map(([zone, points], k) => <polygon key={k} points={xyPoints(points)} fill={ZONE_COLORS[zone]} fillOpacity={zone === "mixed" ? 0.13 : 0.12} stroke={ZONE_COLORS[zone]} strokeOpacity={0.55} strokeWidth={0.8} />)}
        {zoneLayer.outsideHole && <path d={`M${topX(-data.table[0])},${topY(data.table[1])}h${data.table[0] * 2 * XY_SCALE}v${data.table[1] * 2 * XY_SCALE}h${-data.table[0] * 2 * XY_SCALE}Z M${zoneLayer.outsideHole.map(([x, y]) => `${topX(x)},${topY(y)}`).join("L")}Z`} fill={ZONE_COLORS.red} fillOpacity={0.12} fillRule="evenodd" />}
        {scan.map(({ x, y, ok }, k) => ok ? <circle key={k} cx={topX(x)} cy={topY(y)} r={1.6} fill={ZONE_COLORS.green} opacity={0.75} /> : <path key={k} d={`M${topX(x) - 2.2},${topY(y) - 2.2}l4.4,4.4m0,-4.4l-4.4,4.4`} stroke={ZONE_COLORS.red} strokeWidth={1.3} strokeLinecap="round" />)}
      </g>}
      {editable("bowl", bowl.pos[0], bowl.pos[1], "Bát", <>
        <text x={topX(bowl.pos[0])} y={topY(bowl.pos[1]) - 0.06 * XY_SCALE} fontSize={10} textAnchor="middle" fill={bowlColor} fontWeight={700}>bát</text>
        {bowl.geoms.map((g, k) => g.type === "cylinder" ? <circle key={k} cx={topX(g.pos[0])} cy={topY(g.pos[1])} r={g.size[0] * XY_SCALE} fill={bowlColor} opacity={0.2} /> : <polygon key={k} points={xyPoints(wallOutline(g))} fill={bowlColor} fillOpacity={0.4} stroke={bowlColor} strokeWidth={1} />)}
        <circle cx={topX(bowl.pos[0])} cy={topY(bowl.pos[1])} r={bowl.radius * XY_SCALE} fill="transparent" stroke={selected === "bowl" ? INK : bowlColor} strokeDasharray="4 3" />
      </>)}
      {initial && <g pointerEvents="none">
        {initial.type === "sphere" ? <circle cx={topX(initial.x)} cy={topY(initial.y)} r={(initial.size ?? 0.03) / 2 * XY_SCALE} fill="none" stroke={BOX} strokeDasharray="3 3" opacity={0.6} /> : <polygon points={xyPoints(boxOutline(initial))} fill="none" stroke={BOX} strokeDasharray="3 3" opacity={0.6} />}
        <text x={topX(initial.x)} y={topY(initial.y) + (initial.size ?? 0.03) / 2 * XY_SCALE + 11} fontSize={9} textAnchor="middle" fill={BOX}>{initial.type === "sphere" ? "bóng" : "hộp"} ban đầu</text>
        {current && Math.hypot(current.x - initial.x, current.y - initial.y) > 0.004 && <line x1={topX(initial.x)} y1={topY(initial.y)} x2={topX(current.x)} y2={topY(current.y)} stroke="var(--bad,#ff7b72)" strokeWidth={2} strokeDasharray="2 2" />}
      </g>}
      {data.ghosts?.map((ghost, k) => <g key={k} pointerEvents="none">
        <polyline points={xyPoints(ghost.joints)} fill="none" stroke={ghost.color} strokeWidth={5} opacity={ghost.opacity} strokeLinecap="round" strokeLinejoin="round" />
        {ghost.joints.length > 0 && <text x={topX(ghost.joints[ghost.joints.length - 1][0]) + 5} y={topY(ghost.joints[ghost.joints.length - 1][1]) - 5} fontSize={9} fill={ghost.color} opacity={0.9}>{ghost.label}</text>}
      </g>)}
      {arm && <g pointerEvents="none">
        {arm.hulls.map((hull, k) => <polygon key={k} points={xyPoints(hull.points)} fill={hull.color} fillOpacity={0.3} stroke="#3a2c00" strokeOpacity={0.5} strokeWidth={0.8} />)}
        {arm.joints.slice(1).map((p, k) => <line key={k} x1={topX(arm.joints[k][0])} y1={topY(arm.joints[k][1])} x2={topX(p[0])} y2={topY(p[1])} stroke={arm.colors[k] || MUTED} strokeWidth={2.2} strokeLinecap="round" opacity={0.9} />)}
        {arm.joints.slice(1, 6).map((p, k) => <g key={k}><circle cx={topX(p[0])} cy={topY(p[1])} r={7} fill={CARD} stroke={INK} strokeWidth={1.2} /><text x={topX(p[0])} y={topY(p[1]) + 3.2} fontSize={9} fontWeight={700} textAnchor="middle" fill={INK}>{k + 1}</text></g>)}
      </g>}
      {data.paths?.map((path, k) => <g key={k} pointerEvents="none"><polyline points={xyPoints(path.points)} fill="none" stroke={path.color} strokeWidth={path.width} opacity={path.opacity ?? 1} strokeDasharray={path.dash} />{path.dots && path.points.slice(1).map((p, j) => <circle key={j} cx={topX(p[0])} cy={topY(p[1])} r={3} fill={path.color} />)}</g>)}
      {data.objects.map((obj, k) => editable(obj.id, obj.x, obj.y, `${obj.type === "sphere" ? "Bóng" : "Hộp"} ${k + 1}${data.targetId === obj.id ? " · đích" : ""}`, <>
        {obj.type === "sphere" ? <circle cx={topX(obj.x)} cy={topY(obj.y)} r={(obj.size ?? 0.03) / 2 * XY_SCALE} fill={obj.color} stroke={selected === obj.id ? INK : "#00000066"} strokeWidth={selected === obj.id ? 2 : 1} /> : <polygon points={xyPoints(boxOutline(obj))} fill={obj.color} stroke={selected === obj.id ? INK : "#00000066"} strokeWidth={selected === obj.id ? 2 : 1} />}
        {onMove && <circle cx={topX(obj.x)} cy={topY(obj.y)} r={Math.max(9, (obj.size ?? 0.03) / 2 * XY_SCALE)} fill="transparent" />}
      </>))}
      {data.target && ee && <g pointerEvents="none"><line x1={topX(ee[0])} y1={topY(ee[1])} x2={topX(data.target[0])} y2={topY(data.target[1])} stroke="var(--warn,#fbbf24)" strokeWidth={2} /><circle cx={topX(data.target[0])} cy={topY(data.target[1])} r={7} fill="none" stroke="var(--goal,#4ade80)" strokeWidth={2.5} /></g>}
      {ee && editable("arm", ee[0], ee[1], "Đầu tay", <><path d={`M${topX(ee[0]) - 8} ${topY(ee[1])}h16 M${topX(ee[0])} ${topY(ee[1]) - 8}v16`} stroke={HAND} strokeWidth={2.5} />{onMove && <circle cx={topX(ee[0])} cy={topY(ee[1])} r={10} fill="transparent" />}</>)}
      <g pointerEvents="none"><circle cx={topX(data.origin[0])} cy={topY(data.origin[1])} r={3} fill={CARD} stroke={INK} strokeWidth={1.25} /><path d={`M${topX(data.origin[0])} ${topY(data.origin[1])}l-8 19h-10`} fill="none" stroke={INK} /><text x={topX(data.origin[0]) - 23} y={topY(data.origin[1]) + 22} textAnchor="end" fill={INK} stroke={CARD} strokeWidth={3} paintOrder="stroke" fontSize={10} fontWeight={600}>(0, 0) · tâm khớp 1</text></g>
    </svg>
  );
}
