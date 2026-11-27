import type { PickZones, ScanPoint, StopRegion } from "./zones";
import type { XYViewData } from "./components/topViewGeometry";
export type Vec3 = [number, number, number];
export type Item = {
  id: string;
  type: "box" | "sphere";
  color: string;
  x: number;
  y: number;
  yaw: number;
  size?: number; // hộp: cạnh, bóng: đường kính (m), mặc định 0,03
  pos?: Vec3;
  quat?: [number, number, number, number];
};
export type Pose = {
  pan: number;
  lift: number;
  elbow: number;
  wflex: number;
  wroll: number;
  grip: number;
  bx: number;
  by: number;
  byaw: number;
  wx: number;
  wy: number;
};
export type Bowl = { color: string; radius: number; height: number }; // radius: tới tâm tường (m), height: chiều cao tường (m)
export type Scene = { pose: Pose; objects: Item[]; target_id: string; bowl: Bowl; mode?: "single" | "all" };
export type Region = {
  pan: [number, number];
  r_min: number;
  r_max: number;
  angle_min_deg: number;
  angle_max_deg: number;
};
export type BodyPose = { p: Vec3; R: number[] };
export type ViewCamera = {
  yaw: number;
  pitch: number;
  dist: number;
  t: Vec3;
  follow: boolean;
  orthographic: boolean;
};
export type SensorCamera = BodyPose & {
  fovy: number;
  mount?: { bodyPose: BodyPose; local: BodyPose };
};
export type PlaybackSample = {
  stream: string;
  step: number;
  time: number;
  durationMs: number;
  continuous: boolean;
};
export type ReplayViewState = {
  xy?: XYViewData;
  poses: Record<string, BodyPose>;
  objects: Item[];
  bowl: Vec3;
  bowlColor?: string;
  bowlGeoms: BowlGeom[];
  ee: Vec3;
  origin: Vec3;
  region?: Region;
  zones?: PickZones | null;
  showRegion: boolean;
  sensor: SensorCamera | null;
  camera: ViewCamera;
  target: Vec3 | null;
  markers?: { hand: boolean; target: boolean; link: boolean };
  lines: { points: Vec3[]; color: string; opacity: number; width: number; dashed?: boolean }[];
  contacts: { box: string[]; table: string[]; tablePts?: Vec3[]; bowl: string[]; cam?: string[] };
  playing: boolean;
  sample: PlaybackSample;
};
export type Snapshot = {
  pose: Record<string, BodyPose>;
  ee: Vec3;
  angles: number[];
  contacts: { a: string; b: string; pen_mm: number }[];
  tilt: number;
  clearance_mm: Record<string, number>;
};
export type BowlGeom = { type: string; pos: Vec3; size: Vec3; yaw: number };
export type Meta = {
  home: Partial<Pose>;
  limits: [number, number][];
  gopen: number;
  gclose: number;
  saved_pose: (Pose & { objects?: Item[]; target_id?: string; bowl?: Bowl; mode?: Scene["mode"] }) | null;
  pick_region: Region;
  pick_scan?: ScanPoint[];
  unreachable_region?: StopRegion;
  bowl_geoms: BowlGeom[];
  bowl_origin: Vec3;
};
export type Model = {
  mesh: Record<
    string,
    { v: number[]; f: number[]; colors: [number, number[], number][] }
  >;
};
export type Frame = {
  t: number;
  ph: number;
  phn: number;
  ee: Vec3;
  jq: number[];
  fo: number;
  ok: boolean;
  pose?: Record<string, BodyPose>;
  objects?: Item[];
  ct: string[];
  tb: string[];
};
export type Replay = {
  scene: { step_dt: number };
  episodes: {
    frames: Frame[];
    bowl: Vec3;
    bowl_geoms: BowlGeom[];
    success: boolean;
    steps: number;
    blocked_reason?: string;
    initial_options: { box_xy: number[]; bowl_xy: number[] };
  }[];
};
export type Run = {
  status: "idle" | "running" | "done" | "error";
  run_id?: string;
  success?: boolean;
  steps?: number;
  error?: string;
  blocked_reason?: string;
  placed?: number;
  total?: number;
};
export type RunRecord = {
  id: string;
  title: string;
  created_at: string;
  success: boolean;
  steps: number;
  blocked_reason?: string;
  objects: number;
  seed: number;
  favorite: boolean;
  notes: string;
  duration: number;
};
export type EpisodeSummary = {
  index: number;
  seed: number;
  name: string;
  success: boolean;
  steps: number;
  tags: string[];
  blocked_reason?: string;
  zone?: "green" | "mixed" | "red" | null;
};
export type Catalog = {
  sources: { id: string; title: string; episodes: EpisodeSummary[] }[];
  runs: RunRecord[];
};
export type Health = {
  status: string;
  mujoco: string;
  control_hz: number;
  physics_dt: number;
  max_steps: number;
  scene: string;
  workspace: Region;
  run: Run;
  capabilities: string[];
};
export const JOINTS = [
  ["pan", "Xoay đế"],
  ["lift", "Nâng vai"],
  ["elbow", "Gập khuỷu"],
  ["wflex", "Gập cổ tay"],
  ["wroll", "Xoay cổ tay"],
] as const;
