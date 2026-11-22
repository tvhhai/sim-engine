export const PANEL_LABELS = {
  map: "Tọa độ XY",
  objects: "Vật thể",
  joints: "Khớp và hàm kẹp",
  bowl: "Vị trí bát",
  metrics: "Số liệu và va chạm",
};
export type Settings = {
  schema_version: 1;
  theme: "dark" | "light" | "auto";
  startPage: "replay" | "lab" | "runs" | "settings";
  defaultSpeed: string;
  autoPlay: boolean;
  autoOpenReplay: boolean;
  collapsed: boolean;
  panels: Record<keyof typeof PANEL_LABELS, boolean>;
};
export const DEFAULT_SETTINGS: Settings = {
  schema_version: 1,
  theme: "dark",
  startPage: "replay",
  defaultSpeed: "1",
  autoPlay: false,
  autoOpenReplay: true,
  collapsed: false,
  panels: { map: true, objects: true, joints: true, bowl: true, metrics: true },
};
export function parseSettings(value: unknown): Settings {
  if (!value || typeof value !== "object")
    throw new Error("Tệp cài đặt không hợp lệ");
  const v = value as Partial<Settings>;
  if (v.schema_version !== undefined && v.schema_version !== 1)
    throw new Error("Phiên bản cài đặt chưa được hỗ trợ");
  const result: Settings = {
    ...DEFAULT_SETTINGS,
    panels: { ...DEFAULT_SETTINGS.panels },
  };
  if (["dark", "light", "auto"].includes(v.theme || ""))
    result.theme = v.theme!;
  if (["replay", "lab", "runs", "settings"].includes(v.startPage || ""))
    result.startPage = v.startPage!;
  if (
    ["0.1", "0.25", "0.5", "1", "1.5", "2", "3", "4"].includes(
      v.defaultSpeed || "",
    )
  )
    result.defaultSpeed = v.defaultSpeed!;
  for (const key of ["autoPlay", "autoOpenReplay", "collapsed"] as const)
    if (typeof v[key] === "boolean") result[key] = v[key]!;
  if (v.panels && typeof v.panels === "object")
    for (const key of Object.keys(
      PANEL_LABELS,
    ) as (keyof typeof PANEL_LABELS)[])
      if (typeof v.panels[key] === "boolean")
        result.panels[key] = v.panels[key];
  return result;
}
