import type { EpisodeSummary } from "../types";

// Bộ lọc danh sách episode của Replay, cùng nghĩa với replay canvas (static/replay/replay.js: bộ lọc "Vùng" và "Kết quả").
export const RESULT_FILTERS = [
  { value: "all", label: "Tất cả" },
  { value: "ok", label: "Thành công" },
  { value: "fail", label: "Thất bại" },
  { value: "blocked", label: "Đứng yên" },
];
export const ZONE_FILTERS = [
  { value: "all", label: "Tất cả vùng" },
  { value: "green", label: "Gắp được" },
  { value: "mixed", label: "Hỗn hợp" },
  { value: "red", label: "Không gắp được" },
];

export function matchesResult(e: EpisodeSummary, result: string) {
  if (result === "ok") return e.success;
  if (result === "fail") return !e.success && !e.blocked_reason;
  if (result === "blocked") return !!e.blocked_reason;
  return true;
}

// Episode đang xem luôn có trong danh sách (đánh dấu ngoài bộ lọc) để ô chọn không bị trống.
export function filterEpisodes(
  episodes: EpisodeSummary[],
  result: string,
  zone: string,
  current: number,
) {
  return episodes.flatMap((e) => {
    const matches =
      matchesResult(e, result) && (zone === "all" || e.zone === zone);
    return matches || e.index === current ? [{ ...e, matches }] : [];
  });
}
