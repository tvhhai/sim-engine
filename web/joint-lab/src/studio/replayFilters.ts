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

// Bỏ dấu và chữ hoa để "hong" khớp "Hỏng"
export const fold = (s: string) =>
  s.normalize("NFD").replace(/[̀-ͯ]/g, "").replace(/đ/g, "d").replace(/Đ/g, "D").toLowerCase();

// Chuỗi tìm kiếm của một episode: số seed, tên, nhãn (hỏng, timeout, chậm...) và kết quả
export const episodeHaystack = (e: EpisodeSummary) =>
  fold([e.seed, `seed ${e.seed}`, e.name, ...e.tags, e.success ? "ok thanh cong" : "fail hong"].join(" "));

// Nhiều từ = phải khớp hết, theo thứ tự bất kỳ
export const matchesSearch = (haystack: string, query: string) =>
  fold(query).split(/\s+/).filter(Boolean).every((w) => haystack.includes(w));

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
