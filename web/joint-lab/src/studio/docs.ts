import type { Catalog } from "../types";

// Hướng dẫn Markdown (docs/*.md) hiển thị trong màn "Hướng dẫn".
// Link dạng studio:replay?seed=2&step=40 hoặc studio:lab?seed=2 mở thẳng Replay / Lab của Studio;
// trên GitHub chúng chỉ hiện thành chữ.

export type DocHeading = { depth: 2 | 3; text: string; id: string };

export function slugify(text: string) {
  return text
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .replace(/đ/g, "d")
    .replace(/Đ/g, "D")
    .toLowerCase()
    .replace(/`/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

// Bỏ định dạng inline để lấy chữ thuần của tiêu đề (cùng chữ mà React render ra, để id khớp nhau).
export function plainText(markdown: string) {
  return markdown
    .replace(/`([^`]*)`/g, "$1")
    .replace(/\*\*([^*]*)\*\*/g, "$1")
    .replace(/\*([^*]*)\*/g, "$1")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .trim();
}

export function headings(markdown: string): DocHeading[] {
  const out: DocHeading[] = [];
  let fenced = false;
  for (const line of markdown.split("\n")) {
    if (line.startsWith("```")) fenced = !fenced;
    if (fenced) continue;
    const m = /^(##|###) (.+)$/.exec(line);
    if (m) {
      const text = plainText(m[2]);
      out.push({ depth: m[1].length as 2 | 3, text, id: slugify(text) });
    }
  }
  return out;
}

export type StudioLink =
  | { screen: "replay"; seed: number; step?: number; source: string }
  | { screen: "lab"; seed: number; source: string };

export function parseStudioLink(href: string | undefined): StudioLink | null {
  const m = /^studio:(replay|lab)\?(.*)$/.exec(href || "");
  if (!m) return null;
  const q = new URLSearchParams(m[2]);
  const seed = Number(q.get("seed"));
  if (!Number.isInteger(seed)) return null;
  const source = q.get("source") || "baseline";
  if (m[1] === "lab") return { screen: "lab", seed, source };
  const step = q.has("step") ? Number(q.get("step")) : undefined;
  return {
    screen: "replay",
    seed,
    source,
    step: Number.isInteger(step) && step! >= 0 ? step : undefined,
  };
}

// Vị trí (episode index) của seed trong một nguồn replay; null nếu nguồn chưa có seed đó.
export function episodeIndex(catalog: Catalog, source: string, seed: number) {
  const episodes = catalog.sources.find((s) => s.id === source)?.episodes;
  const found = episodes?.find((e) => e.seed === seed);
  return found ? found.index : null;
}

// Ảnh trong Markdown viết tương đối (image/x.svg) để đọc được trên GitHub; Studio lấy qua server.
export function docAssetUrl(src: string | undefined) {
  if (!src || /^(https?:|data:|\/)/.test(src)) return src;
  return "/studio/docs/" + src.replace(/^\.\//, "");
}
