import { describe, expect, it } from "vitest";
import { filterEpisodes } from "./replayFilters";
import type { EpisodeSummary } from "../types";

const ep = (index: number, success: boolean, zone: EpisodeSummary["zone"], blocked?: string): EpisodeSummary => ({
  index,
  seed: index,
  name: `seed ${index}`,
  success,
  steps: 100,
  tags: [],
  zone,
  blocked_reason: blocked,
});
const episodes = [ep(0, true, "green"), ep(1, false, "mixed"), ep(2, false, "red", "ngoài vùng"), ep(3, true, "mixed")];

describe("replay filters", () => {
  it("filters by result and zone", () => {
    expect(filterEpisodes(episodes, "ok", "all", 0).map((e) => e.index)).toEqual([0, 3]);
    expect(filterEpisodes(episodes, "all", "mixed", 1).map((e) => e.index)).toEqual([1, 3]);
    expect(filterEpisodes(episodes, "fail", "all", 1).map((e) => e.index)).toEqual([1]);
    expect(filterEpisodes(episodes, "blocked", "red", 2).map((e) => e.index)).toEqual([2]);
  });

  it("keeps the current episode, marked as outside the filter", () => {
    const out = filterEpisodes(episodes, "ok", "green", 1);
    expect(out.map((e) => [e.index, e.matches])).toEqual([
      [0, true],
      [1, false],
    ]);
  });
});

import { episodeHaystack, fold, matchesSearch } from "./replayFilters";
describe("replay search", () => {
  const e = { ...ep(1011, false, "mixed"), name: "seed 1011 · hộp XA", tags: ["HỎNG: chưa kẹp", "timeout phase 1"] };
  it("folds diacritics and case", () => {
    expect(fold("HỎNG Đế")).toBe("hong de");
  });
  it("matches seed number, tags and multiple words", () => {
    const h = episodeHaystack(e);
    expect(matchesSearch(h, "1011")).toBe(true);
    expect(matchesSearch(h, "hong timeout")).toBe(true);
    expect(matchesSearch(h, "chậm")).toBe(false);
    expect(matchesSearch(h, "")).toBe(true);
  });
});
