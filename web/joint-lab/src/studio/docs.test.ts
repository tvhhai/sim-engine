import { describe, expect, it } from "vitest";
import {
  docAssetUrl,
  episodeIndex,
  headings,
  parseStudioLink,
  slugify,
} from "./docs";

describe("docs helpers", () => {
  it("slugs Vietnamese headings to ASCII ids", () => {
    expect(slugify("10.2 Bên trong `select_action` mỗi bước")).toBe(
      "10-2-ben-trong-select-action-moi-buoc",
    );
    expect(slugify("Đường đi")).toBe("duong-di");
  });

  it("collects h2/h3 outside code fences", () => {
    const md = "# T\n## 1. Một\n```python\n## not a heading\n```\n### Hai **đậm**\n";
    expect(headings(md)).toEqual([
      { depth: 2, text: "1. Một", id: "1-mot" },
      { depth: 3, text: "Hai đậm", id: "hai-dam" },
    ]);
  });

  it("parses studio links", () => {
    expect(parseStudioLink("studio:replay?seed=2&step=40")).toEqual({
      screen: "replay",
      seed: 2,
      step: 40,
      source: "baseline",
    });
    expect(parseStudioLink("studio:lab?seed=7&source=lab")).toEqual({
      screen: "lab",
      seed: 7,
      source: "lab",
    });
    expect(parseStudioLink("https://x.y")).toBeNull();
    expect(parseStudioLink("studio:replay?seed=abc")).toBeNull();
  });

  it("finds episode index by seed", () => {
    const catalog = {
      runs: [],
      sources: [
        {
          id: "baseline",
          title: "",
          episodes: [
            { index: 0, seed: 1, name: "", success: true, steps: 1, tags: [] },
            { index: 1, seed: 2, name: "", success: true, steps: 1, tags: [] },
          ],
        },
      ],
    };
    expect(episodeIndex(catalog as never, "baseline", 2)).toBe(1);
    expect(episodeIndex(catalog as never, "baseline", 9)).toBeNull();
  });

  it("rewrites relative image paths to the docs endpoint", () => {
    expect(docAssetUrl("image/a.svg")).toBe("/studio/docs/image/a.svg");
    expect(docAssetUrl("https://x/a.png")).toBe("https://x/a.png");
  });
});
