import { describe, it, expect } from "vitest";
import { navigationSeek } from "./replayNavigation";
const url = (step: number, view = "analysis", source = "lab") =>
  `http://127.0.0.1:8765/lab/#/replay?source=${source}&episode=0&step=${step}&view=${view}`;
describe("Replay navigation does not feed playback reports back as seeks", () => {
  it("ignores synthetic URL updates from the playback engine", () => {
    expect(
      navigationSeek(
        { isTrusted: false, oldURL: url(1), newURL: url(2) },
        "lab",
      ),
    ).toBeNull();
  });
  it("does not seek when changing the view mode during playback", () => {
    expect(
      navigationSeek(
        { isTrusted: true, oldURL: url(30), newURL: url(30, "3d") },
        "lab",
      ),
    ).toBeNull();
  });
  it("honors browser back/forward and explicit step navigation", () => {
    expect(
      navigationSeek(
        { isTrusted: true, oldURL: url(40), newURL: url(10) },
        "lab",
      ),
    ).toEqual({ episode: 0, step: 10 });
  });
  it("leaves changing the data source to the normal document loader", () => {
    expect(
      navigationSeek(
        {
          isTrusted: true,
          oldURL: url(40),
          newURL: url(0, "analysis", "baseline"),
        },
        "lab",
      ),
    ).toBeNull();
  });
  it("rejects invalid steps and unrelated pages", () => {
    expect(
      navigationSeek(
        { isTrusted: true, oldURL: url(0), newURL: url(-4) },
        "lab",
      ),
    ).toBeNull();
    expect(
      navigationSeek(
        {
          isTrusted: true,
          oldURL: url(0),
          newURL: "http://127.0.0.1:8765/lab/#/lab",
        },
        "lab",
      ),
    ).toBeNull();
  });
});
