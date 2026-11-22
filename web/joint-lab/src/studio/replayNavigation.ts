// Playback reports use synthetic hash events. Only browser navigation is a seek command.
export function navigationSeek(
  event: Pick<HashChangeEvent, "isTrusted" | "oldURL" | "newURL">,
  source: string,
) {
  if (!event.isTrusted || !event.oldURL || !event.newURL) return null;
  const parse = (url: string) => {
    const [page, query = ""] = new URL(url).hash
      .replace(/^#\/?/, "")
      .split("?");
    return { page, query: new URLSearchParams(query) };
  };
  const before = parse(event.oldURL),
    after = parse(event.newURL);
  if (
    after.page !== "replay" ||
    (after.query.get("source") || source) !== source
  )
    return null;
  if (
    before.query.get("episode") === after.query.get("episode") &&
    before.query.get("step") === after.query.get("step")
  )
    return null;
  const episode = Number(after.query.get("episode") || 0),
    step = Number(after.query.get("step") || 0);
  if (
    !Number.isInteger(episode) ||
    !Number.isInteger(step) ||
    episode < 0 ||
    step < 0
  )
    return null;
  return { episode, step };
}
