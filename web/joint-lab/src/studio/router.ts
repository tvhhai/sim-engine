import { useMemo, useSyncExternalStore } from "react";
const subscribe = (callback: () => void) => {
  window.addEventListener("hashchange", callback);
  return () => window.removeEventListener("hashchange", callback);
};
const snapshot = () => window.location.hash;
export function navigate(
  screen: string,
  params: Record<string, string | number | undefined> = {},
  replace = false,
) {
  const q = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined) q.set(key, String(value));
  });
  const hash = `#/${screen}${q.size ? "?" + q : ""}`;
  if (location.hash === hash) return;
  if (replace) {
    history.replaceState(null, "", hash);
    window.dispatchEvent(new HashChangeEvent("hashchange"));
  } else location.hash = hash;
}
export function useRoute() {
  const hash = useSyncExternalStore(subscribe, snapshot);
  return useMemo(() => {
    const [id, search = ""] = hash.replace(/^#\/?/, "").split("?");
    return { id, params: new URLSearchParams(search), navigate };
  }, [hash]);
}
