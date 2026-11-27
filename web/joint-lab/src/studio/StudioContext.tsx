import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";
import type { ReactNode } from "react";
import { useLocalStorage } from "@mantine/hooks";
import { useMantineColorScheme } from "@mantine/core";
import { api } from "../api";
import type {
  Catalog,
  Health,
  Meta,
  Model,
  Pose,
  Run,
  Scene,
  Snapshot,
  Vec3,
} from "../types";
import { JOINTS } from "../types";
import { constrainBowl } from "../workspace";
import { DEFAULT_SETTINGS, parseSettings } from "./settings";
import type { Settings } from "./settings";
import { navigate, useRoute } from "./router";
import { clampSize, fitsBowl, NEW_OBJECT_SIZE, parseBowl, parseScene, request } from "./scene";
const DEFAULT_POSE: Pose = {
  pan: 0,
  lift: 0,
  elbow: 0,
  wflex: 0,
  wroll: 0,
  grip: 1.2,
  bx: 0.15,
  by: 0,
  byaw: 0,
  wx: -0.15,
  wy: 0,
};
function useStudioController() {
  const [meta, setMeta] = useState<Meta | null>(null),
    [model, setModel] = useState<Model | null>(null),
    [scene, setScene] = useState<Scene | null>(null);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null),
    [selected, setSelected] = useState("");
  const [settings, setSettings] = useLocalStorage<Settings>({
    key: "sim-studio-settings-v1",
    defaultValue: DEFAULT_SETTINGS,
    deserialize: (value) => {
      try {
        return parseSettings(JSON.parse(value || "null"));
      } catch {
        return DEFAULT_SETTINGS;
      }
    },
  });
  const visible = settings.panels;
  const route = useRoute();
  const { setColorScheme } = useMantineColorScheme();
  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const requestedRun = useRef(false);
  const [catalog, setCatalog] = useState<Catalog>({ sources: [], runs: [] });
  const [health, setHealth] = useState<Health | null>(null);
  const [connected, setConnected] = useState(false);
  const refreshCatalog = useCallback(async () => {
    const data = await api<Catalog>("/studio/catalog");
    setCatalog(data);
  }, []);
  useEffect(() => {
    setColorScheme(settings.theme);
  }, [settings.theme, setColorScheme]);
  useEffect(() => {
    if (!route.id) navigate(settingsRef.current.startPage, {}, true);
  }, [route.id]);
  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const data = await api<Health>("/studio/health");
        if (active) {
          setHealth(data);
          setConnected(true);
        }
      } catch {
        if (active) setConnected(false);
      } finally {
        if (active) timer = setTimeout(poll, 10000);
      }
    };
    poll();
    refreshCatalog().catch(() => {});
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [refreshCatalog]);
  const [error, setError] = useState(""),
    [notice, setNotice] = useState(""),
    [previewing, setPreviewing] = useState(false),
    [ikBusy, setIkBusy] = useState(false);
  const [run, setRun] = useState<Run>({ status: "idle" }),
    [submitting, setSubmitting] = useState(false);
  const sceneRef = useRef(scene),
    snapshotRef = useRef(snapshot),
    ikTarget = useRef<Vec3 | null>(null),
    ikRunning = useRef(false),
    lastReplay = useRef("");
  sceneRef.current = scene;
  snapshotRef.current = snapshot;
  useEffect(() => {
    const ctrl = new AbortController();
    Promise.all([
      api<Meta>("/meta", undefined, ctrl.signal),
      api<Model>("/model?exact=1", undefined, ctrl.signal),
    ])
      .then(([m, robot]) => {
        const saved = m.saved_pose,
          p = { ...DEFAULT_POSE, ...m.home, ...saved };
        JOINTS.forEach(([key], i) => {
          p[key] = Math.max(m.limits[i][0], Math.min(m.limits[i][1], p[key]));
        });
        p.grip = Math.max(m.gclose, Math.min(m.gopen, p.grip));
        // cảnh lưu từ trước có thể có vật ngoài giới hạn hiện tại (vd hộp 6 cm): kẹp cỡ về SIZE_RANGE
        const objects = saved?.objects?.map((o) => ({ ...o, size: clampSize(o.size) })) || [
          {
            id: "object-1",
            type: "box" as const,
            color: "#ed8067",
            x: p.bx,
            y: p.by,
            yaw: p.byaw,
            size: NEW_OBJECT_SIZE,
          },
        ];
        setMeta(m);
        setModel(robot);
        setScene({
          pose: p,
          objects,
          target_id: saved?.target_id || objects[0].id,
          bowl: parseBowl(saved?.bowl),
          mode: saved?.mode,
        });
        setSelected(objects[0].id);
        try {
          const draft = localStorage.getItem("sim-studio-draft-v1");
          if (draft) {
            const restored = parseScene(JSON.parse(draft), m);
            setScene(restored);
            setSelected(restored.target_id);
          }
        } catch {}
      })
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message);
      });
    return () => ctrl.abort();
  }, []);
  useEffect(() => {
    if (!scene) return;
    const timer = setTimeout(() => {
      try {
        localStorage.setItem("sim-studio-draft-v1", JSON.stringify(scene));
      } catch {}
    }, 250);
    return () => clearTimeout(timer);
  }, [scene]);
  // Debounce edits, and ignore old replies so rapid dragging never rewinds the preview.
  useEffect(() => {
    if (!scene) return;
    let active = true;
    const ctrl = new AbortController();
    setPreviewing(true);
    const timer = setTimeout(
      () =>
        api<Snapshot>("/scene-pose", request(scene), ctrl.signal)
          .then((data) => {
            if (active) {
              setSnapshot(data);
              setError("");
            }
          })
          .catch((e) => {
            if (active && e.name !== "AbortError") setError(e.message);
          })
          .finally(() => {
            if (active) setPreviewing(false);
          }),
      90,
    );
    return () => {
      active = false;
      clearTimeout(timer);
      ctrl.abort();
    };
  }, [scene]);
  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const next = await api<Run>("/run-status");
        if (!active) return;
        if (next.status === "done" && next.run_id !== lastReplay.current) {
          lastReplay.current = next.run_id || "";
          void refreshCatalog().catch(() => {});
          if (requestedRun.current) {
            requestedRun.current = false;
            setNotice(next.blocked_reason || (next.total !== undefined
              ? `Gắp tất cả · ${next.placed}/${next.total} vật trong bát`
              : next.success ? "Gắp thành công." : "Đã kết thúc mô phỏng. Xem Replay để kiểm tra lần gắp."));
            if (settingsRef.current.autoOpenReplay)
              navigate("replay", { source: "lab" });
          }
        }
        setRun(next);
        if (next.status === "error")
          setError(
            next.error || "Mô phỏng gặp lỗi; hãy kiểm tra cảnh rồi chạy lại.",
          );
        if (next.status === "running") timer = setTimeout(poll, 650);
      } catch (e) {
        if (active) {
          setError((e as Error).message);
          timer = setTimeout(poll, 1500);
        }
      }
    };
    poll();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [run.status === "running"]);
  const solveIK = useCallback(async (target: Vec3) => {
    ikTarget.current = target;
    if (ikRunning.current) return;
    ikRunning.current = true;
    setIkBusy(true);
    try {
      while (ikTarget.current && sceneRef.current) {
        const point = ikTarget.current;
        ikTarget.current = null;
        const p = request(sceneRef.current);
        const q = new URLSearchParams(
          Object.fromEntries(
            Object.entries({
              ...p,
              x: point[0],
              y: point[1],
              z: point[2],
              vertical: 0,
              w: 0,
              K: 0,
              free: "1,1,1,1,1",
            })
              .filter(([k]) => k !== "objects")
              .map(([k, v]) => [k, String(v)]),
          ),
        );
        const result = await api<{ angles: number[]; err_mm: number }>(
          "/ik?" + q,
        );
        setScene((prev) =>
          prev
            ? {
                ...prev,
                pose: {
                  ...prev.pose,
                  ...Object.fromEntries(
                    JOINTS.map(([key], i) => [key, result.angles[i]]),
                  ),
                },
              }
            : prev,
        );
        setNotice(
          `IK · sai số ${result.err_mm.toFixed(1)} mm${result.err_mm > 5 ? " · đích gần hoặc ngoài giới hạn tư thế" : ""}`,
        );
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      ikRunning.current = false;
      setIkBusy(false);
    }
  }, []);
  const move = useCallback(
    (id: string, x: number, y: number, z?: number) => {
      if (
        !meta ||
        !Number.isFinite(x) ||
        !Number.isFinite(y) ||
        (z !== undefined && !Number.isFinite(z))
      )
        return;
      if (id === "arm") {
        const ee = snapshotRef.current?.ee;
        if (ee) solveIK([x, y, z ?? ee[2]]);
        return;
      }
      setScene((prev) => {
        if (!prev) return prev;
        if (id === "bowl") {
          const [wx, wy] = constrainBowl(x, y, meta.pick_region);
          return { ...prev, pose: { ...prev.pose, wx, wy } };
        }
        return {
          ...prev,
          objects: prev.objects.map((obj) =>
            obj.id === id
              ? {
                  ...obj,
                  x: Math.max(-0.435, Math.min(0.435, x)),
                  y: Math.max(-0.385, Math.min(0.385, y)),
                }
              : obj,
          ),
        };
      });
    },
    [meta, solveIK],
  );
  function setJoint(key: keyof Pose, value: number) {
    if (!Number.isFinite(value)) return;
    setScene((prev) =>
      prev ? { ...prev, pose: { ...prev.pose, [key]: value } } : prev,
    );
  }
  async function runScene(candidate: Scene) {
    setError("");
    setNotice("");
    if (candidate.mode === "all" && !fitsBowl(candidate.objects, candidate.bowl)) {
      setError("Bát chưa đủ chỗ cho tất cả vật. Hãy tăng bán kính bát hoặc giảm cỡ hay số vật.");
      return;
    }
    setSubmitting(true);
    requestedRun.current = true;
    try {
      const next = await api<Run>("/run", { ...request(candidate), mode: candidate.mode });
      setRun(next);
    } catch (e) {
      requestedRun.current = false;
      setError((e as Error).message);
      setRun({ status: "error" });
    } finally {
      setSubmitting(false);
    }
  }
  async function start() {
    if (scene) await runScene(scene);
  }
  async function loadCase(source: string, index = 0) {
    if (!meta) throw new Error("Mô hình chưa sẵn sàng");
    const data = await api<{ pose: unknown }>(
      "/studio/case?" + new URLSearchParams({ source, episode: String(index) }),
    );
    const next = parseScene(data.pose, meta);
    setScene(next);
    setSelected(next.target_id);
    navigate("lab");
  }
  async function rerun(source: string, index = 0) {
    if (!meta) throw new Error("Mô hình chưa sẵn sàng");
    const data = await api<{ pose: unknown }>(
      "/studio/case?" + new URLSearchParams({ source, episode: String(index) }),
    );
    await runScene(parseScene(data.pose, meta));
  }
  return {
    meta,
    model,
    scene,
    setScene,
    snapshot,
    selected,
    setSelected,
    settings,
    setSettings,
    visible,
    error,
    setError,
    notice,
    setNotice,
    previewing,
    ikBusy,
    run,
    submitting,
    running: run.status === "running" || submitting,
    route,
    move,
    setJoint,
    start,
    runScene,
    loadCase,
    rerun,
    catalog,
    refreshCatalog,
    health,
    connected,
  };
}
type Studio = ReturnType<typeof useStudioController>;
const StudioContext = createContext<Studio | null>(null);
export function StudioProvider({ children }: { children: ReactNode }) {
  const value = useStudioController();
  return (
    <StudioContext.Provider value={value}>{children}</StudioContext.Provider>
  );
}
export function useStudio() {
  const value = useContext(StudioContext);
  if (!value) throw new Error("StudioProvider is missing");
  return value;
}
