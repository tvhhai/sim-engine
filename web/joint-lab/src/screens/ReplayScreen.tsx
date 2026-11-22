import { useEffect, useMemo, useRef, useState } from "react";
import { useComputedColorScheme } from "@mantine/core";
import {
  ActionIcon,
  Collapse,
  Alert,
  Badge,
  Button,
  Group,
  Loader,
  Modal,
  Select,
  SegmentedControl,
  Stack,
  Text,
  TextInput,
  Tooltip,
} from "@mantine/core";
import {
  Bookmark,
  Download,
  Expand,
  FlaskConical,
  RefreshCw,
  Trash2,
} from "lucide-react";
import { useLocalStorage } from "@mantine/hooks";
import { useStudio } from "../studio/StudioContext";
import type { ReplayViewState } from "../types";
import { navigationSeek } from "../studio/replayNavigation";
import { ReplayWebGL } from "../components/ReplayWebGL";
import { api } from "../api";
import { filterEpisodes, RESULT_FILTERS, ZONE_FILTERS } from "../studio/replayFilters";

export default function ReplayScreen() {
  const {
    route,
    catalog,
    refreshCatalog,
    model,
    settings,
    loadCase,
    setError,
    run,
  } = useStudio();
  const [version, setVersion] = useState(0),
    [ready, setReady] = useState(false),
    [dataError, setDataError] = useState("");
  const [portalIdentity, setPortalIdentity] = useState("");
  const [portalDocument, setPortalDocument] = useState<Document | null>(null);
  const [viewState, setViewState] = useState<ReplayViewState | null>(null);
  const [bookmarksOpen, setBookmarksOpen] = useState(false);
  // Bộ lọc danh sách episode (giống replay canvas): kết quả và vùng của vị trí hộp, lưu trên trình duyệt
  const [resultFilter, setResultFilter] = useLocalStorage({ key: "sim-studio-replay-result", defaultValue: "all" }),
    [zoneFilter, setZoneFilter] = useLocalStorage({ key: "sim-studio-replay-zone", defaultValue: "all" });
  const [confirmDelete, setConfirmDelete] = useState(false),
    [deleting, setDeleting] = useState(false);
  const [note, setNote] = useState(""),
    [marks, setMarks] = useLocalStorage<
      Record<string, { step: number; label: string }[]>
    >({ key: "sim-studio-bookmarks-v1", defaultValue: {} });
  const frame = useRef<HTMLIFrameElement>(null),
    pendingNativeSeeks = useRef(new Set<string>()),
    colorScheme = useComputedColorScheme("dark");
  const source =
    route.params.get("source") ||
    catalog.sources.find((item) => item.id === "baseline")?.id ||
    catalog.sources[0]?.id ||
    "lab";
  const episode = Math.max(0, Number(route.params.get("episode")) || 0),
    step = Math.max(0, Number(route.params.get("step")) || 0);
  const mode = route.params.get("view") === "3d" ? "3d" : "analysis";
  const selectedSource = catalog.sources.find((item) => item.id === source);
  const runRecord = source.startsWith("run:")
    ? catalog.runs.find((item) => item.id === source.slice(4))
    : null;
  const summary = selectedSource?.episodes[episode] || runRecord;
  const shownEpisodes = selectedSource
    ? filterEpisodes(selectedSource.episodes, resultFilter, zoneFilter, episode)
    : [];
  const deletable = source === "lab" || source.startsWith("run:");
  async function deleteReplay() {
    setDeleting(true);
    try {
      await api(source === "lab" ? "/studio/lab/delete" : `/studio/runs/${source.slice(4)}/delete`, {});
      setConfirmDelete(false);
      await refreshCatalog();
      route.navigate("replay", { source: "baseline", episode: 0, step: 0, view: mode }, true);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setDeleting(false);
    }
  }
  const sourceOptions = [
    ...catalog.sources.map((item) => ({ value: item.id, label: item.title })),
    ...catalog.runs.map((item) => ({
      value: "run:" + item.id,
      label: item.title,
    })),
  ];
  const change = (
    values: Record<string, string | number | undefined>,
    replace = false,
  ) => {
    const next = { source, episode, step, view: mode, ...values };
    if (
      ready &&
      next.source === source &&
      (values.episode !== undefined || values.step !== undefined)
    ) {
      pendingNativeSeeks.current.add(`${source}/${next.episode}/${next.step}`);
      frame.current?.contentWindow?.postMessage(
        {
          type: "studio-replay-command",
          episode: next.episode,
          step: next.step,
        },
        location.origin,
      );
    }
    route.navigate("replay", next, replace);
  };
  // Only changing source/reloading replaces the document. Scrubbing is synchronized by messages.
  const iframeUrl = useMemo(
    () =>
      "/studio/replay-view?" +
      new URLSearchParams({ source, revision: String(version) }),
    [source, version],
  );
  useEffect(() => {
    setReady(false);
    setPortalDocument(null);
    setViewState(null);
    pendingNativeSeeks.current.clear();
    setDataError("");
  }, [source, version]);
  const commandRef = useRef({
    source,
    episode,
    step,
    speed: settings.defaultSpeed,
    theme: colorScheme,
    autoPlay: settings.autoPlay,
    mode,
  });
  commandRef.current = {
    source,
    episode,
    step,
    speed: settings.defaultSpeed,
    theme: colorScheme,
    autoPlay: settings.autoPlay,
    mode,
  };
  useEffect(() => {
    const listener = (event: MessageEvent) => {
      if (
        event.origin !== location.origin ||
        event.source !== frame.current?.contentWindow ||
        event.data?.source !== source
      )
        return;
      if (event.data.type === "studio-replay-ready") {
        setReady(true);
        setPortalIdentity(iframeUrl);
        setPortalDocument(frame.current?.contentDocument || null);
        const c = commandRef.current;
        frame.current?.contentWindow?.postMessage(
          {
            type: "studio-replay-command",
            episode: c.episode,
            step: c.step,
            speed: c.speed,
            theme: c.theme,
            play: c.autoPlay,
            focus3d: c.mode === "3d",
          },
          location.origin,
        );
      }
      if (event.data.type === "studio-replay-scene")
        setViewState(event.data.scene);
      if (
        event.data.type === "studio-replay-state" &&
        Number.isInteger(event.data.step) &&
        Number.isInteger(event.data.episode)
      ) {
        const c = commandRef.current;
        if (event.data.episode !== c.episode || event.data.step !== c.step)
          route.navigate(
            "replay",
            {
              source,
              episode: event.data.episode,
              step: event.data.step,
              view: c.mode,
            },
            true,
          );
      }
    };
    window.addEventListener("message", listener);
    return () => window.removeEventListener("message", listener);
  }, [source, route.navigate, iframeUrl]);
  useEffect(() => {
    const listener = (event: HashChangeEvent) => {
      if (!ready) return;
      const seek = navigationSeek(event, source);
      if (!seek) return;
      const key = `${source}/${seek.episode}/${seek.step}`;
      if (pendingNativeSeeks.current.delete(key)) return;
      frame.current?.contentWindow?.postMessage(
        { type: "studio-replay-command", ...seek },
        location.origin,
      );
    };
    window.addEventListener("hashchange", listener);
    return () => window.removeEventListener("hashchange", listener);
  }, [ready, source]);
  useEffect(() => {
    if (ready)
      frame.current?.contentWindow?.postMessage(
        {
          type: "studio-replay-command",
          speed: settings.defaultSpeed,
          theme: colorScheme,
        },
        location.origin,
      );
  }, [ready, settings.defaultSpeed, colorScheme]);
  useEffect(() => {
    if (ready)
      frame.current?.contentWindow?.postMessage(
        { type: "studio-replay-command", focus3d: mode === "3d" },
        location.origin,
      );
  }, [ready, mode]);
  useEffect(() => {
    if (source === "lab" && run.status === "done") {
      setVersion((v) => v + 1);
    }
  }, [run.run_id, run.status, source]);
  const key = `${source === "lab" ? "lab:" + (catalog.runs[0]?.id || "latest") : source}/${episode}`,
    currentMarks = marks[key] || [];
  const runAction = async () => {
    try {
      await loadCase(source, episode);
    } catch (e) {
      setError((e as Error).message);
    }
  };
  if (!sourceOptions.length)
    return (
      <div className="screen-empty">
        <Bookmark size={32} />
        <Text fw={600}>Chưa có dữ liệu replay</Text>
        <Text c="dimmed">Tạo một lần chạy trong Lab để bắt đầu.</Text>
        <Button component="a" href="#/lab">
          Mở Lab
        </Button>
      </div>
    );
  return (
    <div className="replay-screen">
      <div className="screen-toolbar replay-toolbar">
        <Group align="end" gap="sm" className="replay-selectors">
          <Select
            label="Nguồn replay"
            aria-label="Nguồn replay"
            searchable
            w={270}
            value={source}
            onChange={(value) => {
              if (value) change({ source: value, episode: 0, step: 0 });
            }}
            data={sourceOptions}
            allowDeselect={false}
          />
          {selectedSource && selectedSource.episodes.length > 1 && (
            <>
              <Select
                label="Kết quả"
                aria-label="Lọc theo kết quả"
                w={140}
                value={resultFilter}
                onChange={(value) => value && setResultFilter(value)}
                data={RESULT_FILTERS}
                allowDeselect={false}
              />
              <Select
                label="Vùng"
                aria-label="Lọc theo vùng của hộp"
                w={160}
                value={zoneFilter}
                onChange={(value) => value && setZoneFilter(value)}
                data={ZONE_FILTERS}
                allowDeselect={false}
              />
              <Select
                label={`Episode / seed (${shownEpisodes.filter((e) => e.matches).length}/${selectedSource.episodes.length})`}
                aria-label="Episode / seed"
                searchable
                w={240}
                value={String(episode)}
                onChange={(value) => change({ episode: Number(value), step: 0 })}
                data={shownEpisodes.map((item) => ({
                  value: String(item.index),
                  label: item.matches ? item.name : `${item.name} (ngoài bộ lọc)`,
                }))}
                nothingFoundMessage="Không có episode khớp bộ lọc"
                allowDeselect={false}
              />
            </>
          )}
          <SegmentedControl
            aria-label="Chế độ xem replay"
            value={mode}
            onChange={(value) => change({ view: value })}
            data={[
              { value: "analysis", label: "Phân tích đầy đủ" },
              {
                value: "3d",
                label: "Chỉ 3D",
              },
            ]}
          />
        </Group>
        <Group gap="xs">
          <Tooltip label="Đánh dấu và ghi chú bước">
            <ActionIcon
              aria-label="Mở đánh dấu bước"
              aria-expanded={bookmarksOpen}
              variant={bookmarksOpen ? "light" : "default"}
              size="lg"
              onClick={() => setBookmarksOpen(!bookmarksOpen)}
            >
              <Bookmark size={17} />
            </ActionIcon>
          </Tooltip>
          <Tooltip label="Nạp cảnh ban đầu về Lab">
            <ActionIcon
              aria-label="Nạp cảnh về Lab"
              variant="default"
              size="lg"
              onClick={runAction}
            >
              <FlaskConical size={17} />
            </ActionIcon>
          </Tooltip>
          <Tooltip label="Xuất JSON của episode">
            <ActionIcon
              component="a"
              href={
                "/studio/replay-data?" +
                new URLSearchParams({ source, episode: String(episode) })
              }
              download={`replay-${source.replace(":", "-")}.json`}
              aria-label="Xuất dữ liệu replay"
              variant="default"
              size="lg"
            >
              <Download size={17} />
            </ActionIcon>
          </Tooltip>
          <Tooltip label="Làm mới kết quả">
            <ActionIcon
              aria-label="Làm mới replay"
              variant="default"
              size="lg"
              onClick={() => {
                setVersion((v) => v + 1);
                refreshCatalog().catch((e) => setError(e.message));
              }}
            >
              <RefreshCw size={17} />
            </ActionIcon>
          </Tooltip>
          {deletable && (
            <Tooltip label="Xoá replay này">
              <ActionIcon
                aria-label="Xoá replay này"
                variant="default"
                color="red"
                size="lg"
                onClick={() => setConfirmDelete(true)}
              >
                <Trash2 size={17} />
              </ActionIcon>
            </Tooltip>
          )}
          <Tooltip label="Toàn màn hình replay">
            <ActionIcon
              aria-label="Toàn màn hình replay"
              variant="default"
              size="lg"
              onClick={() =>
                frame.current
                  ?.requestFullscreen()
                  .catch((e) => setError(e.message))
              }
            >
              <Expand size={17} />
            </ActionIcon>
          </Tooltip>
        </Group>
      </div>
      <Modal opened={confirmDelete} onClose={() => setConfirmDelete(false)} title="Xoá replay?" centered>
        <Stack>
          <Text size="sm">
            {source === "lab"
              ? "Xoá replay của lần chạy Lab gần nhất. Cảnh đã đặt trong Lab vẫn giữ nguyên."
              : `Xoá hẳn "${runRecord?.title ?? source}" khỏi lịch sử (cả cảnh và dữ liệu replay).`}{" "}
            Không hoàn tác được.
          </Text>
          <Group justify="flex-end">
            <Button variant="default" onClick={() => setConfirmDelete(false)}>
              Huỷ
            </Button>
            <Button color="red" loading={deleting} onClick={deleteReplay}>
              Xoá
            </Button>
          </Group>
        </Stack>
      </Modal>
      <div className="replay-summary">
        <Group gap="xs">
          <Badge
            color={
              summary?.blocked_reason
                ? "orange"
                : summary?.success
                  ? "teal"
                  : "red"
            }
            variant="light"
          >
            {summary?.blocked_reason
              ? "Đứng yên"
              : summary?.success
                ? "Thành công"
                : "Thất bại"}
          </Badge>
          <Text size="sm">{summary?.steps ?? "—"} bước</Text>
        </Group>
        <Text size="xs" c="dimmed">
          Đang xem dữ liệu đã ghi · không chạy mô phỏng
        </Text>
      </div>
      <Collapse in={bookmarksOpen}>
        <div className="replay-bookmarks">
          <TextInput
            aria-label="Tên đánh dấu"
            placeholder="Ghi chú bước hiện tại…"
            value={note}
            onChange={(e) => setNote(e.currentTarget.value)}
            size="xs"
            w={210}
          />
          <Button
            size="xs"
            variant="default"
            leftSection={<Bookmark size={14} />}
            onClick={() => {
              setMarks({
                ...marks,
                [key]: [
                  ...currentMarks.filter((mark) => mark.step !== step),
                  { step, label: note.trim() || `Bước ${step}` },
                ].sort((a, b) => a.step - b.step),
              });
              setNote("");
            }}
          >
            Đánh dấu bước {step}
          </Button>
          {currentMarks.map((mark) => (
            <Group gap={2} key={mark.step}>
              <Button
                size="compact-xs"
                variant="light"
                color="gray"
                onClick={() => change({ step: mark.step })}
              >
                {mark.label}
              </Button>
              <ActionIcon
                aria-label={`Bỏ đánh dấu bước ${mark.step}`}
                variant="subtle"
                color="gray"
                size="xs"
                onClick={() =>
                  setMarks({
                    ...marks,
                    [key]: currentMarks.filter(
                      (item) => item.step !== mark.step,
                    ),
                  })
                }
              >
                <Trash2 size={11} />
              </ActionIcon>
            </Group>
          ))}
        </div>
      </Collapse>
      {
        <div className="full-replay-wrap">
          {!ready && (
            <div className="iframe-loading">
              <Loader size="sm" color="teal" />
              <Text size="sm">Đang nạp replay đầy đủ…</Text>
            </div>
          )}
          {portalIdentity === iframeUrl &&
            portalDocument &&
            viewState &&
            model && (
              <ReplayWebGL
                document={portalDocument}
                scene={viewState}
                model={model}
              />
            )}
          <iframe
            ref={frame}
            src={iframeUrl}
            title="Replay phân tích đầy đủ"
            allowFullScreen
            className="full-replay"
            onLoad={() => {
              const document = frame.current?.contentDocument;
              if (document && !document.getElementById("stepCount")) {
                setDataError(
                  document.body.textContent || "Không mở được replay",
                );
                setReady(true);
              }
            }}
          />
          {dataError && (
            <Alert color="red" className="replay-load-error">
              {dataError}
            </Alert>
          )}
        </div>
      }
    </div>
  );
}
