import { useEffect, useState } from "react";
import {
  Alert,
  Badge,
  Button,
  Group,
  Select,
  Slider,
  Stack,
  Text,
} from "@mantine/core";
import { Pause, Play, SkipBack, SkipForward } from "lucide-react";
import type { Model, Replay } from "../types";
import { Panel } from "./Panel";
import { SceneView } from "./LazySceneView";
export function ReplayPanel({
  data,
  model,
  initialStep = 0,
  defaultSpeed = "1",
  autoPlay = false,
  onFrameChange,
}: {
  data: Replay | null;
  model: Model;
  initialStep?: number;
  defaultSpeed?: string;
  autoPlay?: boolean;
  onFrameChange?: (step: number) => void;
}) {
  const [index, setIndex] = useState(initialStep),
    [playing, setPlaying] = useState(false),
    [speed, setSpeed] = useState<string | null>(defaultSpeed);
  const episode = data?.episodes[0],
    frames = episode?.frames || [];
  useEffect(() => {
    setIndex(Math.min(initialStep, Math.max(0, frames.length - 1)));
    setPlaying(autoPlay);
  }, [data]);
  useEffect(() => {
    setIndex(Math.min(initialStep, Math.max(0, frames.length - 1)));
  }, [initialStep, frames.length]);
  useEffect(() => {
    onFrameChange?.(index);
  }, [index]);
  useEffect(() => {
    if (!playing || !data || index >= frames.length - 1) return;
    const timer = setTimeout(
      () => setIndex((n) => n + 1),
      (data.scene.step_dt * 1000) / Number(speed),
    );
    return () => clearTimeout(timer);
  }, [playing, index, speed, data, frames.length]);
  const frame = frames[Math.min(index, frames.length - 1)];
  const isPlaying = playing && index < frames.length - 1;
  return (
    <Panel
      title="Replay"
      badge="Chỉ 1 seed lab"
      tools={
        episode && (
          <Badge
            color={
              episode.blocked_reason
                ? "orange"
                : episode.success
                  ? "teal"
                  : "red"
            }
            variant="light"
          >
            {episode.blocked_reason
              ? "Đứng yên"
              : episode.success
                ? "Thành công"
                : "Thất bại"}
          </Badge>
        )
      }
      className="replay-panel"
    >
      {!episode || !frame?.pose ? (
        <div className="empty-replay">
          <Play size={30} />
          <Text fw={500}>Cảnh sẵn sàng để chạy thử</Text>
          <Text c="dimmed" size="sm">
            Đặt vật thể, chỉnh tay rồi bấm Chạy mô phỏng. Kết quả mới sẽ xuất
            hiện ở đây.
          </Text>
        </div>
      ) : (
        <>
          {episode.blocked_reason && (
            <Alert color="orange" m="md" title="Không di chuyển cánh tay">
              {episode.blocked_reason}
            </Alert>
          )}
          <SceneView
            animate={isPlaying}
            model={model}
            poses={frame.pose}
            objects={frame.objects || []}
            bowl={episode.bowl}
            bowlGeoms={episode.bowl_geoms}
            bowlOrigin={episode.bowl}
            ee={frame.ee}
          />
          <Stack p="md" gap="md">
            <Group justify="space-between">
              <Group gap="xs">
                <Button
                  aria-label="Về đầu replay"
                  variant="default"
                  onClick={() => {
                    setIndex(0);
                    setPlaying(false);
                  }}
                >
                  <SkipBack size={16} />
                </Button>
                <Button
                  disabled={frames.length < 2}
                  leftSection={
                    isPlaying ? <Pause size={16} /> : <Play size={16} />
                  }
                  onClick={() => {
                    if (index >= frames.length - 1) setIndex(0);
                    setPlaying(!isPlaying);
                  }}
                >
                  {isPlaying ? "Tạm dừng" : "Phát replay"}
                </Button>
                <Button
                  aria-label="Đến cuối replay"
                  variant="default"
                  onClick={() => {
                    setIndex(frames.length - 1);
                    setPlaying(false);
                  }}
                >
                  <SkipForward size={16} />
                </Button>
              </Group>
              <Group>
                <Text size="sm" className="mono">
                  Bước {index} / {frames.length - 1} · phase {frame.ph}
                </Text>
                <Select
                  aria-label="Tốc độ replay"
                  w={90}
                  value={speed}
                  onChange={setSpeed}
                  data={["0.1", "0.25", "0.5", "1", "1.5", "2", "3", "4"].map(
                    (value) => ({
                      value,
                      label: `${value}×`,
                    }),
                  )}
                  allowDeselect={false}
                />
              </Group>
            </Group>
            <Slider
              aria-label="Bước replay"
              min={0}
              max={Math.max(1, frames.length - 1)}
              step={1}
              value={index}
              onChange={(v) => {
                setIndex(v);
                setPlaying(false);
              }}
              label={(v) => `Bước ${v}`}
              disabled={frames.length === 1}
            />
            <Group justify="space-between">
              <Text size="sm" c="dimmed">
                {(index * (data?.scene.step_dt || 0)).toFixed(2)} s ·{" "}
                {frames.length - 1} bước · {frame.objects?.length || 1} vật
              </Text>
              <Text size="xs" c="dimmed">
                Phát lại dữ liệu đã ghi từ MuJoCo
              </Text>
            </Group>
          </Stack>
        </>
      )}
    </Panel>
  );
}
