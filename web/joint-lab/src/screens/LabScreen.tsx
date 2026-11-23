import {
  ActionIcon,
  Alert,
  Badge,
  Button,
  ColorInput,
  FileButton,
  Group,
  Loader,
  NumberInput,
  Select,
  Slider,
  Stack,
  Tabs,
  Text,
  Title,
} from "@mantine/core";
import { useLocalStorage } from "@mantine/hooks";
import {
  RotateCcw,
  SlidersHorizontal,
  Download,
  Upload,
  MonitorPlay,
  Box,
  Map as MapIcon,
} from "lucide-react";
import { JOINTS } from "../types";
import { reachable } from "../workspace";
import { Panel } from "../components/Panel";
import { TopView } from "../components/TopView";
import { SceneView } from "../components/LazySceneView";
import { ObjectPanel } from "../components/ObjectPanel";
import { useStudio } from "../studio/StudioContext";
import { bowlGeoms, downloadJson, parseScene } from "../studio/scene";
import { toBase } from "../studio/coordinates";
export default function LabScreen() {
  const {
    scene,
    setScene,
    meta,
    model,
    snapshot,
    selected,
    setSelected,
    visible,
    move,
    setJoint,
    notice,
    previewing,
    error,
    setError,
    route,
    start,
    running,
    ikBusy,
    settings,
  } = useStudio();
  const [labTab, setLabTab] = useLocalStorage({ key: "sim-studio-lab-tab", defaultValue: "scene" });
  if (!scene || !meta || !model)
    return (
      <div className="screen-empty">
        <Loader color="teal" />
        <Title order={2}>Đang nạp Lab…</Title>
        <Text c="dimmed">{error || "Đang nạp mô hình và cảnh đã lưu."}</Text>
      </div>
    );
  const target = scene.objects.find((o) => o.id === scene.target_id)!;
  const bowlOK = reachable(scene.pose.wx, scene.pose.wy, meta.pick_region),
    targetOK = reachable(target.x, target.y, meta.pick_region);
  const showScene = visible.map;
  // Vùng gắp xanh / hỗn hợp / đỏ và điểm quét, cùng dữ liệu với replay (server /meta)
  const zones = {
    region: meta.pick_region,
    stop: meta.unreachable_region ?? null,
    scan: meta.pick_scan ?? [],
  };
  return (
    <>
      <div className="screen-toolbar">
        <div>
          <Text fw={600}>
            Cảnh thử nghiệm{" "}
            <Badge ml="xs" variant="light">
              {scene.objects.length} / 5 vật
            </Badge>
          </Text>
          <Text size="sm" c="dimmed">
            Tọa độ theo đế: XY tại trục khớp 1, Z = 0 trên mặt bàn. Mũi tên ở
            vật hoặc đầu tay dùng để kéo.
          </Text>
        </div>
        <Group gap="xs">
          <Button
            variant="default"
            leftSection={<Download size={16} />}
            onClick={() =>
              downloadJson("lab-scene.json", { schema_version: 1, ...scene })
            }
          >
            Xuất cảnh
          </Button>
          <FileButton
            accept="application/json,.json"
            onChange={async (file) => {
              if (!file) return;
              try {
                setScene(parseScene(JSON.parse(await file.text()), meta));
              } catch (e) {
                setError((e as Error).message);
              }
            }}
          >
            {(props) => (
              <Button
                {...props}
                variant="default"
                leftSection={<Upload size={16} />}
              >
                Nạp cảnh
              </Button>
            )}
          </FileButton>
          <Button
            leftSection={<MonitorPlay size={16} />}
            loading={running}
            disabled={previewing || ikBusy}
            onClick={() => start()}
          >
            Chạy mô phỏng
          </Button>
        </Group>
      </div>
      {(!bowlOK || !targetOK) && (
        <Alert color="orange" mx="lg" mt="md" title="Cánh tay sẽ đứng yên">
          {!bowlOK ? "Bát" : "Vật cần gắp"} đang ngoài vùng thao tác. Bạn có thể
          chạy để ghi lại trường hợp này.
        </Alert>
      )}
      <div className="workspace lab-workspace">
        <div className="main-column">
          {/* Đặt cảnh (XY) là việc chính của Lab nên đứng trước; 3D là tab kết quả để xem tư thế và va chạm */}
          <Tabs
            value={showScene ? labTab : "result"}
            onChange={(v) => v && setLabTab(v)}
            keepMounted={false}
            className="lab-tabs"
          >
            <Tabs.List>
              {showScene && (
                <Tabs.Tab value="scene" leftSection={<MapIcon size={15} />}>
                  Đặt cảnh · mặt bàn
                </Tabs.Tab>
              )}
              <Tabs.Tab value="result" leftSection={<Box size={15} />}>
                Kết quả 3D
              </Tabs.Tab>
            </Tabs.List>
            {showScene && (
              <Tabs.Panel value="scene" pt="sm">
                <Panel title="Tọa độ XY" badge="Kéo hộp, bóng, bát">
                  <TopView
                    scene={scene}
                    region={meta.pick_region}
                    zones={zones}
                    snapshot={snapshot}
                    selected={selected}
                    onSelect={setSelected}
                    onMove={move}
                  />
                </Panel>
              </Tabs.Panel>
            )}
            <Tabs.Panel value="result" pt="sm">
              <Panel
                title="Không gian 3D"
                badge="Lưới gốc MuJoCo"
                tools={
                  <Select
                    aria-label="Chọn đối tượng điều chỉnh trong 3D"
                    w={155}
                    size="xs"
                    value={selected}
                    onChange={(v) => setSelected(v || "")}
                    data={[
                      { value: "", label: "Xoay góc nhìn" },
                      { value: "arm", label: "Đầu tay" },
                      { value: "bowl", label: "Bát" },
                      ...scene.objects.map((o, i) => ({
                        value: o.id,
                        label: `${o.type === "sphere" ? "Bóng" : "Hộp"} ${i + 1}`,
                      })),
                    ]}
                  />
                }
              >
                {snapshot ? (
                  <SceneView
                    origin={[...meta.pick_region.pan, 0.42]}
                    region={meta.pick_region}
                    zones={zones}
                    model={model}
                    poses={snapshot.pose}
                    objects={scene.objects}
                    bowl={[scene.pose.wx, scene.pose.wy, 0.42]}
                    bowlGeoms={bowlGeoms(scene.bowl)}
                    bowlOrigin={[0, 0, 0]}
                    bowlColor={scene.bowl.color}
                    ee={snapshot.ee}
                    selected={selected}
                    onSelect={setSelected}
                    onMove={(id, p) => move(id, p[0], p[1], p[2])}
                  />
                ) : (
                  <div className="scene-loading">
                    <Loader color="teal" />
                  </div>
                )}
              </Panel>
            </Tabs.Panel>
          </Tabs>
        </div>
        <aside className="controls-column">
          {visible.objects && (
            <ObjectPanel
              scene={scene}
              selected={selected}
              region={meta.pick_region}
              onSelect={setSelected}
              onChange={setScene}
            />
          )}
          {visible.bowl && (
            <Panel title="Vị trí bát" badge="Giới hạn vùng">
              <Stack p="md" gap="sm">
                <Group grow>
                  <NumberInput
                    label="Bát X (cm)"
                    value={Number(
                      ((scene.pose.wx - meta.pick_region.pan[0]) * 100).toFixed(
                        1,
                      ),
                    )}
                    onChange={(v) =>
                      move(
                        "bowl",
                        Number(v) / 100 + meta.pick_region.pan[0],
                        scene.pose.wy,
                      )
                    }
                    step={0.5}
                  />
                  <NumberInput
                    label="Bát Y (cm)"
                    value={Number(
                      ((scene.pose.wy - meta.pick_region.pan[1]) * 100).toFixed(
                        1,
                      ),
                    )}
                    onChange={(v) =>
                      move(
                        "bowl",
                        scene.pose.wx,
                        Number(v) / 100 + meta.pick_region.pan[1],
                      )
                    }
                    step={0.5}
                  />
                </Group>
                <Group grow>
                  <NumberInput
                    label="Bán kính (cm)"
                    description="3 – 9"
                    value={Number((scene.bowl.radius * 100).toFixed(1))}
                    onChange={(v) =>
                      setScene({ ...scene, bowl: { ...scene.bowl, radius: Math.max(0.03, Math.min(0.09, Number(v) / 100 || 0.045)) } })
                    }
                    min={3}
                    max={9}
                    step={0.5}
                  />
                  <NumberInput
                    label="Cao (cm)"
                    description="2 – 8"
                    value={Number((scene.bowl.height * 100).toFixed(1))}
                    onChange={(v) =>
                      setScene({ ...scene, bowl: { ...scene.bowl, height: Math.max(0.02, Math.min(0.08, Number(v) / 100 || 0.04)) } })
                    }
                    min={2}
                    max={8}
                    step={0.5}
                  />
                </Group>
                <ColorInput
                  label="Màu bát"
                  value={scene.bowl.color}
                  onChange={(v) => /^#[0-9a-f]{6}$/i.test(v) && setScene({ ...scene, bowl: { ...scene.bowl, color: v } })}
                  format="hex"
                  swatches={["#3366d9", "#e5484d", "#30a46c", "#f5c542"]}
                />
                <Text size="xs" c="dimmed">
                  Bát dừng tại biên vùng thao tác. Hộp và bóng được đặt ngoài
                  vùng để test. Màu bát dùng cho các bài gắp theo màu sau này.
                </Text>
              </Stack>
            </Panel>
          )}
          {visible.joints && (
            <Panel
              title="Khớp và hàm kẹp"
              tools={
                <ActionIcon
                  variant="subtle"
                  color="gray"
                  aria-label="Đưa tay về home"
                  onClick={() =>
                    setScene({
                      ...scene,
                      pose: { ...scene.pose, ...meta.home },
                    })
                  }
                >
                  <RotateCcw size={16} />
                </ActionIcon>
              }
            >
              <Stack p="md" gap="lg">
                {JOINTS.map(([key, label], i) => (
                  <div key={key}>
                    <Group justify="space-between" mb={8}>
                      <Text size="sm">{label}</Text>
                      <NumberInput
                        aria-label={`${label} (độ)`}
                        w={90}
                        size="xs"
                        value={Number(scene.pose[key].toFixed(1))}
                        onChange={(v) =>
                          setJoint(
                            key,
                            Math.max(
                              meta.limits[i][0],
                              Math.min(meta.limits[i][1], Number(v)),
                            ),
                          )
                        }
                        min={meta.limits[i][0]}
                        max={meta.limits[i][1]}
                        step={0.5}
                      />
                    </Group>
                    <Slider
                      aria-label={label}
                      min={meta.limits[i][0]}
                      max={meta.limits[i][1]}
                      step={0.1}
                      value={scene.pose[key]}
                      onChange={(v) => setJoint(key, v)}
                      label={(v) => `${v.toFixed(1)}°`}
                    />
                  </div>
                ))}
                <div>
                  <Text size="sm" mb="sm">
                    Hàm kẹp · {((scene.pose.grip * 180) / Math.PI).toFixed(1)}°
                  </Text>
                  <Slider
                    aria-label="Độ mở hàm kẹp"
                    min={meta.gclose}
                    max={meta.gopen}
                    step={0.01}
                    value={scene.pose.grip}
                    onChange={(v) => setJoint("grip", v)}
                  />
                  <Group justify="space-between" mt="sm">
                    <Button
                      size="compact-xs"
                      variant="subtle"
                      color="gray"
                      onClick={() => setJoint("grip", meta.gclose)}
                    >
                      Đóng kẹp
                    </Button>
                    <Button
                      size="compact-xs"
                      variant="subtle"
                      color="gray"
                      onClick={() => setJoint("grip", meta.gopen)}
                    >
                      Mở kẹp
                    </Button>
                  </Group>
                </div>
                {notice && (
                  <Text size="xs" c="dimmed" aria-live="polite">
                    {notice}
                  </Text>
                )}
              </Stack>
            </Panel>
          )}
          {visible.metrics && snapshot && (
            <Panel
              title="Số liệu và va chạm"
              badge={previewing ? "Đang cập nhật" : "Tư thế tĩnh"}
            >
              <Stack p="md" gap="sm">
                <div className="metric">
                  <Text size="sm" c="dimmed">
                    Đầu tay (cm)
                  </Text>
                  <Text size="sm" className="mono">
                    {toBase(snapshot.ee, [...meta.pick_region.pan, 0.42])
                      .map((v) => (v * 100).toFixed(1))
                      .join(" / ")}
                  </Text>
                </div>
                <div className="metric">
                  <Text size="sm" c="dimmed">
                    Độ nghiêng kẹp
                  </Text>
                  <Text size="sm" className="mono">
                    {snapshot.tilt.toFixed(1)}°
                  </Text>
                </div>
                <Text
                  size="sm"
                  c={snapshot.contacts.length ? "orange" : "teal"}
                >
                  {snapshot.contacts.length
                    ? `${snapshot.contacts.length} tiếp xúc`
                    : "Không có va chạm với tay"}
                </Text>
                {snapshot.contacts.map((c, i) => (
                  <Text key={i} size="xs" c="dimmed">
                    {c.a} · {c.b} ({c.pen_mm.toFixed(1)} mm)
                  </Text>
                ))}
                <Text size="xs" c="dimmed">
                  Preview chỉnh tư thế tĩnh; bấm Chạy để tính trọng lực, ma sát
                  và va chạm theo thời gian.
                </Text>
              </Stack>
            </Panel>
          )}
          <div className="sidebar-note">
            <SlidersHorizontal size={15} />
            <Text size="xs" c="dimmed">
              Ẩn hoặc hiện card trong màn Settings.
            </Text>
          </div>
        </aside>
      </div>
    </>
  );
}
