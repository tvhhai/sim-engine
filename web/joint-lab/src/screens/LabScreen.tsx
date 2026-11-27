import {
  Badge,
  Button,
  Checkbox,
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
import { useMemo, useState } from "react";
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
import { Panel } from "../components/Panel";
import { TopView } from "../components/TopView";
import { labTopView } from "../components/topViewGeometry";
import { SceneView } from "../components/LazySceneView";
import { ObjectPanel } from "../components/ObjectPanel";
import { useStudio } from "../studio/StudioContext";
import { bowlGeoms, downloadJson, minBowlRadius, parseScene } from "../studio/scene";
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
  const [showRegion, setShowRegion] = useState(true);
  const topView = useMemo(() => scene && meta && model ? labTopView(scene, snapshot, model, {
    region: meta.pick_region, stop: meta.unreachable_region, scan: meta.pick_scan,
  }) : null, [scene, snapshot, model, meta]);
  if (!scene || !meta || !model || !topView)
    return (
      <div className="screen-empty">
        <Loader color="teal" />
        <Title order={2}>Đang nạp Lab…</Title>
        <Text c="dimmed">{error || "Đang nạp mô hình và cảnh đã lưu."}</Text>
      </div>
    );
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
          <Checkbox
            label="Gắp tất cả"
            checked={scene.mode === "all"}
            disabled={running}
            onChange={(event) => setScene({ ...scene, mode: event.currentTarget.checked ? "all" : "single" })}
          />
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
                <Panel title="Nhìn từ trên · XY (m)" badge="Kéo hộp, bóng, bát" tools={
                  <label className="xy-region-toggle"><input type="checkbox" checked={showRegion} onChange={(e) => setShowRegion(e.currentTarget.checked)} /> Vùng gắp</label>
                }>
                  <div className="top-view">
                  <TopView
                    data={topView}
                    showRegion={showRegion}
                    selected={selected}
                    onSelect={setSelected}
                    onMove={move}
                  />
                  </div>
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
        </aside>
        <aside className="controls-column controls-secondary">
          {visible.bowl && (
            <Panel title="Vị trí bát" badge="Giới hạn vùng">
              <Stack p="sm" gap="xs">
                <Group grow>
                  <NumberInput
                    size="xs"
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
                    size="xs"
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
                    size="xs"
                    label="Bán kính (cm)"
                    description={`${(minBowlRadius(scene.objects) * 100).toFixed(1)} – 9 (chứa được vật)`}
                    value={Number((scene.bowl.radius * 100).toFixed(1))}
                    onChange={(v) =>
                      setScene({ ...scene, bowl: { ...scene.bowl, radius: Math.max(minBowlRadius(scene.objects), Math.min(0.09, Number(v) / 100 || 0.045)) } })
                    }
                    min={minBowlRadius(scene.objects) * 100}
                    max={9}
                    step={0.5}
                  />
                  <NumberInput
                    size="xs"
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
                  size="xs"
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
                <Button
                  size="compact-sm"
                  variant="subtle"
                  color="gray"
                  leftSection={<RotateCcw size={14} />}
                  onClick={() => {
                    const pose = { ...scene.pose };
                    for (const [key] of JOINTS) pose[key] = meta.home[key] ?? pose[key];
                    setScene({ ...scene, pose });
                  }}
                >
                  Về tư thế nghỉ
                </Button>
              }
            >
              <Stack p="sm" gap="sm">
                {JOINTS.map(([key, label], i) => (
                  <div key={key} className="lab-joint-row">
                    <Text size="sm">{label}</Text>
                    <Slider
                      aria-label={label}
                      min={meta.limits[i][0]}
                      max={meta.limits[i][1]}
                      step={0.1}
                      value={scene.pose[key]}
                      onChange={(v) => setJoint(key, v)}
                      label={(v) => `${v.toFixed(1)}°`}
                    />
                    <NumberInput
                      aria-label={`${label} (độ)`}
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
                  </div>
                ))}
                <div>
                  <div className="lab-joint-row">
                    <Text size="sm">Hàm kẹp</Text>
                    <Slider
                      aria-label="Độ mở hàm kẹp"
                      min={meta.gclose}
                      max={meta.gopen}
                      step={0.01}
                      value={scene.pose.grip}
                      onChange={(v) => setJoint("grip", v)}
                    />
                    <NumberInput
                      aria-label="Hàm kẹp (độ)"
                      size="xs"
                      value={Number(((scene.pose.grip * 180) / Math.PI).toFixed(1))}
                      min={(meta.gclose * 180) / Math.PI}
                      max={(meta.gopen * 180) / Math.PI}
                      step={0.5}
                      onChange={(v) => setJoint("grip", Math.max(meta.gclose, Math.min(meta.gopen, Number(v) * Math.PI / 180)))}
                    />
                  </div>
                  <Group justify="space-between" mt={4}>
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
