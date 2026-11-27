import {
  ActionIcon,
  Badge,
  Button,
  ColorInput,
  Group,
  NumberInput,
  Select,
  Stack,
  Text,
} from "@mantine/core";
import { Plus, Trash2 } from "lucide-react";
import { useState } from "react";
import type { Item, Region, Scene } from "../types";
import { fitsBowl, maxObjectSize, NEW_OBJECT_SIZE } from "../studio/scene";
import { reachable } from "../workspace";
import { Panel } from "./Panel";
// Màu chọn nhanh cho vật mới: đỏ, xanh, vàng (vật mới lần lượt lấy màu kế tiếp)
const PALETTE = ["#e5484d", "#30a46c", "#f5c542"];
export function ObjectPanel({
  scene,
  selected,
  region,
  onSelect,
  onChange,
}: {
  scene: Scene;
  selected: string;
  region: Region;
  onSelect: (id: string) => void;
  onChange: (scene: Scene) => void;
}) {
  const [kind, setKind] = useState<string | null>("box"),
    [count, setCount] = useState<string | number>(1),
    [undo, setUndo] = useState<Scene | null>(null);
  const newType = kind === "sphere" ? "sphere" : "box";
  const amount = Math.max(1, Math.floor(Number(count) || 1));
  const withinCount = scene.objects.length + amount <= 5;
  const preferredSize = Math.min(NEW_OBJECT_SIZE, maxObjectSize(scene.bowl));
  const newSize = withinCount
    ? [preferredSize, .035, .03, .025, .02].find((size) =>
        size <= preferredSize && fitsBowl([
          ...scene.objects,
          ...Array.from({ length: amount }, (): Pick<Item, "type" | "size"> => ({ type: newType, size })),
        ], scene.bowl),
      )
    : undefined;
  const [sizeError, setSizeError] = useState("");
  function update(id: string, patch: Partial<Item>) {
    if (
      [patch.x, patch.y, patch.yaw].some(
        (value) => value !== undefined && !Number.isFinite(value),
      )
    )
      return;
    const objects = scene.objects.map((obj) => obj.id === id ? { ...obj, ...patch } : obj);
    if (patch.size !== undefined && !fitsBowl(objects, scene.bowl)) {
      setSizeError("Bát chưa đủ chỗ cho cỡ vật này.");
      return;
    }
    setSizeError("");
    onChange({ ...scene, objects });
  }
  function add() {
    if (newSize === undefined) return;
    const added = Array.from(
      { length: amount },
      (_, i): Item => ({
        id: crypto.randomUUID().slice(0, 8),
        type: newType,
        color: PALETTE[(scene.objects.length + i) % PALETTE.length],
        x: -0.1 + (scene.objects.length + i) * 0.055,
        y: 0.12,
        yaw: 0,
        size: newSize,
      }),
    );
    onChange({ ...scene, objects: [...scene.objects, ...added] });
    if (added[0]) onSelect(added[0].id);
  }
  return (
    <Panel title="Vật thể" badge={`${scene.objects.length} / 5`} className="lab-object-panel">
      <Stack gap="xs" p="sm">
        <div className="lab-object-fields">
          <Select
            size="xs"
            label="Loại"
            value={kind}
            onChange={setKind}
            data={[
              { value: "box", label: "Hộp" },
              { value: "sphere", label: "Trái bóng" },
            ]}
            allowDeselect={false}
          />
          <NumberInput
            size="xs"
            label="Số lượng"
            value={count}
            onChange={setCount}
            min={1}
            max={Math.max(1, 5 - scene.objects.length)}
            allowDecimal={false}
          />
        </div>
        <Button leftSection={<Plus size={16} />} onClick={add} disabled={newSize === undefined}>
          Thêm
        </Button>
        <Text size="xs" c="dimmed">
          {!withinCount ? "Tối đa 5 vật trong cảnh." : newSize === undefined
            ? "Bát chưa đủ chỗ. Thử giảm số vật hoặc tăng bán kính bát."
            : `Cỡ vật khi thêm: ${Number((newSize * 100).toFixed(2))} cm.`}
        </Text>
        {sizeError && <Text size="xs" c="red" role="status">{sizeError}</Text>}
        <Select
            size="xs"
          label="Vật cần gắp"
          value={scene.target_id}
          onChange={(value) => {
            if (value) onChange({ ...scene, target_id: value });
          }}
          data={scene.objects.map((obj, i) => ({
            value: obj.id,
            label: `${obj.type === "sphere" ? "Bóng" : "Hộp"} ${i + 1}`,
          }))}
          allowDeselect={false}
        />
        {scene.objects.map((obj, i) => (
          <div
            className={`object-row ${selected === obj.id ? "is-selected" : ""}`}
            key={obj.id}
          >
            <Group justify="space-between" mb="xs">
              <Button
                variant="subtle"
                size="compact-sm"
                color="gray"
                onClick={() => onSelect(obj.id)}
                leftSection={
                  <span
                    className="color-chip"
                    style={{ background: obj.color }}
                  />
                }
              >
                {obj.type === "sphere" ? "Bóng" : "Hộp"} {i + 1}
              </Button>
              <Group gap="xs">
                {!reachable(obj.x, obj.y, region) && (
                  <Badge size="xs" color="orange" variant="light">
                    Ngoài vùng
                  </Badge>
                )}
                <ActionIcon
                  aria-label={`Xóa vật ${i + 1}`}
                  variant="subtle"
                  color="gray"
                  disabled={scene.objects.length === 1}
                  onClick={() => {
                    setUndo(scene);
                    const objects = scene.objects.filter(
                      (o) => o.id !== obj.id,
                    );
                    onChange({
                      ...scene,
                      objects,
                      target_id:
                        scene.target_id === obj.id
                          ? objects[0].id
                          : scene.target_id,
                    });
                  }}
                >
                  <Trash2 size={15} />
                </ActionIcon>
              </Group>
            </Group>
            <div className="lab-object-fields">
              <NumberInput
            size="xs"
                label="X (cm)"
                aria-label={`Vật ${i + 1} X (cm)`}
                value={Math.round((obj.x - region.pan[0]) * 1000) / 10}
                onChange={(v) =>
                  update(obj.id, { x: Number(v) / 100 + region.pan[0] })
                }
                min={-43.5 - region.pan[0] * 100}
                max={43.5 - region.pan[0] * 100}
                step={0.5}
              />
              <NumberInput
            size="xs"
                label="Y (cm)"
                aria-label={`Vật ${i + 1} Y (cm)`}
                value={Math.round((obj.y - region.pan[1]) * 1000) / 10}
                onChange={(v) =>
                  update(obj.id, { y: Number(v) / 100 + region.pan[1] })
                }
                min={-38.5 - region.pan[1] * 100}
                max={38.5 - region.pan[1] * 100}
                step={0.5}
              />
            </div>
            {selected === obj.id && (
              <div className="lab-object-fields lab-object-details">
                <ColorInput
            size="xs"
                  eyeDropperButtonProps={{
                    "aria-label": "Lấy màu cho vật đang chọn",
                  }}
                  label="Màu"
                  value={obj.color}
                  onChange={(v) => {
                    if (/^#[0-9a-f]{6}$/i.test(v)) update(obj.id, { color: v });
                  }}
                  format="hex"
                  swatches={PALETTE}
                />
                <NumberInput
            size="xs"
                  label="Xoay (°)"
                  value={obj.yaw}
                  onChange={(v) => update(obj.id, { yaw: Number(v) })}
                  min={-180}
                  max={180}
                />
                <NumberInput
            size="xs"
                  label={obj.type === "sphere" ? "Đường kính (cm)" : "Cạnh (cm)"}
                  value={Number(((obj.size ?? 0.03) * 100).toFixed(1))}
                  onChange={(v) => {
                    const cm = Number(v);
                    if (Number.isFinite(cm)) update(obj.id, { size: Math.max(0.02, Math.min(maxObjectSize(scene.bowl), cm / 100)) });
                  }}
                  min={2}
                  max={maxObjectSize(scene.bowl) * 100}
                  step={0.5}
                />
              </div>
            )}
          </div>
        ))}
        {undo && (
          <Button
            variant="subtle"
            color="gray"
            onClick={() => {
              onChange(undo);
              setUndo(null);
            }}
          >
            Hoàn tác xóa vật
          </Button>
        )}
      </Stack>
    </Panel>
  );
}
