import {
  ActionIcon,
  Badge,
  Button,
  ColorInput,
  Group,
  MultiSelect,
  NumberInput,
  Select,
  Stack,
  Text,
} from "@mantine/core";
import { Plus, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import type { Item, Region, Scene } from "../types";
import { fitsBowl, maxObjectSize, NEW_OBJECT_SIZE } from "../studio/scene";
import { reachable } from "../workspace";
import { Panel } from "./Panel";
// Màu chọn nhanh cho vật mới: đỏ, xanh, vàng (vật mới lần lượt lấy màu kế tiếp)
const PALETTE = ["#e5484d", "#30a46c", "#f5c542"];

function ObjectAppearance({ item, index, maxSize, onChange }: {
  item: Item;
  index: number;
  maxSize: number;
  onChange: (patch: Partial<Item>) => boolean;
}) {
  const liveSize = Number(((item.size ?? .03) * 100).toFixed(2));
  const [color, setColor] = useState(item.color);
  const [yaw, setYaw] = useState<string | number>(item.yaw);
  const [size, setSize] = useState<string | number>(liveSize);
  useEffect(() => { setColor(item.color); }, [item.color]);
  useEffect(() => { setYaw(item.yaw); }, [item.yaw]);
  useEffect(() => { setSize(liveSize); }, [liveSize]);
  const validColor = (value: string) => /^#[0-9a-f]{6}$/i.test(value);
  const submitOnEnter = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter") event.currentTarget.blur();
  };

  return (
    <div className="lab-object-fields lab-object-details">
      <Text size="sm" fw={600}>Chỉnh {item.type === "sphere" ? "bóng" : "hộp"} {index + 1}</Text>
      <ColorInput
        size="xs"
        label="Màu"
        eyeDropperButtonProps={{ "aria-label": "Lấy màu cho vật đang chọn" }}
        value={color}
        format="hex"
        swatches={PALETTE}
        onChange={(value) => {
          setColor(value);
          if (validColor(value)) onChange({ color: value });
        }}
        onBlur={() => { if (!validColor(color)) setColor(item.color); }}
      />
      {item.type === "box" && <NumberInput
        size="xs"
        label="Góc xoay (°)"
        value={yaw}
        min={-180}
        max={180}
        onChange={setYaw}
        onKeyDown={submitOnEnter}
        onBlur={() => {
          const degrees = yaw === "" ? item.yaw : Math.max(-180, Math.min(180, Number(yaw)));
          setYaw(onChange({ yaw: degrees }) ? degrees : item.yaw);
        }}
      />}
      <NumberInput
        size="xs"
        label={item.type === "sphere" ? "Đường kính (cm)" : "Cạnh (cm)"}
        value={size}
        min={2}
        max={maxSize}
        step={0.5}
        onChange={setSize}
        onKeyDown={submitOnEnter}
        onBlur={() => {
          const cm = size === "" ? liveSize : Math.max(2, Math.min(maxSize, Number(size)));
          setSize(Number.isFinite(cm) && onChange({ size: cm / 100 }) ? cm : liveSize);
        }}
      />
    </div>
  );
}

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
  const selectedIndex = scene.objects.findIndex((obj) => obj.id === selected);
  const appearanceIndex = selectedIndex < 0 ? 0 : selectedIndex;
  const appearance = scene.objects[appearanceIndex];
  const targetIds = scene.target_ids ?? [scene.target_id];
  const allSelected = targetIds.length === scene.objects.length;
  function chooseTargets(ids: string[]) {
    const preferred = ids.find((id) => !targetIds.includes(id)) ?? ids[0];
    onChange({ ...scene, target_ids: ids, target_id: preferred ?? scene.target_id });
    if (preferred) onSelect(preferred);
    setSizeError("");
  }
  function update(id: string, patch: Partial<Item>) {
    if (
      [patch.x, patch.y, patch.yaw].some(
        (value) => value !== undefined && !Number.isFinite(value),
      )
    )
      return false;
    const objects = scene.objects.map((obj) => obj.id === id ? { ...obj, ...patch } : obj);
    if (patch.size !== undefined && !fitsBowl(objects, scene.bowl)) {
      setSizeError("Bát chưa đủ chỗ cho cỡ vật này.");
      return false;
    }
    setSizeError("");
    onChange({ ...scene, objects });
    return true;
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
    setSizeError("");
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
        <Group justify="space-between">
          <Text component="label" htmlFor="lab-target-ids" size="xs" fw={600}>Vật cần gắp</Text>
          <Button size="compact-xs" variant="subtle" disabled={allSelected} onClick={() => chooseTargets(scene.objects.map((obj) => obj.id))}>
            Chọn tất cả
          </Button>
        </Group>
        <MultiSelect
          id="lab-target-ids"
          size="xs"
          value={targetIds}
          onChange={chooseTargets}
          placeholder="Chọn một hoặc nhiều vật"
          hidePickedOptions
          data={scene.objects.map((obj, i) => ({
            value: obj.id,
            label: `${obj.type === "sphere" ? "Bóng" : "Hộp"} ${i + 1}`,
          }))}
        />
        <ObjectAppearance
          key={appearance.id}
          item={appearance}
          index={appearanceIndex}
          maxSize={maxObjectSize(scene.bowl) * 100}
          onChange={(patch) => update(appearance.id, patch)}
        />
        {sizeError && <Text size="xs" c="red" role="status">{sizeError}</Text>}
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
                aria-pressed={appearance.id === obj.id}
                onClick={() => {
                  setSizeError("");
                  onSelect(obj.id);
                }}
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
                      target_ids: targetIds.filter((id) => id !== obj.id),
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
