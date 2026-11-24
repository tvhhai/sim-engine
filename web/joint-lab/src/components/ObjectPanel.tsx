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
import { maxObjectSize, NEW_OBJECT_SIZE } from "../studio/scene";
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
    [color, setColor] = useState(PALETTE[0]),
    [yaw, setYaw] = useState<string | number>(0),
    [size, setSize] = useState<string | number>(NEW_OBJECT_SIZE * 100),
    [count, setCount] = useState<string | number>(1),
    [undo, setUndo] = useState<Scene | null>(null);
  function update(id: string, patch: Partial<Item>) {
    if (
      [patch.x, patch.y, patch.yaw].some(
        (value) => value !== undefined && !Number.isFinite(value),
      )
    )
      return;
    onChange({
      ...scene,
      objects: scene.objects.map((obj) =>
        obj.id === id ? { ...obj, ...patch } : obj,
      ),
    });
  }
  function add() {
    const amount = Math.min(
      5 - scene.objects.length,
      Math.max(1, Number(count) || 1),
    );
    const added = Array.from(
      { length: amount },
      (_, i): Item => ({
        id: crypto.randomUUID().slice(0, 8),
        type: kind === "sphere" ? "sphere" : "box",
        color,
        x: -0.1 + (scene.objects.length + i) * 0.055,
        y: 0.12,
        yaw: kind === "sphere" ? 0 : Number(yaw) || 0,
        size: Math.max(0.02, Math.min(maxObjectSize(scene.bowl), (Number(size) || NEW_OBJECT_SIZE * 100) / 100)),
      }),
    );
    onChange({ ...scene, objects: [...scene.objects, ...added] });
    const next = PALETTE.indexOf(color);
    if (next >= 0) setColor(PALETTE[(next + 1) % PALETTE.length]);
    if (added[0]) onSelect(added[0].id);
  }
  return (
    <Panel title="Vật thể" badge={`${scene.objects.length} / 5`}>
      <Stack gap="xs" p="sm">
        <Group grow align="end">
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
        </Group>
        <Group grow align="end">
          <NumberInput
            size="xs"
            label={kind === "sphere" ? "Đường kính (cm)" : "Cạnh hộp (cm)"}
            description={`2 – ${(maxObjectSize(scene.bowl) * 100).toFixed(1)} (theo bát)`}
            value={size}
            onChange={setSize}
            min={2}
            max={maxObjectSize(scene.bowl) * 100}
            step={0.5}
          />
        <NumberInput
            size="xs"
          label="Góc xoay hộp mới (°)"
          description="0° là hộp thẳng trục; hộp vuông lặp lại sau 90°"
          value={kind === "sphere" ? 0 : yaw}
          onChange={setYaw}
          disabled={kind === "sphere"}
          min={-180}
          max={180}
          step={5}
        />
        </Group>
        <Group align="end" wrap="nowrap">
          <ColorInput
            size="xs"
            eyeDropperButtonProps={{ "aria-label": "Lấy màu cho vật mới" }}
            label="Màu vật mới"
            value={color}
            onChange={setColor}
            format="hex"
            swatches={PALETTE}
          />
          <Button
            leftSection={<Plus size={16} />}
            onClick={add}
            disabled={scene.objects.length >= 5}
          >
            Thêm
          </Button>
        </Group>
        <Text size="xs" c="dimmed">
          Tổng tối đa 5 vật. Vật được chọn làm đích sẽ được policy gắp; những
          vật còn lại có va chạm thật.
        </Text>
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
            <Group grow>
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
            </Group>
            {selected === obj.id && (
              <Group grow mt="xs">
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
              </Group>
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
