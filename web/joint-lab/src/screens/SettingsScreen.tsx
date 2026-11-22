import { useState } from "react";
import {
  Alert,
  Button,
  FileButton,
  Group,
  Select,
  Stack,
  Switch,
  Table,
  Text,
} from "@mantine/core";
import { Download, RotateCcw, Upload } from "lucide-react";
import { useStudio } from "../studio/StudioContext";
import {
  DEFAULT_SETTINGS,
  PANEL_LABELS,
  parseSettings,
} from "../studio/settings";
import { downloadJson } from "../studio/scene";
import { Panel } from "../components/Panel";
export default function SettingsScreen() {
  const { settings, setSettings, health, connected, setError } = useStudio();
  const [message, setMessage] = useState(""),
    [previous, setPrevious] = useState<typeof settings | null>(null);
  return (
    <div className="standard-screen settings-screen">
      <div className="screen-toolbar">
        <div>
          <Text fw={600}>Cài đặt công cụ</Text>
          <Text size="sm" c="dimmed">
            Thay đổi được lưu tự động trên trình duyệt này.
          </Text>
        </div>
        <Group>
          <Button
            variant="default"
            leftSection={<Download size={16} />}
            onClick={() => downloadJson("sim-studio-settings.json", settings)}
          >
            Xuất cài đặt
          </Button>
          <FileButton
            accept="application/json,.json"
            onChange={async (file) => {
              if (!file) return;
              try {
                setPrevious(settings);
                setSettings(parseSettings(JSON.parse(await file.text())));
                setMessage("Đã nạp cài đặt.");
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
                Nạp cài đặt
              </Button>
            )}
          </FileButton>
        </Group>
      </div>
      {message && (
        <Alert
          color="teal"
          withCloseButton
          onClose={() => setMessage("")}
          mb="md"
        >
          {message}
          {previous && (
            <Button
              size="compact-sm"
              variant="subtle"
              ml="sm"
              onClick={() => {
                setSettings(previous);
                setPrevious(null);
                setMessage("Đã hoàn tác.");
              }}
            >
              Hoàn tác
            </Button>
          )}
        </Alert>
      )}
      <div className="settings-grid">
        <Panel title="Giao diện">
          <Stack p="lg" gap="lg">
            <Select
              label="Chế độ màu"
              value={settings.theme}
              onChange={(value) =>
                setSettings({
                  ...settings,
                  theme: value as typeof settings.theme,
                })
              }
              data={[
                { value: "dark", label: "Tối" },
                { value: "light", label: "Sáng" },
                { value: "auto", label: "Theo hệ thống" },
              ]}
              allowDeselect={false}
            />
            <Select
              label="Màn hình khi mở ứng dụng"
              value={settings.startPage}
              onChange={(value) =>
                setSettings({
                  ...settings,
                  startPage: value as typeof settings.startPage,
                })
              }
              data={[
                { value: "replay", label: "Replay" },
                { value: "lab", label: "Lab" },
                { value: "runs", label: "Lịch sử chạy" },
                { value: "settings", label: "Settings" },
              ]}
              allowDeselect={false}
            />
            <Switch
              label="Thu gọn sidebar"
              checked={settings.collapsed}
              onChange={(event) =>
                setSettings({
                  ...settings,
                  collapsed: event.currentTarget.checked,
                })
              }
            />
          </Stack>
        </Panel>
        <Panel title="Replay">
          <Stack p="lg" gap="lg">
            <Select
              label="Tốc độ phát mặc định"
              value={settings.defaultSpeed}
              onChange={(value) =>
                setSettings({ ...settings, defaultSpeed: value || "1" })
              }
              data={["0.1", "0.25", "0.5", "1", "1.5", "2", "3", "4"].map(
                (value) => ({ value, label: value + "×" }),
              )}
              allowDeselect={false}
            />
            <Switch
              label="Tự phát khi mở replay"
              description="Có thể tạm dừng bằng Space trong khung Replay."
              checked={settings.autoPlay}
              onChange={(event) =>
                setSettings({
                  ...settings,
                  autoPlay: event.currentTarget.checked,
                })
              }
            />
            <Switch
              label="Mở Replay sau khi Lab chạy xong"
              checked={settings.autoOpenReplay}
              onChange={(event) =>
                setSettings({
                  ...settings,
                  autoOpenReplay: event.currentTarget.checked,
                })
              }
            />
          </Stack>
        </Panel>
        <Panel title="Card hiển thị trong Lab">
          <Stack p="lg" gap="lg">
            {Object.entries(PANEL_LABELS).map(([key, label]) => (
              <Switch
                key={key}
                label={label}
                checked={settings.panels[key as keyof typeof settings.panels]}
                onChange={(event) =>
                  setSettings({
                    ...settings,
                    panels: {
                      ...settings.panels,
                      [key]: event.currentTarget.checked,
                    },
                  })
                }
              />
            ))}
            <Text size="sm" c="dimmed">
              Không gian 3D luôn hiển thị. Màn Replay giữ bố cục phân tích
              riêng.
            </Text>
          </Stack>
        </Panel>
        <Panel
          title="Hệ thống và mô phỏng"
          badge={connected ? "Đã kết nối" : "Ngoại tuyến"}
        >
          <Stack p="lg">
            <Table verticalSpacing="sm">
              <Table.Tbody>
                {[
                  [
                    "Động cơ vật lý",
                    health ? "MuJoCo " + health.mujoco : "Đang kiểm tra…",
                  ],
                  ["Mô hình", health?.scene || "—"],
                  [
                    "Tần số điều khiển",
                    health ? health.control_hz + " Hz" : "—",
                  ],
                  ["Bước vật lý", health ? health.physics_dt + " s" : "—"],
                  [
                    "Giới hạn episode",
                    health ? health.max_steps + " bước" : "—",
                  ],
                  [
                    "Vùng thao tác",
                    health
                      ? `${Math.round(health.workspace.r_min * 100)}–${Math.round(health.workspace.r_max * 100)} cm`
                      : "—",
                  ],
                  ["Vật thể", "Tối đa 5 hộp / bóng"],
                  ["Lưu dữ liệu", "Cục bộ trên máy này"],
                ].map(([key, value]) => (
                  <Table.Tr key={key}>
                    <Table.Td>
                      <Text size="sm" c="dimmed">
                        {key}
                      </Text>
                    </Table.Td>
                    <Table.Td>
                      <Text size="sm">{value}</Text>
                    </Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
            <Text size="sm" c="dimmed">
              Thông số vật lý ở đây chỉ để kiểm tra. Cấu hình mô hình và policy
              được quản lý trong mã nguồn để các lần chạy có thể tái lập.
            </Text>
          </Stack>
        </Panel>
      </div>
      <Group mt="xl">
        <Button
          variant="default"
          leftSection={<RotateCcw size={16} />}
          onClick={() => {
            setPrevious(settings);
            setSettings({
              ...DEFAULT_SETTINGS,
              panels: { ...DEFAULT_SETTINGS.panels },
            });
            setMessage("Đã khôi phục cài đặt mặc định.");
          }}
        >
          Khôi phục mặc định
        </Button>
        <Text size="sm" c="dimmed">
          Không xóa cảnh, lịch sử hoặc dữ liệu replay.
        </Text>
      </Group>
    </div>
  );
}
