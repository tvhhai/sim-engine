import { Suspense, useEffect, useState } from "react";
import {
  ActionIcon,
  Alert,
  Badge,
  Burger,
  Button,
  Drawer,
  Group,
  Loader,
  Modal,
  Stack,
  Text,
  Title,
  Tooltip,
} from "@mantine/core";
import {
  CircleHelp,
  FlaskConical,
  Layers3,
  PanelLeftClose,
  PanelLeftOpen,
} from "lucide-react";
import { SCREENS } from "./screens";
import { useStudio } from "./StudioContext";
import { ScreenBoundary } from "../components/ScreenBoundary";
export default function StudioShell() {
  const studio = useStudio();
  const {
    route,
    settings,
    setSettings,
    error,
    setError,
    run,
    running,
    connected,
    health,
  } = studio;
  const [mobile, setMobile] = useState(false),
    [help, setHelp] = useState(false);
  const active = SCREENS.find((screen) => screen.id === route.id),
    Screen = active?.component;
  useEffect(() => {
    document.title = `Sim Studio · ${active?.title || "Không tìm thấy"}`;
    setMobile(false);
  }, [active?.title]);
  useEffect(() => {
    const listener = (event: KeyboardEvent) => {
      if (event.altKey && ["1", "2", "3", "4"].includes(event.key)) {
        event.preventDefault();
        route.navigate(SCREENS[Number(event.key) - 1].id);
      }
      if (
        event.ctrlKey &&
        event.key === "Enter" &&
        route.id === "lab" &&
        !studio.running &&
        !studio.previewing &&
        !studio.ikBusy
      ) {
        event.preventDefault();
        studio.start();
      }
      if (
        event.key === "?" &&
        !(event.target as HTMLElement).closest(
          "input,textarea,[contenteditable]",
        )
      )
        setHelp(true);
    };
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, [route, studio]);
  const navigation = (
    <>
      <a
        className="studio-brand"
        href="#/replay"
        aria-label="Sim Studio · Replay"
      >
        <span className="brand-icon">
          <Layers3 size={23} />
        </span>
        {!settings.collapsed && (
          <span>
            <strong>Sim Studio</strong>
            <small>SO-101 WORKSPACE</small>
          </span>
        )}
      </a>
      <nav aria-label="Điều hướng chính" className="studio-nav">
        {SCREENS.map(({ id, title, icon: Icon }) => (
          <Tooltip
            key={id}
            label={title}
            position="right"
            disabled={!settings.collapsed}
          >
            <a
              href={`#/${id}`}
              className={route.id === id ? "active" : ""}
              aria-current={route.id === id ? "page" : undefined}
              aria-label={title}
            >
              <Icon size={20} aria-hidden="true" />
              {!settings.collapsed && <span>{title}</span>}
              {id === "lab" && running && <span className="nav-running" />}
            </a>
          </Tooltip>
        ))}
      </nav>
      <div className="studio-sidebar-bottom">
        {!settings.collapsed && (
          <div className="backend-summary">
            <span className={`dot ${connected ? "mint" : "offline-dot"}`} />
            <Text size="sm">
              {connected ? "MuJoCo " + health?.mujoco : "Đang kết nối…"}
            </Text>
            <Text size="xs" c="dimmed">
              Chạy cục bộ · seed lab riêng
            </Text>
          </div>
        )}
        <Tooltip label={settings.collapsed ? "Mở rộng menu" : "Thu gọn menu"}>
          <ActionIcon
            aria-label={settings.collapsed ? "Mở rộng menu" : "Thu gọn menu"}
            variant="subtle"
            color="gray"
            size="lg"
            onClick={() =>
              setSettings({ ...settings, collapsed: !settings.collapsed })
            }
          >
            {settings.collapsed ? (
              <PanelLeftOpen size={19} />
            ) : (
              <PanelLeftClose size={19} />
            )}
          </ActionIcon>
        </Tooltip>
      </div>
    </>
  );
  return (
    <div
      className={`studio-shell ${settings.collapsed ? "sidebar-collapsed" : ""}`}
    >
      <a
        href="#studio-content"
        className="skip-link"
        onClick={(event) => {
          event.preventDefault();
          document.getElementById("studio-content")?.focus();
        }}
      >
        Đến nội dung chính
      </a>
      <aside className="studio-sidebar">{navigation}</aside>
      <div className="studio-body">
        <header className="studio-header">
          <Group gap="md">
            <Burger
              className="mobile-navigation"
              opened={mobile}
              onClick={() => setMobile(!mobile)}
              aria-label="Mở menu điều hướng"
              size="sm"
            />
            <div>
              <Title order={1} size="h3">
                {active?.title || "Không tìm thấy màn hình"}
              </Title>
              <Text c="dimmed" size="sm">
                {active?.description}
              </Text>
            </div>
          </Group>
          <Group gap="sm">
            <Badge color={connected ? "teal" : "orange"} variant="light">
              {connected ? "Kết nối cục bộ" : "Mất kết nối"}
            </Badge>
            <Tooltip label="Hướng dẫn và phím tắt">
              <ActionIcon
                aria-label="Hướng dẫn và phím tắt"
                variant="default"
                size="lg"
                onClick={() => setHelp(true)}
              >
                <CircleHelp size={19} />
              </ActionIcon>
            </Tooltip>
            {route.id !== "lab" && (
              <Button
                component="a"
                href="#/lab"
                variant="default"
                leftSection={<FlaskConical size={16} />}
              >
                Mở Lab
              </Button>
            )}
          </Group>
        </header>
        {running && (
          <div className="global-run-status" role="status">
            <Loader size={14} color="teal" />
            <span>
              Đang mô phỏng seed lab. Bạn có thể xem các kết quả đã lưu trong
              lúc đợi.
            </span>
          </div>
        )}
        {error && (
          <Alert
            color="red"
            m="md"
            withCloseButton
            title="Có lỗi cần kiểm tra"
            onClose={() => setError("")}
          >
            {error}
          </Alert>
        )}
        <main
          id="studio-content"
          tabIndex={-1}
          className={`studio-content screen-${route.id}`}
        >
          <ScreenBoundary resetKey={route.id}>
            <Suspense
              fallback={
                <div className="screen-empty">
                  <Loader color="teal" />
                  <Text c="dimmed">Đang mở màn hình…</Text>
                </div>
              }
            >
              {Screen ? (
                <Screen />
              ) : (
                <div className="screen-empty">
                  <Title order={2}>Màn hình không tồn tại</Title>
                  <Button component="a" href="#/replay">
                    Về Replay
                  </Button>
                </div>
              )}
            </Suspense>
          </ScreenBoundary>
        </main>
        <div className="studio-statusbar">
          <span>
            {connected ? "MuJoCo sẵn sàng" : "Không kết nối được máy chủ"}
            {health ? ` · ${health.control_hz} Hz` : ""}
          </span>
          {route.id !== "replay" && (
            <span>
              {run.status === "done"
                ? `${run.success ? "Thành công" : "Kết thúc"} · ${run.steps} bước`
                : running
                  ? "Đang chạy"
                  : "Sẵn sàng"}{" "}
              · Không chạy lại seed cũ
            </span>
          )}
        </div>
      </div>
      <Drawer
        opened={mobile}
        onClose={() => setMobile(false)}
        title="Điều hướng"
        position="left"
        closeButtonProps={{ "aria-label": "Đóng menu" }}
      >
        <div className="mobile-nav-content">{navigation}</div>
      </Drawer>
      <Modal
        opened={help}
        onClose={() => setHelp(false)}
        title="Hướng dẫn sử dụng"
        centered
        closeButtonProps={{ "aria-label": "Đóng hướng dẫn" }}
      >
        <Stack>
          <Text size="sm">
            <b>Lab:</b> kéo vật/bát trên XY hoặc chọn vật trong 3D để kéo trục.
            Bấm Chạy mô phỏng để tạo một kết quả mới.
          </Text>
          <Text size="sm">
            <b>Replay:</b> chế độ Phân tích đầy đủ giữ giao diện replay.html;
            chế độ 3D nhiều vật dùng mesh gốc. Chọn nguồn để xem ván cũ mà không
            chạy lại.
          </Text>
          <Text size="sm">
            <b>Lịch sử:</b> đặt tên, ghi chú, đánh dấu, so sánh hoặc nạp cảnh về
            Lab.
          </Text>
          <Text size="sm" className="mono">
            Alt + 1…4: đổi màn hình
            <br />
            Ctrl + Enter: chạy Lab
            <br />
            ?: mở hướng dẫn
          </Text>
          <Text size="sm" c="dimmed">
            Trong khung Replay: ←/→ từng bước, ↑/↓ chuyển phase, Space phát/tạm
            dừng, Home/End về đầu/cuối.
          </Text>
        </Stack>
      </Modal>
    </div>
  );
}
