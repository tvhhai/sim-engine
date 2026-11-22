import { lazy } from "react";
import { FlaskConical, History, MonitorPlay, Settings2 } from "lucide-react";
// Register a new screen here; the shell builds navigation automatically.
export const SCREENS = [
  {
    id: "replay",
    title: "Replay",
    description: "Xem và phân tích mô phỏng từng bước",
    icon: MonitorPlay,
    component: lazy(() => import("../screens/ReplayScreen")),
  },
  {
    id: "lab",
    title: "Lab",
    description: "Đặt cảnh và thử nghiệm cánh tay",
    icon: FlaskConical,
    component: lazy(() => import("../screens/LabScreen")),
  },
  {
    id: "runs",
    title: "Lịch sử chạy",
    description: "Kết quả, cấu hình và ghi chú thử nghiệm",
    icon: History,
    component: lazy(() => import("../screens/RunsScreen")),
  },
  {
    id: "settings",
    title: "Settings",
    description: "Cấu hình giao diện và thông tin hệ thống",
    icon: Settings2,
    component: lazy(() => import("../screens/SettingsScreen")),
  },
];
