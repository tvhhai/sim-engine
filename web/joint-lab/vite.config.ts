import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react()],
  base: "/lab/",
  build: { outDir: "../../static/studio", emptyOutDir: true },
  server: {
    proxy: Object.fromEntries(
      [
        "/meta",
        "/model",
        "/pose",
        "/ik",
        "/run",
        "/run-status",
        "/replay-lab",
        "/lab-replay-data",
        "/scene-pose",
        "/scene-ik",
        "/studio",
      ].map((path) => [path, "http://127.0.0.1:8765"]),
    ),
  },
});
