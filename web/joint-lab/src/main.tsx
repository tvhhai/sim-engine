import React from "react";
import ReactDOM from "react-dom/client";
import { MantineProvider, createTheme } from "@mantine/core";
import "@mantine/core/styles.css";
import "./styles.css";
import "./studio/studio.css";
import App from "./App";
const theme = createTheme({
  primaryColor: "teal",
  fontFamily: "Inter, Segoe UI, system-ui, sans-serif",
  defaultRadius: "md",
  colors: {
    teal: [
      "#e6fcf4",
      "#c3f4e2",
      "#a0ecd0",
      "#7ee4be",
      "#5bdcad",
      "#3dd39c",
      "#2bb984",
      "#20956a",
      "#167350",
      "#0a5138",
    ],
  },
  components: {
    Button: { defaultProps: { fw: 600 } },
    NumberInput: {
      defaultProps: { size: "sm", clampBehavior: "strict", decimalScale: 1 },
    },
    Input: { styles: { input: { fontVariantNumeric: "tabular-nums" } } },
  },
});
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <MantineProvider theme={theme} defaultColorScheme="dark">
      <App />
    </MantineProvider>
  </React.StrictMode>,
);
