import { Component } from "react";
import type { ErrorInfo, ReactNode } from "react";
import { Alert, Button, Stack, Text } from "@mantine/core";
export class ScreenBoundary extends Component<
  { children: ReactNode; resetKey: string },
  { error: string }
> {
  state = { error: "" };
  static getDerivedStateFromError(error: Error) {
    return { error: error.message };
  }
  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Studio screen failed", error, info.componentStack);
  }
  componentDidUpdate(previous: { resetKey: string }) {
    if (previous.resetKey !== this.props.resetKey && this.state.error)
      this.setState({ error: "" });
  }
  render() {
    return this.state.error ? (
      <Stack p="xl">
        <Alert color="red" title="Không mở được màn hình">
          {this.state.error}
        </Alert>
        <Text c="dimmed">
          Bản nháp Lab đã được tự lưu. Tải lại trang để lấy phiên bản giao diện
          mới nhất.
        </Text>
        <Button variant="default" onClick={() => location.reload()}>
          Tải lại trang
        </Button>
      </Stack>
    ) : (
      this.props.children
    );
  }
}
