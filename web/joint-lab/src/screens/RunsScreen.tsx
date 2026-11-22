import { useMemo, useState } from "react";
import {
  ActionIcon,
  Alert,
  Badge,
  Button,
  Checkbox,
  Group,
  Modal,
  Pagination,
  Select,
  Stack,
  Table,
  Text,
  Textarea,
  TextInput,
  Tooltip,
} from "@mantine/core";
import {
  Download,
  FlaskConical,
  History,
  MonitorPlay,
  Pencil,
  Play,
  RefreshCw,
  Search,
  Star,
} from "lucide-react";
import { useStudio } from "../studio/StudioContext";
import { api } from "../api";
import type { RunRecord } from "../types";
const date = (value: string) =>
  new Intl.DateTimeFormat("vi-VN", {
    dateStyle: "short",
    timeStyle: "short",
  }).format(new Date(value));
export default function RunsScreen() {
  const { catalog, refreshCatalog, route, loadCase, rerun, running, setError } =
    useStudio();
  const [search, setSearch] = useState(""),
    [filter, setFilter] = useState<string | null>("all"),
    [sort, setSort] = useState<string | null>("newest"),
    [page, setPage] = useState(1);
  const [editing, setEditing] = useState<RunRecord | null>(null),
    [title, setTitle] = useState(""),
    [notes, setNotes] = useState(""),
    [saving, setSaving] = useState(false);
  const [selected, setSelected] = useState<string[]>([]),
    [compare, setCompare] = useState(false);
  const records = useMemo(() => {
    const text = search.toLowerCase().trim();
    const result = catalog.runs.filter(
      (record) =>
        (!text ||
          `${record.title} ${record.notes} ${record.id}`
            .toLowerCase()
            .includes(text)) &&
        (filter === "all" ||
          (filter === "favorites" && record.favorite) ||
          (filter === "blocked" && record.blocked_reason) ||
          (filter === "ok" && record.success) ||
          (filter === "fail" && !record.success && !record.blocked_reason)),
    );
    return result.sort((a, b) =>
      sort === "steps"
        ? a.steps - b.steps
        : sort === "oldest"
          ? a.id.localeCompare(b.id)
          : b.id.localeCompare(a.id),
    );
  }, [catalog.runs, search, filter, sort]);
  const maxPage = Math.max(1, Math.ceil(records.length / 10)),
    current = Math.min(page, maxPage),
    rows = records.slice((current - 1) * 10, current * 10);
  async function action(fn: () => Promise<unknown>) {
    try {
      await fn();
    } catch (e) {
      setError((e as Error).message);
    }
  }
  async function favorite(record: RunRecord) {
    await action(async () => {
      await api("/studio/runs/" + record.id + "/meta", {
        favorite: !record.favorite,
      });
      await refreshCatalog();
    });
  }
  async function save() {
    if (!editing || !title.trim()) return;
    setSaving(true);
    try {
      await api("/studio/runs/" + editing.id + "/meta", { title, notes });
      await refreshCatalog();
      setEditing(null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  }
  const chosen = selected
    .map((id) => catalog.runs.find((record) => record.id === id))
    .filter((record): record is RunRecord => !!record);
  return (
    <div className="standard-screen">
      <div className="screen-toolbar">
        <div>
          <Text fw={600}>Các lần chạy đã lưu</Text>
          <Text size="sm" c="dimmed">
            Mỗi dòng giữ nguyên kết quả và cảnh ban đầu. Xem lại không chạy mô
            phỏng.
          </Text>
        </div>
        <Group>
          <Button
            variant="default"
            disabled={chosen.length !== 2}
            onClick={() => setCompare(true)}
          >
            So sánh {chosen.length ? `(${chosen.length}/2)` : ""}
          </Button>
          <Button
            variant="default"
            leftSection={<RefreshCw size={16} />}
            onClick={() => action(refreshCatalog)}
          >
            Làm mới
          </Button>
        </Group>
      </div>
      <div className="history-stats">
        <div>
          <Text c="dimmed" size="sm">
            Lần chạy đã lưu
          </Text>
          <Text fw={600} size="xl">
            {catalog.runs.length}
          </Text>
        </div>
        <div>
          <Text c="dimmed" size="sm">
            Thành công
          </Text>
          <Text fw={600} size="xl" c="teal">
            {catalog.runs.filter((record) => record.success).length}
          </Text>
        </div>
        <div>
          <Text c="dimmed" size="sm">
            Đứng yên / ngoài vùng
          </Text>
          <Text fw={600} size="xl" c="orange">
            {catalog.runs.filter((record) => record.blocked_reason).length}
          </Text>
        </div>
        <div>
          <Text c="dimmed" size="sm">
            Đánh dấu
          </Text>
          <Text fw={600} size="xl">
            {catalog.runs.filter((record) => record.favorite).length}
          </Text>
        </div>
      </div>
      <Group className="history-filters">
        <TextInput
          aria-label="Tìm lần chạy"
          placeholder="Tìm tên, ghi chú hoặc ID…"
          value={search}
          onChange={(event) => {
            setSearch(event.currentTarget.value);
            setPage(1);
          }}
          leftSection={<Search size={16} />}
          className="history-search"
        />
        <Select
          aria-label="Lọc kết quả lịch sử"
          value={filter}
          onChange={(value) => {
            setFilter(value);
            setPage(1);
          }}
          data={[
            { value: "all", label: "Tất cả kết quả" },
            { value: "ok", label: "Thành công" },
            { value: "fail", label: "Thất bại" },
            { value: "blocked", label: "Đứng yên" },
            { value: "favorites", label: "Đã đánh dấu" },
          ]}
          allowDeselect={false}
        />
        <Select
          aria-label="Sắp xếp lịch sử"
          value={sort}
          onChange={(value) => {
            setSort(value);
            setPage(1);
          }}
          data={[
            { value: "newest", label: "Mới nhất trước" },
            { value: "oldest", label: "Cũ nhất trước" },
            { value: "steps", label: "Ít bước nhất" },
          ]}
          allowDeselect={false}
        />
      </Group>
      {!records.length ? (
        <div className="screen-empty">
          <History size={34} />
          <Text fw={600}>
            {catalog.runs.length
              ? "Không có lần chạy phù hợp"
              : "Chưa có lịch sử chạy"}
          </Text>
          <Text c="dimmed">
            {catalog.runs.length
              ? "Thử đổi từ khóa hoặc bộ lọc."
              : "Các lần chạy Lab tiếp theo sẽ được lưu tự động tại đây."}
          </Text>
          <Button component="a" href="#/lab" variant="default">
            Mở Lab
          </Button>
        </div>
      ) : (
        <div className="data-table-wrap">
          <Table highlightOnHover verticalSpacing="md" className="runs-table">
            <Table.Thead>
              <Table.Tr>
                <Table.Th aria-label="Chọn so sánh" />
                <Table.Th>Lần chạy</Table.Th>
                <Table.Th>Kết quả</Table.Th>
                <Table.Th>Bước / thời gian</Table.Th>
                <Table.Th>Vật thể</Table.Th>
                <Table.Th>Thao tác</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {rows.map((record) => (
                <Table.Tr key={record.id}>
                  <Table.Td>
                    <Checkbox
                      aria-label={`So sánh ${record.title}`}
                      checked={selected.includes(record.id)}
                      disabled={
                        !selected.includes(record.id) && selected.length >= 2
                      }
                      onChange={(event) =>
                        setSelected(
                          event.currentTarget.checked
                            ? [...selected, record.id]
                            : selected.filter((id) => id !== record.id),
                        )
                      }
                    />
                  </Table.Td>
                  <Table.Td>
                    <Group gap="xs">
                      <ActionIcon
                        aria-label={`${record.favorite ? "Bỏ" : "Thêm"} đánh dấu ${record.title}`}
                        variant="subtle"
                        color={record.favorite ? "yellow" : "gray"}
                        onClick={() => favorite(record)}
                      >
                        <Star
                          size={16}
                          fill={record.favorite ? "currentColor" : "none"}
                        />
                      </ActionIcon>
                      <div>
                        <Text size="sm" fw={600}>
                          {record.title}
                        </Text>
                        <Text size="xs" c="dimmed">
                          {date(record.created_at)}
                        </Text>
                        {record.notes && (
                          <Text size="xs" c="dimmed" lineClamp={1}>
                            {record.notes}
                          </Text>
                        )}
                      </div>
                    </Group>
                  </Table.Td>
                  <Table.Td>
                    <Badge
                      color={
                        record.blocked_reason
                          ? "orange"
                          : record.success
                            ? "teal"
                            : "red"
                      }
                      variant="light"
                    >
                      {record.blocked_reason
                        ? "Đứng yên"
                        : record.success
                          ? "Thành công"
                          : "Thất bại"}
                    </Badge>
                  </Table.Td>
                  <Table.Td>
                    <Text size="sm" className="mono">
                      {record.steps} · {record.duration.toFixed(2)} s
                    </Text>
                  </Table.Td>
                  <Table.Td>{record.objects}</Table.Td>
                  <Table.Td>
                    <Group gap="xs" wrap="nowrap">
                      <Button
                        size="xs"
                        variant="light"
                        leftSection={<MonitorPlay size={14} />}
                        onClick={() =>
                          route.navigate("replay", {
                            source: "run:" + record.id,
                          })
                        }
                      >
                        Xem
                      </Button>
                      <Tooltip label="Nạp cảnh về Lab">
                        <ActionIcon
                          aria-label={`Nạp ${record.title} về Lab`}
                          variant="default"
                          onClick={() =>
                            action(() => loadCase("run:" + record.id))
                          }
                        >
                          <FlaskConical size={16} />
                        </ActionIcon>
                      </Tooltip>
                      <Tooltip label="Chạy lại riêng cảnh này">
                        <ActionIcon
                          aria-label={`Chạy lại ${record.title}`}
                          variant="default"
                          disabled={running}
                          onClick={() =>
                            action(() => rerun("run:" + record.id))
                          }
                        >
                          <Play size={15} />
                        </ActionIcon>
                      </Tooltip>
                      <Tooltip label="Sửa tên và ghi chú">
                        <ActionIcon
                          aria-label={`Sửa ${record.title}`}
                          variant="default"
                          onClick={() => {
                            setEditing(record);
                            setTitle(record.title);
                            setNotes(record.notes);
                          }}
                        >
                          <Pencil size={15} />
                        </ActionIcon>
                      </Tooltip>
                      <Tooltip label="Xuất cảnh">
                        <ActionIcon
                          component="a"
                          href={"/studio/case?source=run:" + record.id}
                          download={`scene-${record.id}.json`}
                          aria-label={`Xuất cảnh ${record.title}`}
                          variant="default"
                        >
                          <Download size={15} />
                        </ActionIcon>
                      </Tooltip>
                    </Group>
                  </Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </div>
      )}
      {!!records.length && (
        <Group justify="space-between" mt="md">
          <Text size="sm" c="dimmed">
            {records.length} kết quả · trang {current} / {maxPage}
          </Text>
          <Pagination
            aria-label="Trang lịch sử"
            value={current}
            onChange={setPage}
            total={maxPage}
            color="teal"
          />
        </Group>
      )}
      <Modal
        opened={!!editing}
        onClose={() => setEditing(null)}
        title="Tên và ghi chú lần chạy"
        centered
        closeButtonProps={{ "aria-label": "Đóng chỉnh sửa" }}
      >
        <Stack>
          <TextInput
            label="Tên lần chạy"
            value={title}
            onChange={(event) => setTitle(event.currentTarget.value)}
            maxLength={120}
            required
          />
          <Textarea
            label="Ghi chú"
            placeholder="Điều gì đáng chú ý trong lần chạy này?"
            value={notes}
            onChange={(event) => setNotes(event.currentTarget.value)}
            maxLength={4000}
            minRows={4}
          />
          <Group justify="end">
            <Button variant="default" onClick={() => setEditing(null)}>
              Hủy
            </Button>
            <Button onClick={save} loading={saving} disabled={!title.trim()}>
              Lưu ghi chú
            </Button>
          </Group>
        </Stack>
      </Modal>
      <Modal
        opened={compare}
        onClose={() => setCompare(false)}
        title="So sánh hai lần chạy"
        size="lg"
        centered
        closeButtonProps={{ "aria-label": "Đóng so sánh" }}
      >
        {chosen.length === 2 ? (
          <>
            <Table verticalSpacing="sm">
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Thông số</Table.Th>
                  {chosen.map((record) => (
                    <Table.Th key={record.id}>{record.title}</Table.Th>
                  ))}
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {[
                  [
                    "Kết quả",
                    ...chosen.map((record) =>
                      record.blocked_reason
                        ? "Đứng yên"
                        : record.success
                          ? "Thành công"
                          : "Thất bại",
                    ),
                  ],
                  ["Số bước", ...chosen.map((record) => record.steps)],
                  [
                    "Thời gian mô phỏng",
                    ...chosen.map(
                      (record) => record.duration.toFixed(2) + " s",
                    ),
                  ],
                  ["Số vật thể", ...chosen.map((record) => record.objects)],
                  [
                    "Lý do dừng",
                    ...chosen.map((record) => record.blocked_reason || "—"),
                  ],
                ].map(([label, ...values]) => (
                  <Table.Tr key={String(label)}>
                    <Table.Td>{label}</Table.Td>
                    {values.map((value, i) => (
                      <Table.Td key={i}>{value}</Table.Td>
                    ))}
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
            <Group mt="lg" grow>
              {chosen.map((record) => (
                <Button
                  key={record.id}
                  variant="default"
                  onClick={() => {
                    setCompare(false);
                    route.navigate("replay", { source: "run:" + record.id });
                  }}
                >
                  Mở {record.title}
                </Button>
              ))}
            </Group>
          </>
        ) : (
          <Alert>Hãy chọn đúng hai lần chạy.</Alert>
        )}
      </Modal>
    </div>
  );
}
