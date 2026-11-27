# Tra cứu mã bằng CodeGraph và Graphify

Cài CLI chính thức: `uv tool install graphifyy==0.9.121` (gói có hai chữ y).
CLI dùng môi trường riêng của uv, không thay dependency mô phỏng hoặc LeRobot.
Skill được cài theo project cho Codex (`.codex/skills/graphify/`) và Claude
(`.claude/skills/graphify/`), ở cả VLA lẫn sim-engine.

## Chọn công cụ

| Câu hỏi | Công cụ đầu tiên |
|---|---|
| Định nghĩa hàm, callers/callees, ảnh hưởng thay đổi | CodeGraph |
| Bản đồ tổng thể, cụm thành phần, đường nối giữa hai thành phần | Graphify |
| Chuỗi chữ, thông báo lỗi, nội dung cấu hình | `rg` hoặc đọc đúng file |

Không truy cả hai công cụ cho cùng câu hỏi nếu kết quả đầu đã đủ. Graphify
không thay CodeGraph. Quan hệ INFERRED là suy luận của bộ phân giải, cần kiểm tra
đoạn mã liên quan trước khi dựa vào đó để sửa. Chưa có benchmark tốc độ/token
của cách kết hợp này trên repo.

## Tạo và cập nhật

Chạy từ sim-engine bằng PowerShell:

```powershell
./tools/update_graph.ps1 -Html
graphify query "ScriptedPolicy environment" --budget 1200
graphify explain "ScriptedPolicy"
graphify path "ScriptedPolicy" "tools/make_replay.py::Recorder" --undirected
```

Script chỉ trích xuất code bằng AST rồi phân cụm không dùng LLM. `.graphifyignore`
loại dữ liệu, mesh, bản build và cấu hình trợ lý. Chưa trích xuất ngữ nghĩa tài liệu,
PDF, ảnh, notebook bằng mô hình; không coi graph này là chỉ mục đầy đủ tài liệu.

Sau một đợt sửa mã, chạy `./tools/update_graph.ps1` trước khi dùng lại Graphify.
Graphify hiện không có watcher chạy nền. Nếu graph cũ, dùng CodeGraph đang có
watcher hoặc cập nhật graph trước. `graphify-out/` là cache cục bộ, không commit.

VLA dùng chung graph ở `sim-engine/graphify-out/graph.json`, tránh quét cùng repo
hai lần. Khi đứng ở VLA:

```powershell
graphify query "ScriptedPolicy environment" --graph sim-engine/graphify-out/graph.json --budget 1200
./sim-engine/tools/update_graph.ps1 -Html
```

Codex gọi skill bằng `$graphify`, Claude Code bằng `/graphify`. Quy tắc chọn
công cụ ở AGENTS.md ưu tiên hơn mô tả chung của skill. Skill mới được nhận ở
lượt tiếp theo; mở phiên mới nếu trợ lý chưa thấy. Hai trợ lý cùng dùng CLI và
file graph cục bộ, không cần thêm một MCP server trùng chức năng.
