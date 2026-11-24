# Sổ tay: từ policy viết tay tới policy AI

Mở file này mỗi khi bí: đang ở đâu trên lộ trình, việc này làm ở thư mục nào, gặp lỗi thì vào đâu, khái niệm lạ thì tra từ khoá gì.
Không cần đọc hết một lần. Mục 1 và 2 đọc trước, phần còn lại tra khi cần.

## 1. Đích đến

**Một mạng nơ-ron nhìn ảnh camera + góc khớp, tự quyết định cử động để gắp hộp / bóng 2-4 cm bỏ vào bát, đạt tỉ lệ thành công gần
policy viết tay trên cùng bộ 774 seed.** Sau đó: nhận lệnh bằng chữ ("gắp bóng đỏ") và chạy trên tay SO-101 thật.

Vì sao cần AI khi đã có policy viết tay gắp được 99 %?

- Policy viết tay **gian lận**: nó hỏi thẳng mô phỏng "hộp đang ở toạ độ nào". Tay thật không có thông tin đó, chỉ có camera.
- Mỗi lần thêm vật mới, cỡ mới, cảnh mới lại phải viết thêm luật. Mạng học từ dữ liệu thì chỉ cần thêm dữ liệu.

Vai trò của policy viết tay từ giờ: **thầy giáo**. Nó làm mẫu, mạng học theo.

## 2. Lộ trình

| Chặng | Việc | Xong khi | Thư mục |
|---|---|---|---|
| 0 (đã xong) | Mô phỏng, policy viết tay, Studio, bộ seed đánh giá | Gắp 99 % hộp 3 cm, 98,6 % bóng 4 cm | `env.py`, `policies/`, `studio/`, `tools/` |
| 1 | Ghi dataset từ thầy giáo | Có `datasets/<tên>/`, mở xem được vài ván | `learning/record_dataset.py` |
| 2 | Train ACT bản nhỏ | Loss giảm đều, có checkpoint | `outputs/train/<lần chạy>/` |
| 3 | Đánh giá trên 774 seed | Có bảng thành công theo vùng như bảng của policy viết tay | `learning/eval_policy.py` |
| 4 | Vòng cải thiện: xem lỗi -> thêm dữ liệu -> train lại | Tỉ lệ tăng qua từng vòng | `learning/`, `datasets/` |
| 5 | Đa dạng hoá: nhiều cỡ, màu, ánh sáng; hộp + bóng | Gắp được cả cỡ chưa từng thấy (vd 3,5 cm) | `learning/configs/` |
| 6 (sau) | DAgger, RL tinh chỉnh, SmolVLA nhận lệnh chữ, chạy tay thật | | |

`learning/`, `datasets/`, `outputs/` sẽ được tạo ở chặng 1-2.

## 3. Thư mục nào làm gì

### Phần đã có

| Thư mục / file | Chức năng | Khi nào mở |
|---|---|---|
| `env.py` | "Thế giới": robot, bàn, vật, bát, vật lý MuJoCo. Mỗi `step(action)` = 1/30 giây | Muốn biết robot "thấy" gì (observation), luật thắng (vật vào bát) |
| `config.py` | Mọi con số dùng chung: vị trí, cỡ, giới hạn, vùng gắp | Muốn đổi một thông số |
| `box_sampling.py` | Seed -> vị trí vật. Cùng seed = cùng ván | Muốn hiểu vì sao seed X nằm ở đó |
| `assets/` | Mô hình 3D của robot và cảnh (MJCF + mesh) | Đổi cảnh, camera, vật |
| `policies/scripted_*.py` | Thầy giáo (policy viết tay) | Thầy làm mẫu sai ở ca nào đó |
| `policies/act_policy.py` ... | Khung bài tập của bản gốc, chưa chạy được. Ta dùng LeRobot thay | Đọc phần mô tả kiến trúc để hiểu ACT |
| `tools/` | Công cụ quan sát: replay, chấm seed, xem từng bước | Mọi lúc cần "nhìn" chuyện gì xảy ra |
| `workflows/` | Ghi / train / đánh giá kiểu bản gốc (HDF5, MLP) | Tham khảo, sẽ thay bằng `learning/` |
| `studio/` + `web/joint-lab/` | Sim Studio: máy chủ Python + giao diện | Xem replay, đặt cảnh trong Lab |
| `results/` | Kết quả quét, danh sách seed theo dõi | Xem seed nào đang được theo dõi |
| `docs/` | Tài liệu (file này nằm ở đây) | Khi bí |

### Phần sẽ thêm cho AI

| Thư mục / file | Chức năng | Ví von |
|---|---|---|
| `learning/lerobot_env.py` | Đổi dữ liệu của env sang "ngôn ngữ" LeRobot: `observation.state`, `observation.images.front`, `observation.images.wrist`, `action` | Phiên dịch viên |
| `learning/record_dataset.py` | Cho thầy giáo chạy nhiều ván, chỉ giữ ván thành công, ghi thành dataset | Quay video bài giảng |
| `learning/configs/*.yaml` | Công thức của một lần train: dùng dataset nào, mạng nào, bao nhiêu bước | Công thức nấu ăn |
| `datasets/<tên>/` | Dữ liệu đã ghi (ảnh, góc khớp, action). Không commit vì nặng | Bộ video bài giảng |
| `outputs/train/<lần chạy>/` | Checkpoint (bộ trọng số mạng đã học) + log loss. Không commit | Bộ não đã học xong |
| `learning/eval_policy.py` | Cho mạng tự chạy trên 774 seed, ra bảng thống kê + replay xem trong Studio | Bài kiểm tra cuối kỳ |

## 4. Luồng dữ liệu một vòng

```
thầy giáo (policies/scripted_hybrid.py) chạy trong env.py
   │  mỗi bước ghi: ảnh 2 camera + 6 góc khớp  ->  action thầy chọn
   ▼
datasets/hop4_v1/            (record_dataset.py, chỉ ván thành công)
   ▼
lerobot-train + configs/act_hop4.yaml
   ▼
outputs/train/act_hop4_v1/checkpoints/...
   ▼
eval_policy.py: mạng tự chạy 774 seed  ->  bảng thành công + replay
   ▼
xem ván hỏng trong Studio  ->  quyết định thêm dữ liệu gì  ->  quay lại đầu
```

## 5. Gặp chuyện này thì vào đâu

| Triệu chứng | Nguyên nhân hay gặp | Làm gì, ở đâu |
|---|---|---|
| Thầy giáo cũng hỏng ở ca đó | Dữ liệu ca đó không có (chỉ giữ ván thành công) | Sửa policy viết tay (`policies/`) hoặc chấp nhận; đo lại bằng quét 774 seed |
| Mạng hỏng ở **một vùng** (vd vòng ngoài r >= 37 cm) | Ít dữ liệu vùng đó | Ghi thêm ván ở vùng đó (`record_dataset.py`, chọn seed theo vùng) rồi train lại |
| Mạng hỏng ở **cỡ / màu chưa thấy** | Dữ liệu không đủ đa dạng | Thêm cỡ / màu vào lúc ghi (domain randomization), train lại |
| Mạng đi lệch dần rồi không sửa được | Lỗi cộng dồn: dữ liệu toàn ván đẹp, mạng chưa từng thấy cảnh "đang lệch" | DAgger: cho mạng chạy, hỏi thầy giáo cách sửa ở mỗi bước, thêm vào dataset |
| Loss train không giảm | Dữ liệu lỗi (action lệch ảnh), learning rate sai, chuẩn hoá sai | Mở vài ván trong dataset xem ảnh khớp action không; xem log trong `outputs/train/` |
| Loss train giảm nhưng gắp kém | Học thuộc (overfit) hoặc loss thấp không đồng nghĩa gắp được | Luôn đánh giá bằng tỉ lệ thành công; thử checkpoint sớm hơn; thêm dữ liệu |
| Tay rung, giật | Mỗi bước đoán một action rời rạc | ACT đoán cả chuỗi action (chunk); chỉnh độ dài chunk, bật làm mượt |
| Mạng gắp kẹp đúng nhưng nhầm vật | Nhiều vật, chưa có lệnh chữ | Ghi kèm mô tả nhiệm vụ; chặng 6 dùng SmolVLA |

Nguyên tắc: **sửa dữ liệu trước, sửa mạng sau**. Mỗi thay đổi đo bằng quét cả bộ seed, ghi số trước / sau.

## 6. Kiến thức cần, theo thứ tự

Mỗi dòng có từ khoá để tìm thêm (YouTube, blog, tài liệu LeRobot).

| # | Chủ đề | Cần hiểu tới mức | Từ khoá tra cứu |
|---|---|---|---|
| 1 | Python + numpy | Đọc được code trong `env.py`, `tools/` | "numpy array basics" |
| 2 | Robot tay máy | Khớp, góc khớp, đầu tay (end-effector), động học ngược (IK) | "forward inverse kinematics robot arm" |
| 3 | Mô phỏng MuJoCo | Model, data, step, contact | "MuJoCo tutorial python" |
| 4 | Học có giám sát | Mẫu (input -> nhãn), loss, train / validation, overfit | "supervised learning loss overfitting" |
| 5 | Mạng nơ-ron + PyTorch | Lớp, trọng số, gradient, optimizer, GPU | "pytorch 60 minute blitz" |
| 6 | CNN cho ảnh | Mạng đọc ảnh ra vector đặc trưng (ResNet) | "CNN explained", "ResNet18" |
| 7 | Imitation learning | Behavior cloning, lỗi cộng dồn, DAgger | "behavior cloning compounding error", "DAgger" |
| 8 | ACT | Transformer + đoán chuỗi action (action chunking) | "ACT action chunking transformers ALOHA" |
| 9 | LeRobot | Dataset, `lerobot-train`, đánh giá | "LeRobot tutorial", "LeRobotDataset" |
| 10 | Domain randomization | Đa dạng hoá cảnh để mạng tổng quát | "domain randomization sim2real" |
| 11 | RL (sau) | Phần thưởng, thử và sai, PPO / SAC | "reinforcement learning PPO explained" |
| 12 | VLA / SmolVLA (sau) | Mạng nhìn ảnh + đọc lệnh chữ -> action | "vision language action model SmolVLA" |

Mục 1-3 bạn đã chạm nhiều qua phần policy viết tay. Mục 4-9 học trong lúc làm chặng 1-3.

## 7. Từ điển nhanh

| Từ | Nghĩa |
|---|---|
| **policy** | Hàm "nhìn tình hình -> chọn hành động". Policy viết tay = luật do người viết; policy AI = mạng nơ-ron |
| **observation** | Thứ robot nhìn thấy mỗi bước: ảnh camera + góc khớp |
| **action** | Lệnh mỗi bước: góc khớp đích cho 5 khớp + hàm kẹp |
| **episode / ván** | Một lần thử từ đầu tới hết (tối đa 500 bước) |
| **seed** | Số quyết định vị trí vật. Cùng seed = cùng ván, để so sánh công bằng |
| **dataset** | Tập nhiều ván đã ghi để mạng học |
| **train** | Chỉnh trọng số mạng cho action nó đoán gần action thầy giáo |
| **loss** | Độ sai giữa đoán và đáp án. Giảm = đang học |
| **checkpoint** | Ảnh chụp trọng số mạng tại một thời điểm train |
| **epoch / step** | Một lượt đi hết dataset / một lần cập nhật trọng số |
| **overfit** | Học thuộc dữ liệu train, gặp ca mới thì kém |
| **generalization** | Làm tốt cả ca chưa thấy khi train |
| **behavior cloning** | Học bắt chước thầy giáo, kiểu học của ACT |
| **DAgger** | Cho học trò tự chạy, thầy sửa ở từng bước, thêm vào dữ liệu |
| **RL** | Tự thử, có điểm thưởng (vật vào bát) thì làm nhiều hơn |
| **domain randomization** | Đổi ngẫu nhiên cỡ, màu, ánh sáng, camera khi ghi để mạng không học thuộc một cảnh |
| **sim2real** | Mang thứ học trong mô phỏng ra robot thật |
| **ACT** | Mạng đọc ảnh + góc khớp, đoán một chuỗi ~20-100 action tới |
| **VLA** | Vision-Language-Action: mạng nhìn ảnh, đọc lệnh chữ, ra action (vd SmolVLA) |
| **vùng xanh / hỗn hợp / đỏ** | Vùng gắp được / gắp được một phần / không với tới (`config.PICK_GREEN`) |

## 8. Tra cứu thêm

- Bảng kết quả hiện tại và lệnh hay dùng: [`README.md`](../README.md).
- Cách mô phỏng và policy viết tay chạy: [`guide_vi.md`](guide_vi.md).
- Quy ước làm việc cho agent: [`AGENTS.md`](../AGENTS.md).
- [LeRobot](https://github.com/huggingface/lerobot) và tài liệu của nó: định dạng dataset, train, đánh giá.
- [Buttergipfeli/robotics](https://github.com/Buttergipfeli/robotics): ví dụ gọn nhất của đúng lộ trình này (thầy giáo sim -> ACT -> tay thật).
- [Soarm101_Mujoco](https://github.com/asdqwe9501/Soarm101_Mujoco): thầy giáo -> SmolVLA -> RL tinh chỉnh, có công cụ phân tích kẹp hỏng.
