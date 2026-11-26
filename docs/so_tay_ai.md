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

`learning/` đã có. Dữ liệu và checkpoint nằm ở ổ D (`D:/vla/datasets`, `D:/vla/outputs`) vì ổ C gần đầy.

### Công cụ và lệnh (máy hiện tại: GTX 1650 4 GB)

| Công cụ | Để làm gì | Ở đâu |
|---|---|---|
| Python 3.12 venv riêng cho AI | Không đụng venv của Studio | `D:/vla/venv-lerobot` |
| PyTorch bản CUDA 12.6 | Chạy mạng trên GPU | trong venv trên |
| LeRobot 0.7 (`lerobot[dataset,training]`) | Định dạng dataset, train ACT | trong venv trên |
| `learning/mini_local.ipynb` | Bản Colab thu nhỏ chạy ở nhà (~5 phút): ghi 3 ván, train 300 step, chấm vài seed, tự viết vòng lặp mạng lái | Học cách chạy trước khi đọc code |
| `learning/record_dataset.py` | Ghi dataset từ thầy giáo | chặng 1 |
| `learning/configs/*.yaml` + `learning/train.py` | Một file YAML = một thí nghiệm train | chặng 2 |
| `learning/eval_policy.py` | Bài kiểm tra trên bộ seed cố định | chặng 3 |

Cài lại từ đầu (PowerShell / Git Bash):

```bash
uv venv --python 3.12 D:/vla/venv-lerobot
uv pip install --python D:/vla/venv-lerobot/Scripts/python.exe --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu126 "torch==2.14.1+cu126" "torchvision==0.29.1+cu126" \
    "lerobot[dataset,training]==0.7.0" -e <đường dẫn tới sim-engine>
```

Chạy (dùng `D:/vla/venv-lerobot/Scripts/python.exe`):

```bash
python -m sim_engine.learning.record_dataset --episodes 200 --kind box --size 0.04 --root D:/vla/datasets/so101_box4_v1 --repo-id local/so101_box4_v1
python -m sim_engine.learning.train learning/configs/act_box4_v1.yaml
python -m sim_engine.learning.eval_policy --policy D:/vla/outputs/train/act_box4_v1/checkpoints/last/pretrained_model --limit 100
```

Số đo lần đầu: ghi 200 ván mất 21 phút (bỏ 7 ván hỏng), 34 823 khung hình, 182 MB. Train ACT 52 triệu tham số, batch 8:
2,8 bước / giây, GPU dùng 1,3 GB, 20 000 bước khoảng 2 giờ. Trên Windows, `torchcodec` chưa đọc được video nên LeRobot tự dùng `pyav`.

### Chạy trên Google Colab

Notebook dựng sẵn: [`learning/colab/so101_act_colab.ipynb`](../learning/colab/so101_act_colab.ipynb). Mở thẳng từ GitHub:
<https://colab.research.google.com/github/tvhhai/sim-engine/blob/my-work/learning/colab/so101_act_colab.ipynb>

Cần chuẩn bị: tài khoản Colab (Pro cho GPU A100 / L4), Google Drive còn ~1 GB trống. Không cần cài gì ở máy nhà.
Notebook tự làm: nối Drive (`MyDrive/vla`), clone nhánh `my-work`, cài LeRobot, ghi dataset (hoặc giải nén zip đã có trên Drive),
train, đánh giá 100 seed, chép replay lên Drive. Checkpoint lưu thẳng lên Drive nên Colab ngắt cũng không mất; ô 6b train tiếp.

Lưu ý: code Colab lấy từ GitHub, nên sửa code ở máy nhà phải **push** trước rồi chạy lại ô 3 (`git pull`).
Trên Colab không có màn hình, MuJoCo vẽ ảnh bằng `MUJOCO_GL=egl` **và** file khai báo driver EGL của NVIDIA (ô 3 tự tạo). Thiếu
file đó MuJoCo vẽ bằng CPU (`llvmpipe`): 0,85 s mỗi ảnh, ghi một ván mất 150 s thay vì 4,5 s.
Để qua đêm: dùng ô 9 (chạy nền bằng `nohup`), không xếp hàng ô 5-6-7: trang mất kết nối là hàng chờ ô bị mất.
Đo được (T4 High-RAM, 2,7 đơn vị / giờ): ghi 200 ván ~15 phút; train batch 32 được 1,7 bước / giây (30 000 bước ~4 giờ 50 phút).
A100 và L4 thường hết máy; T4 dễ có máy nhất.

### Kết quả các thí nghiệm ACT (hộp 4 cm, 2026-11-24/25)

Chấm trên 100 seed cách đều trong bộ 774 seed (policy viết tay: ~94 %; gần 99,6 %, xa 81,5 %). Chạy trên GPU nên mỗi lần chấm lệch
vài % (v1: 26 rồi 19 cùng một checkpoint); chênh dưới ~5 % chưa kết luận được.

| Thí nghiệm (config) | Đầu vào | Học (batch x bước) | Thành công | Gần r<37 | Xa r>=37 |
|---|---|---|---|---|---|
| `act_box4_v1` | camera trước + cổ tay | 8 x 20k | 19-26 % | 37 % | 3 % |
| `act_box4_wrist` | chỉ camera cổ tay | 8 x 20k | **3 %** | 3 % | 3 % |
| `act_box4_state` | toạ độ thật vật, bát (không ảnh) | 8 x 20k | 10-17 % (30 seed) | | |
| `act_box4_state_long` | toạ độ thật vật, bát | 32 x 40k | 39-42 % | 49 % | 19 % |
| `act_box4_topwrist` | camera trên cao + cổ tay | 8 x 20k | **36-39 %** | 40 % | 28 % |
| `act_box4_topwrist_noise` | như trên, dữ liệu có đẩy lệch | 8 x 20k | 19 % | 19 % | 19 % |
| `act_box4_topwrist` (Colab) | camera trên cao + cổ tay | 32 x 30k | **40 %** | 49 % | 22 % |
| `act_box4_topwrist_500` (Colab L4) | camera trên cao + cổ tay, **500 ván** | 32 x 60k | **59 %** | 71 % | 34 % |
| `act_box4_topwrist_500_dagger1` (Colab T4) | như trên, train tiếp trên 500 ván + **300 ván DAgger** | 32 x +20k | **74 %** | 90 % | 41 % |
| `act_box4_topwrist_500_dagger2` (Colab L4) | vòng 2: +300 ván DAgger (thầy min-jerk, sửa "hộp đã ở bát") | 32 x +20k | 68 % (last); **85 %** ở checkpoint 10k | 93 % | 69 % |
| `act_box4_topwrist_500_dagger3` (Colab L4) | vòng 3: từ checkpoint 10k vòng 2, +500 ván DAgger (tổng 500 + 1100) | 32 x +30k | 5k/10k/15k/20k/25k/30k = 69/73/53/**79**/73/77 % | 91 % (20k) | 53 % (20k) |
| gộp 3 checkpoint (d2 10k + d3 20k + d3 30k) | trung bình action, **774 seed** | | **84,8 %** | 91 % | 70 % |
| `act_box4_topwrist_mj` (Colab L4) | train từ đầu: 500 ván min-jerk mới + DAgger vòng 2-3, **774 seed** | 32 x 60k | 30k/40k/50k/60k = 57/49/**59**/48 % | 63 % (50k) | 50 % (50k) |

**Chấm đủ 774 seed** (100 seed quá nhiễu): vòng 2 ck 10k **602/774 (77,8 %)**, gần 86 %, xa 58 %; vòng 3 ck 20k 601 (77,6 %), gần 88 %,
xa 53 %; vòng 3 ck 30k 601 (77,6 %). Ba mạng ngang nhau; mạng chọn dùng: `MyDrive/vla/outputs/train/act_box4_topwrist_500_dagger2_colab/checkpoints/010000/pretrained_model` (vùng xa tốt nhất). Chỉ 455 seed đạt ở cả ba, 725 seed đạt ở ít nhất một, 49 seed hỏng cả ba.

Rút ra:
1. **Camera phải thấy vật ngay từ đầu ván.** Ở home camera cổ tay không thấy hộp: chỉ cổ tay 3 %. Camera trước thấy hộp ~6x6 điểm
   ảnh: 19-26 %, vùng xa 3 %. Camera trên cao (hộp ~12x12 điểm ảnh): 36-40 %, vùng xa 22-28 %.
2. **Train thiếu.** Checkpoint 5k / 10k / 20k của v1: 0 / 2 / 6 trên 30 seed. Bản toạ độ: 8 x 20k được 10-17 %, 32 x 40k được 39-42 %.
3. **Biết chính xác vật ở đâu vẫn chỉ ~40 %.** Lỗi còn lại ở điều khiển: AI chậm hơn thầy, nhả kẹp theo "thời gian" chứ chưa theo
   "vị trí" (seed 3000: nhả khi hộp còn cách tâm bát 2-3 cm, cao 4 cm). Cần thêm dữ liệu và học lâu hơn nữa.
4. **Đẩy lệch kiểu đơn giản làm tệ đi (36 -> 19 %).** ACT đoán chuỗi 50 action; trong dữ liệu có nhiễu, các action sau là phản ứng
   với những lần bị đẩy ngẫu nhiên tiếp theo, mạng không đoán được. Muốn dạy sửa sai thì nên dùng DAgger (cho AI tự chạy, thầy sửa
   ở chỗ AI lệch) thay vì đẩy ngẫu nhiên.
5. **Thêm dữ liệu có tác dụng rõ nhất tới giờ:** cùng camera, 200 -> 500 ván (và 30k -> 60k bước) đưa 40 % lên 59 %. Vùng xa vẫn
   kém (34 %): 17 ván kẹp được nhưng hỏng, 24 ván chưa kẹp được. L4 train 4,6 bước/giây (T4 1,75); 60k bước ~3,6 giờ.
6. **DAgger vòng 1 hiệu quả hơn thêm dữ liệu thường:** AI tự lái một đoạn ngẫu nhiên, thầy (bản min-jerk) tiếp quản và làm nốt, ghi
   300 ván "thầy sửa" (`learning/dagger.py`), gộp với 500 ván rồi train tiếp 20k bước: 59 % -> 74 % (gần 71 -> 90 %, xa 34 -> 41 %).
   Ván chưa kẹp được giảm mạnh (24 -> 5), ván kẹp được mà hỏng tăng nhẹ (17 -> 21): chỗ yếu giờ là mang / thả, nhất là vùng xa.
   23 seed mới đạt, 8 seed mới hỏng; bước trung vị của ván cùng đạt 130 -> 148 (thầy min-jerk đi chậm hơn). Ghi 300 ván + train
   20k trên T4 mất khoảng 5,5 giờ.
7. **Checkpoint cuối không phải checkpoint tốt nhất; điểm dao động mạnh.** Chấm mọi checkpoint: vòng 1 5k/15k/20k = 74/63/74, vòng 2
   5k/10k/15k/20k = 78/**85**/78/68. Cùng một vòng mà chênh tới 17 điểm, và giữa hai checkpoint có ~20 seed đổi đạt/hỏng: chấm một
   checkpoint trên 100 seed rất nhiễu (sai số chuẩn ~±4-5 điểm), train tiếp quá lâu với lr cố định làm tụt. Từ vòng 3 notebook tự chấm
   mọi checkpoint và vòng sau xuất phát từ checkpoint tốt nhất (`BASE_CKPT`), không lấy `last`.
8. **Vòng 3 không vượt vòng 2** (tốt nhất 79 % so với 85 %), dù thêm 500 ván và train lâu hơn; checkpoint 15k tụt hẳn xuống 53 %. Mức
   thật của các mạng tốt có lẽ quanh 75-80 %; 85 % của vòng 2 có thể một phần nhờ may trên 100 seed. Chấm lại trên đủ 774 seed để
   chọn mạng (xem dòng dưới bảng).
9. **Chấm 774 seed: DAgger đưa ACT lên ~78 %, vòng 3 không thêm được gì.** 85 % của vòng 2 trên 100 seed là may (774 seed: 77,8 %).
   Ba mạng tốt nhất cùng ~601/774 nhưng hỏng ở seed khác nhau (725 seed đạt ở ít nhất một mạng): lỗi còn lại phần lớn là "không ổn
   định" chứ không phải "không biết làm". Hướng tiếp: lr nhỏ hơn / giảm dần khi train tiếp, trung bình trọng số hoặc trung bình
   action của nhiều checkpoint, temporal ensembling; luôn chấm 774 seed (hoặc ít nhất 300) trước khi kết luận.
10. **Đi một mạch 50 action** (`--n-action-steps 50`) thường bằng hoặc tốt hơn đoán lại sau 20 bước; đoán lại sau 5 bước tệ hẳn.
11. **Gộp 3 checkpoint (trung bình action) đưa 77,8 % lên 84,8 %.** `eval_policy --policy A B C`: mỗi bước cả ba mạng cùng đoán,
    chạy trung bình action thang thật (không trung bình trọng số vì mỗi vòng chuẩn hoá theo thống kê dataset riêng). Ba mạng vòng 2 ck
    10k + vòng 3 ck 20k + ck 30k, 774 seed: **656/774 (84,8 %)**, gần 91 %, xa **70 %** (mạng lẻ tốt nhất: 602, gần 86 %, xa 58 %);
    kẹp được mà hỏng 58, chưa kẹp được 60 (mạng lẻ: 93 / 79). Theo số mạng lẻ đạt seed đó: 3/3 -> gộp đạt 440/455, 2/3 -> 139/169,
    1/3 -> 63/101, 0/3 -> 14/49. Bước trung vị ván đạt gần như cũ (175 so với 171). Cái giá: chạy 3 mạng mỗi bước (L4: ~3,3 s/ván
    so với ~1,2 s). Temporal ensembling (`--temporal-ensemble 0.01`) trên mạng lẻ thì tệ đi rõ (4/9 ván đầu) và chậm, đã dừng.
12. **Dữ liệu lẫn hai thầy.** 500 ván gốc ghi bằng `ScriptedPolicyHybrid` bản gốc (đi nhanh, trung vị ~130 bước), ván DAgger ghi bằng
    bản min-jerk (chậm, thả thấp, kéo vòng cung); DAgger vòng 1 còn chưa có bản sửa "hộp đã ở bát". Cùng cảnh hai lệnh khác nhau thì
    ACT học trung bình, có thể là một lý do mạng chập chờn. Thử train lại từ đầu trên dữ liệu một thầy: 500 ván min-jerk mới
    (seed 3000-4999) + DAgger vòng 2-3 (`learning/configs/act_box4_topwrist_mj.yaml`, notebook mục 11), xem mục 13.
13. **Train lại từ đầu trên dữ liệu một thầy KÉM HƠN: tốt nhất 59 %** (774 seed). Checkpoint 10k/20k/30k/40k/50k/60k =
    ~23 / ~29 (dừng chấm giữa chừng) / 57,0 / 49,2 / **59,2** / 48,4 %. Ck 50k: gần 63 %, xa 50 %; kẹp được mà hỏng 158, chưa kẹp được
    158 (mạng dagger2 ck 10k: 93 / 79); bước trung vị ván đạt 207 (dagger2: 171). Ngang mạng 500 ván gốc (59 % trên 100 seed), tức
    800 ván DAgger gần như không giúp gì khi train từ đầu. Lý giải:
    - Ván DAgger là "thầy sửa lỗi CỦA MỘT MẠNG CỤ THỂ" (cảnh do mạng cũ tự lái tới). Thêm vào khi fine-tune chính mạng đó thì trúng chỗ
      nó yếu; train mạng mới từ đầu thì mạng mới sai ở chỗ khác, các đoạn sửa đó lệch chỗ. Chuỗi cũ còn train tổng cộng nhiều hơn
      (60k + 20k + 20k + 10k step).
    - 800/1300 ván là đoạn "thầy tiếp quản" (nhấc lên rồi tiếp cận lại từ giữa chừng), chỉ 500 ván đầy đủ từ home: mạng thấy ít ví dụ
      đi một mạch từ home tới vật, nên vùng gần còn kém hơn cả mạng 500 ván gốc.
    - Điểm vẫn dao động mạnh giữa các checkpoint (49 / 59 / 48 %) dù dữ liệu chỉ một thầy: lẫn hai thầy không phải nguyên nhân chính
      của sự chập chờn; lr cố định 3e-5 tới cuối vẫn là nghi phạm.
    Kết luận: mạng tốt nhất vẫn là **gộp 3 checkpoint DAgger (84,8 %)**. Muốn dùng dữ liệu một thầy thì phải đi lại chuỗi DAgger từ mạng
    mới (mỗi vòng AI mới lái, thầy min-jerk sửa), không thay được bằng một lần train từ đầu.

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

### Thuật ngữ khi train và chấm

Đọc kèm một dòng log của `lerobot-train`, vd
`step:5K smpl:157K ep:997 epch:0.77 loss:0.136 grdn:8.68 lr:3.0e-05 data_s:0.003 updt_s:0.246 smp/s:128 l1_loss:0.114 kld_loss:0.002`.

| Từ | Nghĩa | Ví dụ ở dự án này |
|---|---|---|
| **khung (frame)** | Một bước của một ván: ảnh 2 camera + 6 góc khớp + 6 số action thầy chọn. Mạng học theo từng khung | Ván ~170 khung; dataset min-jerk ~1300 ván = ~204k khung |
| **batch / `batch_size`** | Số khung lấy ngẫu nhiên cho MỘT lần cập nhật trọng số. Lớn = hướng sửa ổn định hơn, tốn GPU hơn | 8 ở GTX 1650, 32 trên L4 |
| **step (bước train)** | Một lần: lấy 1 batch -> mạng đoán -> tính loss -> sửa trọng số. Khác "bước" của env (1/30 giây) | `steps: 60000` |
| **epoch (`epch`)** | Số lần mạng đã đi hết dataset = step x batch / số khung | 60k x 32 / 204k ~ 9,4 epoch |
| **`smpl`, `ep`** | Tổng số khung / số ván (quy đổi) mạng đã xem tới giờ | `smpl:157K` = 4,9k step x 32 |
| **learning rate (`lr`)** | Mỗi step sửa trọng số mạnh cỡ nào. Lớn: học nhanh nhưng loss nhảy, dễ "quên"; nhỏ: chậm mà chắc. Có thể cố định hoặc giảm dần về cuối | 3e-5 cố định; train tiếp lâu với lr cố định làm điểm tụt (vòng 3 ck 15k: 53 %) |
| **loss** | Sai lệch giữa action mạng đoán và action thầy. Loss train thấp KHÔNG có nghĩa gắp giỏi (xem "vòng kín") | `loss = l1_loss + 10 x kld_loss` |
| **`l1_loss`** | Trung bình trị tuyệt đối sai lệch của 50 action đoán (đã chuẩn hoá) | ~0,11 sau 5k step |
| **`kld_loss`** | Phần phạt của bộ mã hoá biến ẩn (VAE) trong ACT, giữ biến ẩn gọn để lúc chạy mạng không phụ thuộc nó | ~0,002, nhỏ là bình thường |
| **`grdn` (grad norm)** | Độ lớn tổng của hướng sửa ở step đó. Nhảy vọt bất thường = train không ổn định | ~9 |
| **`data_s` / `updt_s` / `smp/s`** | Thời gian chờ đọc dữ liệu / thời gian GPU tính một step / số khung xử lý mỗi giây. `data_s` lớn = nghẽn ở đọc video, tăng `num_workers` | L4: `updt_s` 0,25 s, ~4 step/s |
| **`num_workers`** | Số tiến trình đọc và giải mã video song song cho dataloader | 2 ở nhà, 4 trên Colab |
| **AMP (`use_amp`)** | Tính bằng số 16 bit thay vì 32 bit: nhanh hơn, ít bộ nhớ hơn, gần như không mất độ chính xác | `policy.use_amp: true` |
| **chuẩn hoá (normalization)** | Trước khi vào mạng, mỗi góc khớp / action được trừ trung bình, chia độ lệch chuẩn tính từ dataset; lúc chạy đổi ngược lại. Lưu trong `policy_preprocessor*` / `policy_postprocessor*` | Mỗi dataset có thống kê riêng, nên không trung bình trọng số được giữa hai checkpoint train trên hai dataset |
| **`chunk_size`** | ACT đoán một lần cả chuỗi action tới (action chunking) | 50 action ~ 1,7 s |
| **`n_action_steps`** | Làm bao nhiêu action trong chuỗi rồi mới nhìn ảnh, đoán lại. Các action chờ nằm trong hàng đợi (`_action_queue`) | 20 |
| **temporal ensembling** | Đoán lại MỖI bước, action chạy = trung bình có trọng số các chuỗi đoán chồng lên nhau (bài báo ACT) | Ở đây tệ hơn và chậm: đã bỏ |
| **checkpoint / `last`** | Bản lưu trọng số tại một step (`checkpoints/010000/`). `last` trỏ tới bản mới nhất, không phải bản tốt nhất | Vòng 2: `last` 68 %, ck 10k 85 % (100 seed) |
| **`pretrained_model/`** | Thư mục dùng được của checkpoint: `config.json` + `model.safetensors` + bộ chuẩn hoá. Đưa thẳng cho `eval_policy --policy` | |
| **resume** | Train tiếp từ checkpoint cuối sau khi bị ngắt (giữ cả trạng thái optimizer, step) | `lerobot-train --resume=true` |
| **train từ đầu / fine-tune** | Từ đầu: trọng số ngẫu nhiên (trừ ResNet18 đã học sẵn ImageNet). Fine-tune: bắt đầu từ một checkpoint đã train | DAgger 1-3: fine-tune; bản min-jerk: từ đầu |
| **vòng kín (closed loop) / lỗi cộng dồn** | Lúc chấm không có thầy: action của mạng đổi cảnh, cảnh mới vào lại mạng. Lệch nhỏ cộng dồn đưa tay tới cảnh chưa từng thấy | Lý do loss thấp mà vẫn gắp hỏng |
| **DAgger kiểu thầy tiếp quản** | AI lái `t_b` bước (15-250), thầy reset từ tư thế hiện tại và làm nốt; chỉ ghi đoạn thầy làm. `ai_ok`: AI tự xong (bỏ), `teacher_ok`: ghi, `teacher_fail` / `teacher_slow` (> 400 khung): bỏ | `learning/dagger.py` |
| **gộp dataset (merge)** | Nối nhiều dataset LeRobot thành một để train | `learning/merge_datasets.py` |
| **gộp checkpoint (ensemble)** | Chạy nhiều mạng cùng lúc, lấy trung bình action mỗi bước. Khác **trung bình trọng số** (gộp thành một mạng) | 3 mạng: 77,8 -> 84,8 % |
| **dữ liệu đa mốt (multimodal)** | Cùng một cảnh mà dữ liệu có hai cách làm khác nhau; mạng học trung bình hai cách, ra cách thứ ba sai | Hai thầy (gốc + min-jerk); hai mạng gộp lại đôi khi cũng vậy |
| **bộ chấm 774 seed** | Seed 0-199, seed cố định >= 200, 2000-2499; không trùng seed train (3000+) và DAgger (6000+) | `sim_setup.EVAL_SEEDS` |
| **gần / xa** | Vật cách trục khớp 1 dưới / từ 37 cm. Xa thì tay vươn gần hết, kẹp nghiêng nên khó | |
| **kẹp được mà hỏng / chưa kẹp được** | Hai kiểu hỏng: mang, thả kém / tìm, tiếp cận vật kém | `grasped_but_failed` / `never_grasped` |
| **`close_err_mm`** | Lần đầu mạng khép kẹp, đầu kẹp lệch tâm vật bao nhiêu mm (x, y, z) | Xem "kẹp vào không khí" |
| **nhiễu khi chấm** | Tỉ lệ đo trên n seed lệch khoảng ±sqrt(p(1-p)/n). p ~ 0,8: 100 seed ±4 điểm, 774 seed ±1,4 điểm | 85 % trên 100 seed thành 77,8 % trên 774 |

## 8. Tra cứu thêm

- Bảng kết quả hiện tại và lệnh hay dùng: [`README.md`](../README.md).
- Cách mô phỏng và policy viết tay chạy: [`guide_vi.md`](guide_vi.md).
- Quy ước làm việc cho agent: [`AGENTS.md`](../AGENTS.md).
- [LeRobot](https://github.com/huggingface/lerobot) và tài liệu của nó: định dạng dataset, train, đánh giá.
- [Buttergipfeli/robotics](https://github.com/Buttergipfeli/robotics): ví dụ gọn nhất của đúng lộ trình này (thầy giáo sim -> ACT -> tay thật).
- [Soarm101_Mujoco](https://github.com/asdqwe9501/Soarm101_Mujoco): thầy giáo -> SmolVLA -> RL tinh chỉnh, có công cụ phân tích kẹp hỏng.
