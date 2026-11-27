# sim-engine: hướng dẫn cho agent

Mô phỏng MuJoCo cánh tay SO-101 gắp vật bỏ vào bát. Fork của `RajatDandekar/sim-engine`. Làm việc trên nhánh `my-work`, giữ `main` giống
upstream. Người dùng nói tiếng Việt: trả lời, comment và docstring viết tiếng Việt; tên biến, commit message viết tiếng Anh.

## Chạy và kiểm tra

Đứng ở thư mục cha của `sim-engine/` (gói Python tên `sim_engine`). Python: `C:/Users/Admin/.venvs/sim-engine/Scripts/python.exe`.

```bash
python -m pytest                                   # trong sim-engine/: test Python
python -m sim_engine.studio                        # Studio + Lab: http://127.0.0.1:8765
python -m sim_engine.tools.make_replay             # replay chuẩn -> results/replay.json
python -m sim_engine.tools.object_replay --kind sphere   # replay bóng (mặc định 4 cm)
cd web/joint-lab && pnpm install && pnpm build     # build Studio vào static/studio (luôn dùng pnpm)
pnpm test                                          # test giao diện (vitest)
```

## Bản đồ thư mục

| Đường dẫn | Vai trò | Ghi chú |
|---|---|---|
| `env.py`, `config.py`, `box_sampling.py`, `paths.py` | Env Gymnasium, hằng số, vị trí vật theo seed, đường dẫn | Mọi hằng số dùng chung ở `config.py` |
| `assets/` | MJCF + mesh SO-101, bàn, hộp, bát | Theo quy ước MuJoCo Menagerie |
| `policies/` | `scripted_policy.py` (V1), `scripted_v2.py`, `scripted_hybrid.py`, `scripted_minjerk.py` (mặc định) + khung MLP/ACT/Diffusion/SmolVLA | Policy viết tay = bộ sinh dữ liệu demo |
| `tools/` | CLI chẩn đoán: replay, chấm seed, quét vùng gắp, xem từng bước | `python -m sim_engine.tools.<tên>` |
| `workflows/` | Ghi demo (HDF5), train, eval | `python -m sim_engine record|train|eval` |
| `studio/` | Máy chủ Python của Studio (API, chạy Lab, lịch sử chạy) | |
| `web/joint-lab/` | Mã nguồn giao diện Studio (React, Mantine, Three.js) | Sửa ở đây, không sửa `static/studio/` |
| `static/studio/` | Bản build của `web/joint-lab` | Sinh ra bởi `pnpm build` |
| `static/replay/` | Trang replay canvas cũ, Studio nhúng qua iframe + `studio_bridge.js` | |
| `results/` | Dữ liệu sinh ra có commit: quét vùng, seed replay (`replay_pins.json`), seed theo dõi (`object_pins.json`) | `replay.*`, `replays/`, `studio/` chỉ ở máy |
| `datasets/`, `outputs/` | Dataset và checkpoint huấn luyện | Bỏ qua trong git |

Phần AI (`learning/`, chạy bằng `D:/vla/venv-lerobot`; dữ liệu và checkpoint ở `D:/vla` và Drive `MyDrive/vla`, không commit):

```
learning/
    sim_setup.py               # cầu nối env <-> LeRobot: camera, khung dữ liệu, action 6 <-> 7 số, bộ 774 seed chấm
    record_dataset.py          # thầy giáo (policy viết tay) ghi demo, chỉ giữ ván thành công
    train.py + configs/        # act_scratch / act_finetune (DAgger, lr giảm dần) / smolvla_finetune / diffusion_scratch
    dagger.py, merge_datasets.py   # một vòng DAgger (AI lái, thầy sửa), gộp dataset
    eval_policy.py             # chấm trên 774 seed, gộp nhiều checkpoint (trung bình action), ghi replay
    push_dataset.py            # đẩy dataset lên Hugging Face
    mini_local.ipynb, mini_smolvla.ipynb   # bản thu nhỏ ACT / SmolVLA chạy ở nhà ~5 phút, chú thích từng dòng
    colab/so101_act_colab.ipynb    # chạy thật trên Colab (ghi, train, DAgger, chấm, SmolVLA, đẩy HF)
```
Nhật ký thí nghiệm và kết luận: `docs/so_tay_ai.md`. Luôn chấm đủ 774 seed trước khi kết luận (100 seed lệch ±4 điểm).
Chọn checkpoint / mạng lái DAgger trên bộ validation (`eval_policy --seeds val`, seed 2500-2999); 774 seed chỉ dùng chấm cuối.
Chạy song song nhiều phần ghi demo / DAgger: phần k dùng `--seed-start S+k --seed-step K` (dải liền nhau sẽ trùng seed).

## Quy tắc làm việc

- **Đánh giá thay đổi policy bằng quét cả bộ seed**, không bằng một seed. Bộ chuẩn: seed 0-199, `config.BOX_FIXED_SEEDS` >= 200, 2000-2499
  (774 seed). Báo số trước/sau cho hộp và bóng, liệt kê seed mới hỏng và mới sửa.
- **Sửa theo yêu cầu thì giữ lại**, kể cả khi làm hỏng vài seed khác: ghi seed hỏng vào danh sách theo dõi rồi điều tra dần, không âm thầm
  bỏ thay đổi. Seed đã sửa vẫn phải xem lại được trong replay (`results/object_pins.json`, `results/replay_pins.json`).
- Điều tra lỗi theo `systematic-debugging`: dựng lại ván, in timeline phase / tiếp xúc (`tools/trace_episode.py`, `Recorder` trong
  `tools/make_replay.py`), tìm nguyên nhân gốc trước khi sửa.
- Thông số mới của policy là hằng số lớp có comment ghi lý do và số đo (seed, tỉ lệ trước/sau).
- Giới hạn Lab: tối đa 5 vật, cỡ 2–5 cm. Cỡ 5 cm dùng để thử nghiệm (hộp 5 cm đạt 24/45 trong phép quét trước). Lòng bát phải rộng
  hơn cỡ vật ít nhất 2 cm và chế độ gắp tất cả kiểm tra chỗ xếp một lớp theo kích thước từng vật. Đồng bộ `studio/scene.py` và
  `web/joint-lab/src/studio/scene.ts`.
- Nhiều agent có thể sửa cùng lúc: trước khi commit xem `git diff`, chỉ stage phần của mình (lọc hunk), không commit bản build lẫn việc dở
  của người khác.
- Commit nhỏ theo từng tính năng, push lên `my-work`. Không thêm dòng Co-Authored-By.

## Tham khảo

- [LeRobot](https://github.com/huggingface/lerobot): định dạng dataset, `lerobot-record`, `lerobot-train`, ACT / Diffusion / SmolVLA.
- [Soarm101_Mujoco](https://github.com/asdqwe9501/Soarm101_Mujoco): expert viết tay -> dữ liệu -> SmolVLA, công cụ phân tích kẹp hỏng.
- [SinfonIA SO-101 MuJoCo](https://github.com/SinfonIAUniandes/lerobot_robot_so101_mujoco): bọc sim thành `Robot` của LeRobot.
- [SO101 SimStudio](https://github.com/rocPAI-Forge/so101-simstudio): chia labs / configs / tools / tests.
- [so101-nexus](https://github.com/johnsutor/so101-nexus): bộ env nhiều nhiệm vụ đăng ký Gymnasium, PPO + BC.
- [Buttergipfeli/robotics](https://github.com/Buttergipfeli/robotics): expert -> ACT chỉ từ dữ liệu sim -> chạy trên SO-101 thật.
- [so101-policy-lab](https://github.com/noah-wardlow/so101-policy-lab): expert, ACT chạy trong trình duyệt, so sánh nhiều policy cùng cảnh.
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie): quy ước MJCF và mesh.

## CodeGraph và Graphify

- Quy tắc này ưu tiên hơn hướng dẫn chung trong skill Graphify.
- Tra định nghĩa, callers/callees và impact: dùng CodeGraph trước (context rồi một explore nếu cần). Nếu MCP không khả dụng, dùng Graphify hoặc tìm kiếm có phạm vi.
- Bản đồ tổng thể, cụm thành phần và đường liên hệ: dùng Graphify. Không truy cả hai công cụ khi một kết quả đã đủ.
- Chuỗi chữ, log và cấu hình: dùng rg/đọc đúng file; không bắt buộc truy graph.
- Graph dùng chung nằm trong sim-engine/graphify-out/graph.json khi đứng ở VLA; trong graphify-out/graph.json khi đứng ở sim-engine. Từ VLA thêm --graph sim-engine/graphify-out/graph.json vào query/path/explain.
- Dùng graphify query "<từ khóa>" --budget 1200, path hoặc explain; không nạp toàn bộ graph.json/report vào ngữ cảnh. INFERRED cần đối chiếu mã khi ảnh hưởng quyết định sửa.
- Sau một đợt sửa mã, trước khi dùng Graphify lại: chạy tools/update_graph.ps1 từ sim-engine (hoặc sim-engine/tools/update_graph.ps1 từ VLA). Chỉ AST cục bộ, không LLM; graph không tự cập nhật nền.
- Codex: $graphify, skill .codex/skills/graphify/SKILL.md. Claude: /graphify, skill .claude/skills/graphify/SKILL.md. Chỉ đọc skill khi cần workflow Graphify, không cho mọi câu hỏi code.
- graphify-out/ là cache không commit. Chưa lập chỉ mục ngữ nghĩa docs/PDF/ảnh. Không tự bật semantic extraction dùng API.
- Chi tiết và lệnh dựng lại: docs/code_navigation.md trong sim-engine.
