# PPO chọn phương án gắp vật 4 cm

Mạng chọn một phương án đầu ván từ tọa độ vật và bát, loại vật, vùng biên và vận tốc ban đầu. Thầy giải IK và điều khiển các pha. Bốn lựa chọn là thầy hiện tại, hai giới hạn mang ngang từ lượt thử 5 seed, và vòng cung 82 bước cho hộp sát biên. Lượt chấm chạy trực tiếp phương án PPO chọn trong MuJoCo.

Hai nhịp mang đã thử đặt giới hạn thay đổi lệnh khớp chuẩn hóa ở pha 6 là 0,024 và 0,030 mỗi bước với V1/minimum-jerk. Với V2, giới hạn vận tốc mang ngang tương ứng là 1,8 và 2,0 rad/s. Hai nhịp được áp dụng sau khi thầy chọn tư thế và cách kẹp cho cảnh.

## Huấn luyện và chọn checkpoint

- Train gồm 58 cảnh-vật ở dải 3000–5999. Validation gồm 26 cảnh-vật ở dải 2500–2999. Mỗi bộ lấy thêm hộp sát biên từ đúng dải seed của bộ đó. Train, validation và bộ chuẩn có seed rời nhau.
- Mỗi lựa chọn hợp lệ trên từng cảnh được chạy một ván MuJoCo để lưu thưởng. Vì PPO thực hiện một quyết định mỗi ván, mẫu lấy theo policy hiện tại nhận trực tiếp thưởng cuối ván từ bảng mô phỏng.
- Actor/critic MLP 32 nút, PPO 160 lần cập nhật, 512 mẫu/lần, 4 epoch, batch 64, learning rate 0,003 và clip ratio 0,2. Reward thất bại là −5. Ván đạt nhận số bước tiết kiệm so với thầy chia 100, giới hạn ±2, rồi trừ 0,5 cho mỗi lượt lặp pha hoặc khép hàm tăng thêm.
- Checkpoint được chọn ở lần cập nhật 5 theo số ván đạt, tổng bước và pha lặp trên validation. Năm seed dùng thử các ứng viên trước đó nằm trong bộ chuẩn và đã ảnh hưởng đến thiết kế tập hành động.

| Phương án trên validation | Thành công và từng giữ vật | Tổng bước gồm chờ |
| --- | ---: | ---: |
| teacher | 26/26 | 4149 |
| carry24 | 26/26 | 3940 |
| carry30 | 26/26 | 3818 |
| arc82 | 25/26 | 4311 |
| PPO | 26/26 | 3818 |

## Kết quả 774 seed mỗi loại

| Vật | Thành công trước → PPO | Bước điều khiển trung bình | Ván trên 180 bước | Ván lặp pha |
| --- | ---: | ---: | ---: | ---: |
| Hộp | 774/774 → 767/774 | 158,36 → 145,99 | 175 → 64 | 16 → 17 |
| Bóng | 774/774 → 774/774 | 157,38 → 144,66 | 177 → 49 | 4 → 4 |

- **Hộp.** 766 ván ngắn hơn, 6 ván dài hơn. Seed mới hỏng gồm 229, 230, 1014, 2026, 2087, 2286, 2379. Seed mới thành công gồm không có. Lượt chọn phương án gồm carry30 774.
- Trên 767 ván hộp đạt ở cả hai bản, số bước trung bình 158,29 → 145,93.
- **Bóng.** 774 ván ngắn hơn, 0 ván dài hơn. Seed mới hỏng gồm không có. Seed mới thành công gồm không có. Lượt chọn phương án gồm carry30 774.
- Trên 774 ván bóng đạt ở cả hai bản, số bước trung bình 157,38 → 144,66.

Checkpoint chọn cùng phương án `carry30` cho cả 1.548 ván. Vì vậy, kết quả trên bộ chuẩn tương đương với đặt cố định nhịp `carry30` sau khi thầy chọn cách kẹp. Tập validation chỉ có hai cảnh phù hợp với điều kiện thử vòng cung. Lợi ích của việc dùng PPO để chọn nhịp khác nhau theo từng cảnh vẫn cần được chứng minh. Ví dụ, hộp 1016 chạy 362 bước với PPO và 242 bước khi chọn vòng cung ngắn trong pilot.

Cả bảy ván hộp hỏng đều giữ vật tới pha 8. Ở khung cuối, tâm hộp vẫn gần tâm bát theo mặt phẳng XY (cách 8–18 mm) và cao hơn gốc bát khoảng 35–47 mm, sát hoặc vượt ngưỡng chấm chiều cao 35 mm. Seed 1014 và 2026 còn có tốc độ vật 0,079 và 0,515 m/s ở khung cuối, vượt ngưỡng nằm yên 0,05 m/s. Các trace cho thấy lỗi được ghi nhận ở khâu đặt và thả vật. Replay giữ đủ bảy ván hỏng cùng ván đối chứng tương ứng.

Mốc trước lấy từ lần quét policy mặc định đã ghi trong benchmark pha 4 cm. CSV giữ riêng số bước chờ vật ổn định. PPO chạy bằng công cụ thử nghiệm. Các workflow ghi demo, ACT và SmolVLA vẫn dùng policy mặc định.

[Kết quả từng seed](../results/benchmark_ppo_grasp4.csv) · [Benchmark pha trước](benchmark_phase_4cm.md)

## PPO với kiểm tra trước khi chạy

Sau khi PPO chọn, thử lựa chọn trên bản sao MuJoCo. Nếu thử không giữ hoặc không đặt được vật, dùng thầy gốc. Bảng dùng kết quả PPO trực tiếp ở các ván vượt kiểm tra và chạy lại đối chứng ở toàn bộ ván bị chặn. Kiểm tra chéo sáu ván được giữ để xác minh bản sao cho cùng kết quả.

| Vật | Thành công | Số ván quay về thầy | Bước trung bình | Ván trên 180 bước |
| --- | ---: | ---: | ---: | ---: |
| Hộp | 774/774 | 7 | 146,11 | 65 |
| Bóng | 774/774 | 0 | 144,66 | 49 |

Số bước trong bảng là bước của ván thực thi. Trước mỗi ván có thêm thời gian mô phỏng thử để kiểm tra lựa chọn.

Lượt chấm PPO thuần mất khoảng 42 phút với tám tiến trình CPU. Thời gian chấm gồm tìm tư thế của thầy và chạy ván, ngoài thời gian huấn luyện trên bảng thưởng. Cơ chế dự phòng cải thiện số bước điều khiển trong ván thực thi với chi phí chạy thêm bản thử MuJoCo.

## Replay

- [Hộp 4 cm, thầy trước PPO và kết quả PPO](http://127.0.0.1:8765/studio/replay-view?source=file%3Abox4-ppo-review)
- [Bóng 4 cm, thầy trước PPO và kết quả PPO](http://127.0.0.1:8765/studio/replay-view?source=file%3Asphere4-ppo-review)

## Tái lập

Dùng Python có MuJoCo, NumPy và PyTorch. Chạy từ thư mục cha của repo, lần lượt thay stage bằng `collect`, `fit`, `eval` và `guard-check` trong lệnh `python -m sim_engine.learning.ppo_grasp <stage>`. Mặc định dữ liệu chạy và checkpoint lưu ở `results/replays/ppo-grasp4/`.

[Bài báo PPO](https://arxiv.org/abs/1707.06347)
