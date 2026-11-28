# Policy cuối cho một vật 4 cm

Đã tích hợp chọn nhịp và điểm thả trong policy mặc định. Mỗi cảnh thử cách mang trên bản sao MuJoCo với cùng trạng thái đầu và dùng phương án ngắn nhất trong các ứng viên đã thử đáp ứng chấm thành công, có giữ vật, và không tăng lượt lặp pha, hạ hoặc khép kẹp so với thầy trước.

## Các thay đổi

- `ScriptedPolicyFinal` thử nhịp mang ngang 0,030 (bước lệnh khớp chuẩn hóa) hoặc 2,0 rad/s với V2. Khi phương án nhanh chưa đạt, thử nhịp 0,024 hoặc 1,8 rad/s với V2.
- Với hộp trượt ở lúc đặt, thử hạ đích thả 3 mm hoặc 6 mm. Ứng viên hạ 9 mm đi kèm ngưỡng đến đích 25 mm. Bảy ca hỏng sau khi tăng tốc trong thử nghiệm PPO đều có một phương án đặt đạt và nhanh hơn thầy cũ.
- Với hộp xa sát giới hạn pan, thử vòng cung 82 bước so với 102 bước. Chỉ dùng khi lượt mô phỏng đạt các điều kiện trên.
- Bộ lọc không nhận phương án làm tăng số khung chạm bàn, camera cổ tay hoặc bát trong pha 5–7. Đã chạy lại 41 ván có tiếp xúc hoặc dùng vòng cung ngắn sau khi bổ sung bộ lọc. So với lượt đề xuất nhịp chưa lọc, số ván có tiếp xúc giảm 35 → 22, riêng chạm camera giảm 14 → 1.
- Trong pha 5–7, lọc các ứng viên làm tăng số khung tay chạm bàn, camera hoặc bát so với thầy trước. Chạy lại 41 ván có tiếp xúc hoặc dùng vòng cung ngắn sau khi bổ sung bộ lọc. So với lượt đề xuất nhịp chưa lọc, số ván có tiếp xúc trong pha này giảm 35 → 22, riêng chạm camera giảm 14 → 1. Hộp 2096 dùng nhịp 0,024, gắp trong 169 bước và loại ba khung chạm camera của nhịp 0,030.

## Kết quả trên 774 seed mỗi loại

| Vật | Trước → cuối | Số bước trung bình trước → cuối | Trên 180 bước trước → cuối | Có pha lặp trước → cuối |
| --- | ---: | ---: | ---: | ---: |
| Hộp | 774/774 → 774/774 | 158,38 → 145,68 | 175 → 71 | 16 → 14 |
| Bóng | 774/774 → 774/774 | 157,38 → 144,68 | 177 → 49 | 4 → 4 |

- **Hộp.** 764 ván được rút bước. Số ván tăng bước, mới hỏng và mới thành công đều bằng 0. Tiếp xúc tay/bàn, camera hoặc bát ở pha 5–7 được ghi ở 3 ván.
- Nhịp và điểm thả được chọn ở Mang ngang 0.03 (754 ván), Mang ngang 0.024 (10 ván), Vòng cung 82 bước (6 ván), Hạ điểm thả 3 mm (4 ván), Hạ điểm thả 6 mm (2 ván), Hạ điểm thả 9 mm (1 ván).
- **Bóng.** Cả 774 ván đều được rút bước. Số ván tăng bước, mới hỏng và mới thành công đều bằng 0. Tiếp xúc tay/bàn, camera hoặc bát ở pha 5–7 được ghi ở 19 ván.
- Nhịp và điểm thả được chọn ở Mang ngang 0.03 (772 ván), Mang ngang 0.024 (2 ván).

## Bảy ca hụt ở thử nghiệm PPO

| Seed hộp | Bước trước → cuối | Phương án cuối |
| ---: | ---: | --- |
| 229 | 212 → 185 | Mang ngang 0.03, Hạ điểm thả 3 mm |
| 230 | 163 → 148 | Mang ngang 0.03, Hạ điểm thả 3 mm |
| 1014 | 177 → 165 | Mang ngang 0.03, Hạ điểm thả 6 mm |
| 2026 | 150 → 137 | Mang ngang 0.03, Hạ điểm thả 3 mm |
| 2087 | 152 → 148 | Mang ngang 0.03, Hạ điểm thả 9 mm |
| 2286 | 157 → 149 | Mang ngang 0.03, Hạ điểm thả 6 mm |
| 2379 | 152 → 144 | Mang ngang 0.03, Hạ điểm thả 3 mm |

Số bước gồm cả số bước chờ vật ổn định đã ghi trong Replay. Mốc trước lấy từ bảng chấm trước đó của cùng roster và cùng tiêu chí MuJoCo. Các lượt thử chọn phương án xảy ra trước khi tay chạy ván được ghi. Mức rút bước là trong nhóm phương án đã chạy qua kiểm tra, với tỷ lệ thành công đo trên mô phỏng 4 cm.

Studio lấy bộ hộp cuối làm nguồn mặc định, có bộ bóng riêng. Mỗi seed chỉ có một ván cuối. Studio chỉ tải khung hình của ván đang xem. Các lượt ghi demo mới qua policy mặc định sẽ dùng đường gắp này. Các lần huấn luyện ACT/SmolVLA trên bộ dữ liệu đã ghi tiếp tục dùng quỹ đạo trong bộ dữ liệu đó.

Đã kiểm tra Studio trả đúng 774 mục mỗi loại và tải đủ khung hình của seed 2499. Kiểm tra Playwright trên dữ liệu mẫu xác nhận đổi từ seed 133 sang 1016, trở lại seed 133 ở bước 42 và chỉ giữ một ván đầy đủ trong trình duyệt.

Chạy `python -m sim_engine.tools.final_replay` để tạo lại dữ liệu của bộ chuẩn. Chạy cùng lệnh với `--publish` để kiểm tra các ván cần rà soát và xuất danh sách. Dữ liệu Replay lưu tại `results/replays/`.

[Bảng 1.548 ván](../results/benchmark_final_grasp4.csv)
