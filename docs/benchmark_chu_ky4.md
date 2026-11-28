# Rút ngắn chu kỳ gắp vật 4 cm

Mốc trước sửa là policy `5a5513a`, bản tối ưu là `310710f`. Phép đo đếm bước điều khiển đến khi policy kết thúc, gồm cả đoạn về tư thế nghỉ. Cảnh thử có một vật 4 cm. Chấm đủ 774 seed mỗi loại trong MuJoCo 4.0.1.

| Vật | Thành công trước → sau | Seed rút được bước | Trung vị bước trước → sau | Trung bình bước trước → sau |
| --- | ---: | ---: | ---: | ---: |
| Hộp 4 cm | 774/774 → 774/774 | 774/774 | 174 → 154 | 184,80 → 159,37 |
| Bóng 4 cm | 774/774 → 774/774 | 774/774 | 175 → 154 | 186,43 → 158,73 |

Cả 1.548 ván đều thành công và rút được bước. Không có seed mới hỏng hay tăng số lần hạ hoặc khép kẹp. Tổng số bước giảm **13,76% cho hộp** và **14,86% cho bóng**. Mỗi ván hộp giảm 4–110 bước, mỗi ván bóng giảm 4–362 bước. Số bước trung vị đo theo từng nhóm đồ vật.

## Công thức và đối chiếu GitHub

Với một đoạn điểm–điểm có vận tốc và gia tốc đầu cuối bằng 0, nội suy minimum-jerk dùng `s(u) = 10u³ − 15u⁴ + 6u⁵`. Đỉnh đạo hàm của `s` bằng 1,875, nên công thức hiện tại lấy thời lượng tối thiểu theo vận tốc đỉnh đặt trước là `ceil(1,875 × quãng đường / tốc độ đỉnh mỗi bước)`. Nối vận tốc giữa các đoạn và giới hạn thay đổi lệnh khớp làm thời gian robot thực sự bám tới đích khác với thời lượng nội suy.

- **so101-act-pick-place** dùng cùng đa thức bậc năm cho các waypoint của tay SO-101. Đây là cách nội suy phù hợp để tạo chuyển động mượt giữa các đích. [Mã và mô tả quỹ đạo](https://github.com/muditbhalla/so101-act-pick-place)
- **so101-policy-lab** tách đưa ngang, hạ và nhả thành các chặng riêng. Repo dùng ngưỡng đến nơi rộng hơn cho waypoint đưa ngang và xác minh hộp vẫn được kẹp bằng hai hàm. Cách tách chặng cho phép chỉnh nhịp vận chuyển mà giữ chặng tiếp xúc thận trọng. [ScriptedExpert.tsx](https://github.com/noah-wardlow/so101-policy-lab/blob/main/src/controllers/ScriptedExpert.tsx#L1433-L1540)
- **Ruckig** giải thời gian tối ưu từ trạng thái đầu đến trạng thái đích theo giới hạn vận tốc, gia tốc và jerk. Bản sửa này tối ưu trong một tập nhịp điều khiển trên MuJoCo. Để kết luận tối ưu thời gian theo kiểu Ruckig cần giới hạn động học từng khớp đã hiệu chỉnh và kiểm tra sự ổn định của vật khi có tiếp xúc. [Ruckig](https://github.com/pantor/ruckig) · [Bài báo RSS 2021](https://www.roboticsproceedings.org/rss17/p015.pdf)

## Các thay đổi trong policy

- `ScriptedPolicyHybridMinJerk._choose_profile()` chọn cách gắp theo tiêu chí cũ rồi mới thử nhịp nhanh hơn cho cùng phương án. `_trial()` chạy trên bản sao MuJoCo, kiểm tra kết quả cuối, số bước, số lần hạ và khép kẹp. Chỉ dùng nhịp mới khi ván vẫn thành công, giảm số bước và số lượt thao tác không tăng.
- `policies/scripted_policy.py` và `policies/scripted_minjerk.py` tách giới hạn bước lệnh pha 6, đưa vật ngang, khỏi giới hạn chung của pha 5–7. Các mức thử là **0,018**, **0,014** và **0,012** mỗi bước trên thang lệnh chuẩn hóa, so với **0,010** mặc định. Pha nhấc và hạ tiếp tục dùng giới hạn đã hiệu chỉnh.
- Đoạn về nghỉ của min-jerk thử đỉnh lệnh khớp **0,17–0,20** mỗi bước, so với **0,12** mặc định. V1 có thể giảm làm mượt kép ở pha về nghỉ bằng hệ số 1,0 so với 0,8. V2 được thử đưa ngang ở **1,40–1,60 rad/s** và về nghỉ ở **2,25–2,50 rad/s**, so với **1,20** và **2,00 rad/s** trước đó.
- Nếu đổi nhịp vận chuyển và về nghỉ chưa rút được bước, thầy thử tăng tốc độ đỉnh riêng ở pha tiếp cận từ **30 lên 45 mm/bước**. Phương án cuối là thử lại một cách kẹp đã có, để tránh ván kéo dài do các lượt thử kẹp. Mọi phương án vẫn phải qua cùng phép chấm trước khi được chọn.
- Nếu chu kỳ bóng vẫn từ **300 bước** sau khi đổi nhịp, thầy so thêm V1 và V2 để giảm các lượt gắp lại kéo dài. Seed bóng 2278 giảm từ 500 còn **138 bước** và seed 2296 giảm từ 500 còn **211 bước** trong lượt chạy trực tiếp. Các phương án được chọn theo trạng thái mô phỏng.
- Giữ cùng bộ giải IK, mô hình vật lý, tiêu chí chấm và tần số điều khiển. Ván demo mới có thể ít khung hơn. Dự báo vật lý có thêm lượt thử trước khi ghi, nên giảm số bước của ván và giảm thời gian chạy CPU là hai phép đo khác nhau.

## Đối chiếu pha đưa ngang

| Vật và seed | Pha đưa ngang trước → sau | Cả ván trước → sau |
| --- | ---: | ---: |
| Hộp 87 | 31 → 18 | 178 → 165 |
| Hộp 133 | 137 → 80 | 430 → 373 |
| Bóng 8 | 80 → 46 | 201 → 168 |
| Bóng 2317 | 36 → 21 | 104 → 90 |

Nguồn Studio `box4-shorter-cycle` và `sphere4-shorter-cycle` có các cặp replay trên.

## Kiểm thử

`test_scripted_teacher_retimes_box_and_sphere` kiểm tra cả đường cũ và mới đều thành công, đồng thời đường mới có ít bước hơn ở seed hộp 87, 179 và bóng 2317. Hai test hạ tầng Lab vượt thời hạn chờ 30 giây khi chạy chung nhiều lượt quét. Tạm dừng worker quét rồi chạy lại bộ kiểm thử trên bản chốt, kết quả **22/22 đạt**.
