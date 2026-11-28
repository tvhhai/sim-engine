# Thầy gắp hộp 4 cm

Mốc trước sửa lấy từ lượt chấm ở `4169d3e`, gồm 774 seed hộp 4 cm và 774 seed bóng 4 cm. Bản sửa ở `cf103c0`. Cả hai lượt dùng MuJoCo 4.0.1, cùng kích thước và vị trí vật theo seed, cùng hình học kẹp, bát, ma sát, tiêu chí vật đã được thả và nằm yên trong bát.

| Vật | Trước sửa | Sau sửa | Seed sửa được | Seed mới hỏng |
| --- | ---: | ---: | ---: | ---: |
| Hộp 4 cm | 753/774 | **774/774** | 21 | 0 |
| Bóng 4 cm | 774/774 | **774/774** | 0 | 0 |

753 seed hộp thành công trước đó và cả 774 seed bóng có số bước y hệt sau sửa. Bảng bên dưới ghi phương án sửa của toàn bộ 21 seed hộp từng hỏng.

## Cách sửa

Thầy thử quỹ đạo min-jerk mặc định trên bản sao MuJoCo. Nếu ván hỏng, thầy thử phương án rời tư thế nghỉ bằng cách điều khiển vai, khuỷu và cổ tay cùng lúc, sau đó thử đổi độ cao hoặc độ lệch của điểm kẹp. Quỹ đạo tìm được chạy lại trên cảnh thật của mô phỏng. Mỗi demo được ghi thành một ván liên tục. Phương án được chọn theo trạng thái vật lý hiện tại của cảnh.

Phần lớn các ca trước sửa bị mất điểm kẹp khi tiếp cận hoặc khép hàm. Ví dụ, hộp ở seed 244 dời vị trí khi cụm kẹp chạm ở pha lơ lửng. Seed 246 có tiếp xúc với cả hai hàm, nhưng các lần khép lại vẫn đẩy hộp ra ngoài tầm. Các ca như seed 204 giữ được hộp nhưng hết thời gian khi đưa sang bát. Chỉnh cách rời home làm thay đổi tư thế và nhịp tiếp cận ở toàn đoạn, nên kết quả tốt hơn ở 11 seed dưới đây chưa quy tất cả lỗi về pha rời home.

| Phương án thành công | Seed |
| --- | --- |
| Rời home bằng chuyển động đồng thời | `100, 204, 227, 248, 1016, 2005, 2062, 2169, 2266, 2388, 2462` |
| Nâng điểm kẹp 4 mm | `87, 2047, 2287, 2401` |
| Tăng độ lệch theo trục kẹp 4 mm | `244, 2100, 2300, 2325` |
| Hạ điểm kẹp 4 mm | `2474` |
| Hạ điểm kẹp 4 mm và tăng độ lệch theo trục kẹp 4 mm | `246` |

Nguồn `box4-history` trong Studio (nhóm *Đợt 1*, trước đây là `box4-minjerk-review`) giữ 6 ván đối chiếu của seed 227, 2047 và 2474. Bản đủ 42 ván nằm ở `results/replays/archive/before-cycle-cleanup/box4-minjerk-review.html` trên máy chạy benchmark.

## Ghi demo

Đã chạy thử seed train `3000` qua `record_dataset.py` với camera `top,wrist`. Ván thành công với 146 khung, state và action đều 6 chiều. Lượt chấm 774 seed đo thầy ở lệnh điều khiển sạch trên MuJoCo. Đo ACT hay SmolVLA cần dùng checkpoint và bộ seed đánh giá của từng mô hình.

## Giảm gắp lại cho hộp ở xa

Thầy ưu tiên phương án đã gắp thật và thả thành công chỉ với một lần hạ tay, một lần khép kẹp. Với hộp ở xa, thầy thử nhấc trực tiếp hoặc nhấc nhẹ rồi thu tay về khi vẫn giữ hộp. Phép thử diễn ra trên bản sao vật lý. Khi một lần gắp chưa đủ, thầy chọn phương án thành công có ít lượt khép và hạ nhất trong các phương án đã thử. Các cảnh đủ điều kiện kéo vòng cung sát giới hạn khớp 1 tiếp tục dùng cách chọn cũ.

### Thay đổi trong mã

- `ScriptedPolicyHybridMinJerk._choose_profile()` trong `policies/scripted_minjerk.py` trước đây trả về phương án thành công đầu tiên. Bản mới vẫn thử đường mặc định trước. Với một hộp 4 cm ngoài nhóm kéo vòng cung, nó ưu tiên kết quả cuối thành công, có kẹp vật thật, một lần hạ và một lần khép. Một lần hạ là chuyển pha 2 → 3. Một lần khép là lệnh hàm qua ngưỡng 0 theo chiều đóng trong pha 4. Khi cần dự phòng, các phương án thành công được xếp theo việc có kẹp thật, rồi số lần khép, rồi số lần hạ. Đây là tiêu chí chọn quỹ đạo của thầy. Tiêu chí chấm thành công của mô phỏng vẫn ở `env._get_info()`.
- `ScriptedBoxMinJerk` chọn cách mang hộp sau khi khép kẹp. Từ bán kính 40 cm, thầy có thể bỏ bước kéo, hoặc nhấc trước 2 cm rồi thu tay về khi còn giữ hộp. `ScriptedPolicy._drag_box()` trong `policies/scripted_policy.py` đọc độ cao nhấc từ `_drag_clearance_lift()`. Nếu phương án thu tay không giữ được hộp, nhánh nhả và gắp lại vẫn là phương án dự phòng.
- Công thức điểm kẹp cơ sở của hộp 4 cm cho độ cao đầu kẹp z = 43,5 cm trong hệ tọa độ world và độ lệch ngang 27 mm theo trục kẹp. Các phương án thay đổi tham số đưa vào công thức. Seed 87 và 100 chọn độ lệch **31 mm** rồi nhấc trực tiếp. Seed 245 chọn **z = 43,1 cm**, độ lệch **27 mm**, góc `wrist_roll = −100°` và nhấc trực tiếp. Các tham số được thử trên trạng thái vật lý của cảnh, không tra theo seed.
- Công thức nội suy minimum-jerk vẫn là `s(t) = 10t³ − 15t⁴ + 6t⁵` với `t` chuẩn hoá từ 0 đến 1. Bộ giải IK, cấu hình vật lý, camera và tiêu chí chấm giữ nguyên trong hai commit policy `c31fa61` và `5a5513a`. Thay đổi nằm ở đích, hướng kẹp và thứ tự các đoạn tay đi.
- Demo ghi mới với thầy này có thể ít đoạn nhả và gắp lại hơn. Định dạng camera, state/action và dữ liệu đã ghi giữ nguyên. Điểm benchmark đo policy viết tay. Hai commit không train hay sửa checkpoint ACT/SmolVLA.

### Kiểm chứng

Test `test_scripted_teacher_picks_far_box_without_regrasping` chạy lại seed 87, 100, 245 và kiểm tra thành công cùng đúng một lần hạ, một lần khép. Cả 21 bài kiểm thử Python đều qua.

Quét đủ 774 seed mỗi loại sau tối ưu. Hộp 4 cm và bóng 4 cm đều đạt 774/774. Trong 90 ca hộp có bán kính từ 40 cm và nằm ngoài nhóm phải kéo vòng cung, số ca gắp một lần tăng từ **1/90 lên 90/90**. Số ca phải nhả trong lúc kéo giảm từ 89 xuống 0, số bước trung vị giảm từ 236,5 xuống 185. Cả 10 ca thuộc nhóm kéo vòng cung có số bước y hệt mốc trước tối ưu. Policy chốt ở `5a5513a`.

| Seed | Lần khép kẹp trước → sau | Bước trước → sau |
| ---: | ---: | ---: |
| 87 | 4 → 1 | 310 → 178 |
| 100 | 3 → 1 | 305 → 167 |
| 245, bán kính 42 cm | 2 → 1 | 313 → 237 |
| 246 | 2 → 1 | 219 → 186 |
| 248 | 2 → 1 | 203 → 164 |
| 55, sát giới hạn khớp | 2 → 2 | 456 → 456 |
| 1022, sát giới hạn khớp | 2 → 2 | 369 → 369 |

Các số bước trong bảng đếm tới hành động cuối. Nguồn Studio `box4-history` (nhóm *Đợt 2*, trước đây là `box4-single-grasp-review`) giữ 6 ván đối chiếu của seed 100, 245 và 1022. Bản đủ 14 ván nằm ở `results/replays/archive/before-cycle-cleanup/box4-single-grasp-review.html`.

### Vì sao có ván thả vào bát vẫn bị chấm hỏng?

Điểm được chấm khi ván kết thúc. Với bát chuẩn, miệng bát ở z = 46,0 cm. Tâm hộp phải nằm trong lòng bát theo XY và thấp hơn miệng ít nhất 5 mm, tức z < 45,5 cm. Hộp phải rời hàm và vận tốc tịnh tiến dưới 0,05 m/s. Tiêu chí độ sâu loại các trường hợp hộp còn gác cao gần miệng bát.

Trong replay trước lần sửa đầu tiên, seed 227, 2047 và 2401 kết thúc với hộp vẫn chạm bát nhưng tâm cao lần lượt 45,6 cm, 45,8 cm và 45,9 cm. Seed 2401 có khung hình báo đạt ngay sau khi thả, rồi hộp đổi thế và tâm lên cao hơn ngưỡng. Seed 2474 từng báo đạt rồi nảy ra ngoài bát. Các lượt chấm dùng cùng tiêu chí này để so sánh quỹ đạo.
