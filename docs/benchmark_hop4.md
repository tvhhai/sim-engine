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

Replay đối chiếu 42 ván nằm ở `results/replays/box4-minjerk-review.html` trên máy chạy benchmark. Nguồn `box4-minjerk-review` cũng có thể mở trong Studio.

## Ghi demo

Đã chạy thử seed train `3000` qua `record_dataset.py` với camera `top,wrist`. Ván thành công với 146 khung, state và action đều 6 chiều. Lượt chấm 774 seed đo thầy ở lệnh điều khiển sạch trên MuJoCo. Đo ACT hay SmolVLA cần dùng checkpoint và bộ seed đánh giá của từng mô hình.
