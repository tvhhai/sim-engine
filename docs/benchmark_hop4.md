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

## Giảm gắp lại cho hộp ở xa

Thầy ưu tiên phương án đã gắp thật và thả thành công chỉ với một lần hạ tay, một lần khép kẹp. Với hộp ở xa, thầy thử nhấc trực tiếp hoặc nhấc nhẹ rồi thu tay về khi vẫn giữ hộp. Phép thử diễn ra trên bản sao vật lý. Khi một lần gắp chưa đủ, thầy chọn phương án thành công có ít lượt khép và hạ nhất trong các phương án đã thử. Các cảnh đủ điều kiện kéo vòng cung sát giới hạn khớp 1 tiếp tục dùng cách chọn cũ.

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

Các số bước trong bảng đếm tới hành động cuối. Nguồn Studio `box4-single-grasp-review` cho xem 14 ván đối chiếu.

### Vì sao có ván thả vào bát vẫn bị chấm hỏng?

Điểm được chấm khi ván kết thúc. Với bát chuẩn, miệng bát ở z = 46,0 cm. Tâm hộp phải nằm trong lòng bát theo XY và thấp hơn miệng ít nhất 5 mm, tức z < 45,5 cm. Hộp phải rời hàm và vận tốc tịnh tiến dưới 0,05 m/s. Tiêu chí độ sâu loại các trường hợp hộp còn gác cao gần miệng bát.

Trong replay trước lần sửa đầu tiên, seed 227, 2047 và 2401 kết thúc với hộp vẫn chạm bát nhưng tâm cao lần lượt 45,6 cm, 45,8 cm và 45,9 cm. Seed 2401 có khung hình báo đạt ngay sau khi thả, rồi hộp đổi thế và tâm lên cao hơn ngưỡng. Seed 2474 từng báo đạt rồi nảy ra ngoài bát. Các lượt chấm dùng cùng tiêu chí này để so sánh quỹ đạo.
