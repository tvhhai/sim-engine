# Thầy gắp bóng 4 cm

Bộ chấm gồm seed 0–199, các seed cố định từ 200 trở lên và 2000–2499. Mỗi cấu hình chạy trên cùng 774 cảnh cho từng loại vật. Mỗi ván giữ nguyên camera, vị trí vật, hình học kẹp, ma sát và tiêu chí vật đã được thả và nằm yên trong bát.

Lượt chấm dùng MuJoCo 4.0.1, hộp và bóng đều 4 cm. Mốc trước sửa lấy mã ở `820b70f`. Lượt bóng min-jerk gốc dùng `ScriptedPolicyMinJerk` làm nhánh chính và giữ V2 dự phòng của policy kết hợp. Lượt sau sửa dùng mặc định ở `46745a3`. Có thể chấm lại bản hiện tại bằng `python -m sim_engine.tools.object_replay --kind sphere --size 0.04` và lặp với `--kind box`.

| Loại vật và cấu hình đối chiếu | Trước sửa | Sau sửa với thầy mặc định | Seed sửa được | Seed mới hỏng |
| --- | ---: | ---: | ---: | ---: |
| Hộp 4 cm, thầy mặc định | 753/774 (97,29%) | 753/774 (97,29%) | 0 | 0 |
| Bóng 4 cm, thầy mặc định V1 + V2 | 763/774 (98,58%) | **774/774 (100%)** | 11 | 0 |
| Bóng 4 cm, min-jerk gốc + V2 | 752/774 (97,16%) | **774/774 (100%)** | 22 | 0 |

11 seed sửa được so với thầy bóng mặc định trước đó gồm `55, 152, 201, 205, 209, 1015, 1016, 1018, 2263, 2317, 2326`. Toàn bộ 22 seed của mốc min-jerk được liệt kê trong bảng replay bên dưới. 752 seed bóng vốn đạt với min-jerk gốc có số bước y hệt trong lượt chấm sau sửa.

21 seed hộp còn hỏng ở cả hai lượt gồm `87, 100, 204, 227, 244, 246, 248, 1016, 2005, 2047, 2062, 2100, 2169, 2266, 2287, 2300, 2325, 2388, 2401, 2462, 2474`. Cả 753 seed hộp thành công có số bước y hệt trước và sau.

## Quỹ đạo và tiếp xúc

Minimum-jerk tối thiểu hóa tích phân bình phương tốc độ biến thiên gia tốc của đầu tay theo điều kiện của bài toán. Mô hình của Flash và Hogan được kiểm tra với chuyển động tay tự do trong mặt phẳng. Với SO-101, việc khép hàm lệch tâm, ma sát và giới hạn khớp còn quyết định có giữ được bóng khi nhấc hay không. [Flash và Hogan, 1985](https://pmc.ncbi.nlm.nih.gov/articles/PMC6565116/)

ICL-so101 dùng máy trạng thái và kiểm tra tiếp xúc trước khi nâng. SO101 Real2Sim ghi nhận tăng ma sát hoặc lực kẹp vẫn không giải quyết được ca carton của họ trong MuJoCo. Hai kết quả là cơ sở để kiểm tra riêng lúc hạ, khép hàm, nhấc và chuyển ngang trong replay của bộ seed này. [ICL-so101](https://github.com/desmondzee/ICL-so101) · [SO101 Real2Sim](https://github.com/Yangjingyuan-HK/so101-real2sim)

Với bài một bóng 4 cm, thầy ưu tiên min-jerk. Nếu phương án cơ sở hỏng trên bản sao trạng thái MuJoCo, thầy thử đổi độ cao, hướng hoặc độ lệch của điểm kẹp. Khi bóng đã lăn nhanh lúc bắt đầu, thầy có thể đón ở điểm dự báo rồi nâng. Riêng seed 2317, lúc đặt cảnh hình học của bóng giao với hàm cố định khoảng 17,7 mm. Sau 50 bước ổn định vật lý của reset, bóng đã lăn khoảng 0,31 m/s. Thầy đón được bóng từ chính trạng thái đó. Lượt ghi demo thực thi trên cảnh gốc từ quan sát đầu tiên đến khi kết thúc ván.

## Ghi tiếp dữ liệu cho Colab

Ghi bóng vào một dataset riêng bằng `--kind sphere --size 0.04`, dùng dải seed train từ 3000. Giữ đúng cặp camera, thứ tự trường state/action, FPS và chế độ nhiễu của dataset hộp khi gộp để train tiếp. `record_dataset.py` lấy thầy mặc định và sinh task tiếng Anh có chữ `ball`. Đã chạy thử seed train 3000 với cặp camera `top,wrist`. Ván thành công, có 142 khung và giữ state/action 6 chiều.

Tỉ lệ trong bảng chấm đo thầy trên MuJoCo với lệnh điều khiển sạch. Dự báo vật lý tốn thêm CPU khi ghi demo. 774 seed này được dùng để tìm lỗi. Đo khả năng của ACT và SmolVLA cần chấm riêng theo loại vật và kiểm tra trên các cảnh ngoài bộ đã dùng để chỉnh thầy.

## Xem lại các seed min-jerk hỏng

Độ cao `+4 mm` hoặc `−4 mm` tính từ điểm kẹp ban đầu của thầy. Góc kẹp tính theo phương bán kính từ trục khớp 1 tới bóng. Mốc bước lấy từ timeline của Recorder. Va chạm trong một bước vật lý có thể kết thúc trước khung hình kế tiếp.

| Seed | Dấu hiệu ở min-jerk gốc | Phương án chạy lại |
| ---: | --- | --- |
| 8 | Cụm kẹp chạm ở bước 50 khi hạ. Sau khi nhấc 28 mm, bóng tuột và lăn ra ngoài tầm. | Nâng điểm kẹp 4 mm, 201 bước |
| 96 | Cụm kẹp chạm lúc hạ, bóng lăn sát vai. Chỉ giữ được bóng ở bước 469, ván hết 500 bước khi còn đang mang. | Đổi hướng kẹp sang 30°, 344 bước |
| 147 | Bóng đã được nhấc 49 mm rồi tuột trong pha chuyển ngang. | Hạ điểm kẹp 4 mm, 187 bước |
| 201 | Hàm động đẩy bóng lúc khép. Cả hai hàm báo tiếp xúc nhưng bóng chỉ nhấc được 2 mm rồi tuột. | Nâng điểm kẹp 4 mm, 241 bước |
| 209 | Bóng đã dời hơn 8 mm trong pha tiếp cận. Hàm động chạm khi khép, bóng tiếp tục lăn ra ngoài tầm. | Nâng điểm kẹp 4 mm, 285 bước |
| 210 | Cụm kẹp chạm bóng lúc hạ. Lần hạ lại kẹt gần vai, các lần khép đều hụt. | Hạ điểm kẹp 4 mm, 401 bước |
| 244 | Hai hàm chạm được bóng, bóng tuột ngay pha nhấc rồi lăn ngoài vùng gắp. | Hạ điểm kẹp 4 mm, 271 bước |
| 1022 | Gần giới hạn khớp 1. Nhiều lần chỉ hàm động chạm, cuối cùng bóng lăn khỏi bàn. | Nâng điểm kẹp 4 mm, 347 bước |
| 2020 | Bóng dời hơn 8 mm khi hạ. Lần khép sau chưa giữ được bóng trước khi bóng ra ngoài tầm. | Hạ điểm kẹp 4 mm, 162 bước |
| 2047 | Cụm kẹp đẩy bóng khi hạ, các lần hạ và khép lại không giữ được bóng trước mốc 500 bước. | Hạ điểm kẹp 4 mm, 365 bước |
| 2144 | Cụm kẹp chạm lúc hạ. Các lần hạ và khép lại không giữ được bóng, rồi bóng lăn ngoài tầm. | Chuyển sang V1, 213 bước |
| 2147 | Hai hàm kẹp được bóng nhưng tuột ngay pha nhấc, sau khi lên khoảng 15 mm. | Hạ điểm kẹp 4 mm, 184 bước |
| 2220 | Bóng lăn trong pha khép. Lần hạ lại bị kẹt, hết 500 bước. | Hạ điểm kẹp 4 mm, 164 bước |
| 2263 | Bóng ổn định qua pha nhấc, tuột đầu pha chuyển ngang. | Hạ điểm kẹp 4 mm, 199 bước |
| 2296 | Cụm kẹp đẩy bóng trong pha hạ, bóng lăn ra ngoài tầm sau lần khép lại. | Hạ điểm kẹp 4 mm, 500 bước |
| 2317 | Bóng đã lăn về phía trước từ đầu ván và rời vùng gắp trước khi xong pha nhấc tay khỏi home. | Đón bóng đang lăn, 104 bước |
| 2325 | Hai hàm có tiếp xúc nhưng bóng vẫn trên bàn khi tay đổi sang chuyển ngang. Gắp lại thất bại. | Nâng điểm kẹp 4 mm, 262 bước |
| 2326 | Bóng dời trong pha khép, hàm động chạm nhưng không ghi nhận được kẹp hai hàm. | Kẹp theo phương bán kính, 206 bước |
| 2347 | Nhấc được bóng rồi tuột trong pha chuyển ngang. | Hạ điểm kẹp 4 mm, 184 bước |
| 2420 | Bóng dời lúc hạ. Sau khi khép hụt, lần hạ lại hết thời gian và bóng lăn ngoài tầm. | Hạ điểm kẹp 4 mm, 163 bước |
| 2447 | Cụm kẹp đẩy bóng khi hạ ở bước 59–60. Hai lần khép lại hụt trước khi bóng ra ngoài tầm. | Nâng điểm kẹp 4 mm, 347 bước |
| 2450 | Cụm kẹp chạm ở bước 47 khi hạ. Bóng ra ngoài tầm ngay lần khép đầu. | Nâng điểm kẹp 4 mm, 162 bước |

Replay đối chiếu 44 ván nằm ở `results/replays/sphere4-minjerk-review.html` trên máy chạy benchmark. Mỗi seed trong bảng có một ván min-jerk gốc và một ván sau sửa.
