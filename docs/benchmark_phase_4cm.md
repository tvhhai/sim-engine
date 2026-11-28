# Kiểm tra từng pha gắp hộp và bóng 4 cm

## Phạm vi

Mốc trước sửa là policy ở commit `2ad67cb`. Bộ chuẩn gồm 774 seed cho mỗi loại vật, từ 0–199, các seed cấu hình riêng và 2000–2499. Mỗi cảnh có một vật 4 cm. Một bước điều khiển tương ứng 1/30 giây. Số bước được đếm đến khi policy kết thúc, gồm cả thời gian về nghỉ. Kết quả thành công được xác nhận sau khi cho vật ổn định trong bát.

Đợt rút ngắn chu kỳ trước đã chạy toàn bộ bộ chuẩn, đánh giá tổng số bước và số lần hạ, khép kẹp. Đợt này ghi thêm từng đoạn pha và các đoạn con kéo vòng cung, kéo ngang. Một pha được tính là lặp khi policy đã rời pha đó rồi quay lại, kể cả khi chưa khép hàm. CSV ghi từng seed để tra các ván trên 180 bước và pha lặp.

## Bóng 2317 bắt đầu ở pha 3

Tại tư thế nghỉ, hai má kẹp chồng lên bóng trong mô hình. Đo ở trạng thái trước 50 bước vật lý ổn định cho kết quả độ xuyên sâu lớn nhất 17,7 mm phía cố định và 8,6 mm phía di động. Sau bước khởi tạo, bóng đã có vận tốc ngang 0,313 m/s. `ScriptedSphereIntercept` dự đoán vị trí gặp bóng bằng `xy_đích = xy_bóng + v_xy × (số_bước / tần_số)`. Pha 3 của phương án này gộp chuyển động tiếp cận, hạ tay và khép hàm. Khi đã kẹp được, policy chuyển thẳng sang pha 5 để nhấc. Cả ván mất 90 bước.

## Bóng 2278 và các cú kẹp cao

Mỗi bước điều khiển sau khi MuJoCo cập nhật tiếp xúc, lấy điểm chịu lực pháp tuyến lớn nhất của từng má trên bóng. Chỉ ghi khi cả hai má chịu trên 0,05 N. Độ lệch là `z_điểm_chạm − z_tâm_bóng`. Lấy mẫu lúc đầu tiên bóng đã lên ít nhất 10 mm so với vị trí sau khi khởi tạo. Dùng mốc một má cao từ 12 mm và má còn lại cao từ 3 mm để lọc các cú tương tự 2278 trên bóng bán kính 20 mm. Cột `high_lift_frames` đếm riêng các khung trong pha 3–6 mà bóng đã lên 10 mm và cả hai má cao từ 4 mm. Các mốc phục vụ đối chiếu hình học trong bộ seed này.

| Bóng 2278 | Hàm cố định | Hàm di động | Tỷ số lực tiếp tuyến / pháp tuyến | Bước |
| --- | ---: | ---: | ---: | ---: |
| Trước, V1 | +3,98 mm | +14,40 mm | 0,628 / 0,631 | 138 |
| Sau, min-jerk hướng kẹp theo bán kính | +1,01 mm | +4,64 mm | 0,171 / 0,171 | 138 |

Hai điểm tiếp xúc nằm cao làm cú kẹp phụ thuộc nhiều hơn vào lực ma sát. Bản cũ giữ được ở cấu hình chuẩn với hệ số ma sát tiếp xúc 2,0. Trong thử nghiệm riêng thay hệ số trượt của má kẹp và bóng, đường cũ hỏng ở 1,0, 0,6 và 0,4. Ở 0,8 đường cũ gắp lại rồi hoàn tất sau 275 bước. Đường gần tâm hoàn tất sau 136 bước ở cả bốn mức đó. Cả hai hỏng ở 0,25. Kết quả áp dụng cho cấu hình thử MuJoCo của seed này. [Đo lực tiếp xúc với MuJoCo](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-contactforce) · [Nón ma sát trong Modern Robotics](https://modernrobotics.northwestern.edu/nu-gm-book-resource/12-2-1-friction/)

Khi phát hiện cú kẹp cao trong lượt dự báo, policy thử các hướng kẹp theo bán kính và dịch điểm kẹp. Với 2278, công thức chọn hướng theo bán kính đi qua vật và dịch dọc trục kẹp 4 mm đã đưa hai điểm chịu lực về trong dải ±6 mm. Chỉ nhận phương án khi đặt được vào bát, không tăng số bước, số lần hạ, khép kẹp hay số lần vào lại pha. Với bóng trên 180 bước và có pha lặp, policy thử thêm các hướng kẹp và phương án V1/V2 hiện có để tìm đường gắp ngắn hơn hoặc giảm lặp.

Seed 205 quay từ pha 5 về pha 1 khi bóng rời xa tay trong lúc nhấc. Đường gắp mới giữ bóng ngay lượt đầu và giảm từ 293 còn 196 bước. Seed 2447 quay từ pha 3 về pha 1, kéo dài lần tiếp cận trước khi khép kẹp. Cho V1 tham gia đối chiếu giúp đi hết mỗi pha một lần, giảm từ 294 còn 201 bước.

| Seed có tiếp xúc cao | Hai má trước | Hai má sau | Bước trước → sau |
| --- | ---: | ---: | ---: |
| 84 | +3,01 / +13,01 mm | +0,88 / +4,71 mm | 177 → 170 |
| 214 | +4,06 / +13,39 mm | +1,21 / +4,59 mm | 167 → 165 |
| 2349 | +3,30 / +14,05 mm | +0,81 / +4,71 mm | 192 → 172 |

## Hộp 133 và vùng biên của khớp đế

Seed 133 ở góc 109,86° và cách trục đế 40,10 cm. Cuối pha 1, pan đạt khoảng +110° và cổ tay đã ở −87°. Điều kiện cũ kích hoạt kéo vòng cung khi hộp xa từ 37 cm và ở cách giới hạn pan dưới 3°. Điều kiện này chèn 102 bước vào pha 2 dù tư thế −87° có thể hạ kẹp được. Sau kéo vòng cung, đường cũ còn kéo hộp vào gần, thả và gắp lại.

Policy mới thử hạ từ roll thực đo ở cuối pha 1, kẹp trực tiếp và mang đi. Seed 133 giữ được roll −87° lúc hạ, dịch đích theo trục hàm 4 mm và gắp một lần. Chu kỳ giảm từ 373 xuống 213 bước. Pha 2 từ 110 bước tổng cộng xuống 3 bước, pha 6 mang ngang còn 85 bước.

Ở 133, hình chiếu tâm các mesh hàm di động theo tiếp tuyến hướng ra mép lớn hơn hàm cố định khoảng 47,7 mm ở cuối pha 1. Hộp 55 cũng có hàm di động mở về phía mép, khoảng 32,7 mm. Góc roll ở cuối pha 1 của 55 khoảng +43,6° và vật xa trục đế 41,43 cm. Thử giữ roll rồi gắp như 133 vẫn có pha lặp hoặc hỏng. Hộp 2486 nằm phía biên âm, hàm di động mở hướng vào trong khoảng 47,9 mm, vẫn gắp trực tiếp được trong 158 bước. Hai phía pan có giới hạn đối xứng ±110°. Thử hạ và mang vật với tư thế thực tế cho phép chọn đường gắp theo cả góc cổ tay và tầm với.

Ở seed 55 và 1018, phần kéo ngang sau vòng cung chờ đủ 120 bước vì tay dừng cách đích lần lượt khoảng 6,8 và 7,7 mm, ngoài ngưỡng đến nơi 6 mm. Policy thử kết thúc kéo sau 24 bước. Chỉ chọn nếu cả chu kỳ vẫn đặt được vật vào bát, rút được bước, và số lần hạ, khép hay lặp pha không tăng. Hai ván giảm từ 435 xuống lần lượt 342 và 345 bước.

Các thay đổi nằm ở bước chọn phương án và thời hạn kéo được kiểm tra trên bản sao MuJoCo. Công thức nội suy minimum-jerk, bộ giải IK, mô hình ma sát chuẩn và tiêu chí thành công được dùng chung cho cả hai lượt quét. Số bước trong báo cáo là độ dài ván được ghi. Các lượt dự báo bổ sung diễn ra trước khi ghi ván.

## Replay đối chiếu

- [Hộp 4 cm, seed 55, 133 và 2486](http://127.0.0.1:8765/studio/replay-view?source=file%3Abox4-phase-review)
- [Bóng 4 cm, seed 2278 và 2317](http://127.0.0.1:8765/studio/replay-view?source=file%3Asphere4-phase-review)

## Tổng hợp đủ bộ seed

| Vật | Thành công trước → sau | Trên 180 bước trước → sau | Có pha lặp trước → sau | Bước trung bình trước → sau |
| --- | ---: | ---: | ---: | ---: |
| Hộp 4 cm | 774/774 → 774/774 | 177 → 175 | 19 → 16 | 159,37 → 158,36 |
| Bóng 4 cm | 774/774 → 774/774 | 185 → 177 | 18 → 4 | 158,73 → 157,38 |

Số seed hỏng mới, mới thành công, tăng bước và phát sinh pha lặp đều bằng 0. Kết quả chạy lại mốc trước khớp với số bước và điểm thành công trong log lần rút ngắn chu kỳ trên cả 1.548 ván.

- **Hộp.** 8 seed rút được bước. Seed 1014 và 2486 xuống còn tối đa 180 bước. Seed 133, 243 và 1014 hết lặp pha. Trong 175 ván còn trên 180 bước, pha dài nhất là pha 6 ở 157 ván, pha 5 ở 9 ván, pha 2 ở 6 ván và pha 1 ở 3 ván. Hộp 1016 vẫn tốn 373 bước, trong đó pha 2 chiếm 118 bước và các pha 2–5 có lượt lặp.
- **Bóng.** 20 seed rút được bước. Seed 122, 152, 178, 201, 1015, 2014, 2349 và 2499 xuống còn tối đa 180 bước. Seed 122, 152, 178, 201, 202, 205, 209, 244, 1015, 2014, 2247, 2325, 2447 và 2499 hết lặp pha. Trong 177 ván còn trên 180 bước, pha dài nhất là pha 6 ở 146 ván, pha 5 ở 22 ván, pha 1 ở 7 ván và pha 4 ở 2 ván.
- **Tiếp xúc bóng.** Cả hai bản đều lấy được mẫu đủ hai má ở lần nhấc 10 mm cho 774/774 seed. Bốn seed kẹp cao theo mốc 3/12 mm là 84, 214, 2278 và 2349. Cả bốn ra khỏi nhóm sau sửa. Nếu xét mốc hai má đều cao 4 mm ở bất kỳ khung nhấc nào, số ván giảm từ 133 xuống 127.

Bảng CSV ghi 1.548 hàng, mỗi hàng là một seed và một loại vật, có các số đo ở hai bản. Cột `step_delta` âm là ván ngắn hơn. Cột `before_phases` và `after_phases` ghi thứ tự pha cùng thời lượng từng lần vào pha. Cột `max_phase_steps` cộng thời lượng mỗi pha trên tất cả các lần vào pha.

Hai cấu hình đều thực thi trọn ván từ trạng thái khởi tạo. Với cảnh chạy cặp, các lượt dự báo có cùng phương án, nhịp và trạng thái MuJoCo dùng lại kết quả. Ván chấm trước và sau chạy riêng.

Bộ kiểm thử Python đạt 22/22. API Studio trả đủ các replay đã liệt kê.

[Tải bảng đối chiếu từng seed](../results/benchmark_phase_4cm.csv)
