# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Mức chênh lệch. Tôi nghĩ fine-tune sẽ thắng sát nút baseline (b), vì (b) — base model
với prompt đã tối ưu — vốn đã là một mốc mạnh: 0.765 target, JSON hợp lệ 100%, nhanh gấp
3× baseline (a) chưa tối ưu prompt. Kết quả thật là fine-tune thắng đậm hơn nhiều dự
đoán: +0.205 điểm target (0.765 → 0.970), trong khi năng lực tổng quát gần như không đổi
(regression Δ -0.013). Tôi tưởng phần khó nhất là đánh bại (b), hoá ra khoảng cách còn
lại để tối ưu không hề nhỏ.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Không phải chỗ tôi dự đoán. Tôi nghĩ khó nhất sẽ là lúc train — lo OOM, lo Colab
disconnect giữa chừng. Thực tế train xong suôn sẻ; chỗ mất thời gian nhất lại là **sau**
khi train xong: notebook `RUN_ALL` có sẵn trong repo mặc định ô cấu hình
`EVAL_LIMIT=8`, nên lần chạy đầu chỉ chấm 8/50 mẫu mà không để ý. Tưởng đã xong, nhưng
`scripts/verify.py` báo FAIL ở check "full eval set used" — phải quay lại Colab chạy lại
NB2+NB5 với eval đầy đủ, tốn thêm ~40–45 phút chờ. Một dòng cấu hình mặc định dễ bỏ sót
hoá ra tốn thời gian hơn cả nỗi lo về GPU.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Trước đây nghĩ đơn giản: cứ fine-tune là chắc chắn tốt hơn base model, train xong là có
kết quả tốt hơn ngay. Giờ thấy rõ **prompt engineering một mình** đã kéo target từ 0.000
(baseline a, prompt sơ sài) lên 0.765 (baseline b, prompt có schema + ví dụ) — tức phần
lớn khoảng cách với base model đến từ cách hỏi, không phải từ việc train. Fine-tune chỉ
cộng thêm +0.205 nữa trên nền đó. Nếu lab không bắt đo baseline (b) trước khi train, tôi
sẽ dễ nhầm lẫn công của prompt engineering thành công của fine-tuning.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Dùng Claude Code xuyên suốt: đọc và tóm tắt toàn bộ tài liệu lab, setup môi trường local
(venv, `verify.py --smoke`), tạo bộ notebook chạy trên Kaggle (không có sẵn trong repo
gốc, chỉ có bản Colab), debug lỗi checksum eval bị lệch do Git trên Windows tự đổi
CRLF, và viết phần lớn `REPORT.md` dựa trên số liệu thật trong `results/`.

Hai chỗ nó sai đáng nhớ: **(1)** khi hướng dẫn chạy trên Colab, nó không cảnh báo trước
là notebook `RUN_ALL` có sẵn mặc định `EVAL_LIMIT=8` — dù chính nó đã đọc và tóm tắt bug
này (mục F-27 trong `SIMULATION-FINDINGS.md`) ngay từ đầu buổi; chỉ đến khi `verify.py`
báo FAIL nó mới phát hiện ra, khiến tôi phải chạy lại tốn thêm gần 1 tiếng. **(2)** khi
viết `scripts/qualitative_baseline_b.py` để so sánh với baseline (b), nó quên chưa push
file đó lên GitHub trước khi bảo tôi chạy trên Colab, nên lệnh chạy báo lỗi "No such file
or directory" — phải sửa lại (push file) rồi tôi mới chạy lại được.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Đóng băng một bộ eval đại diện cho task thật của khách hàng, rồi đo baseline với prompt
đã tối ưu kỹ trước khi train — để biết rõ ngưỡng cần vượt qua là bao nhiêu, tránh mất
công train mà không chứng minh được nó tốt hơn việc chỉ viết prompt tốt hơn. Đây là bài
học lớn nhất rút ra từ lab này: phần lớn khoảng cách với base model (0.000 → 0.765) đến
từ prompt engineering, không phải từ fine-tuning — nên trước khi cam kết chi phí train
với khách hàng, cần chứng minh được prompt engineering *không đủ*.

