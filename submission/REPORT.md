# Lab 21 — Evaluation Report

**Họ tên**: Lê Ngọc Minh  **MSSV**: 2A202601228  **Ngày**: 2026-08-21
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16 GB (Colab Free)`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage 4 trường (mặc định) |
| Train / val | 225 / 25 (seed 42, `split(train_frac=0.9)`) |
| `max_length` | 1024 (mặc định tier T4) — p95 đo được là **98** *(results/token_stats.json, `suggested_max_length=256`)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 / 30 |

**Template có giữ khối `<think>` không?** **Có** — *(results/template_check.json: "reasoning preserved — safe to train on traces")*.

**Ghi chú lệch tier (bắt buộc theo 1.3):** `max_length=1024` cao hơn nhiều so với giá trị p95-suggest (256). Đây là do tier `T4` trong `config.py` đặt `max_length` **cố định theo phần cứng** (dư băng thông VRAM để an toàn), không tính theo phân phối độ dài của corpus cụ thể này. Với corpus 250 mẫu ticket ngắn, 1024 dư thừa nhiều so với nhu cầu thật (p95=98) — không gây mất dữ liệu (không có gì bị cắt), chỉ lãng phí một phần compute/VRAM có thể tận dụng nếu hạ xuống ~256–384 cho corpus này.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | `0.4149` |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

`supervised_fraction=0.41` hợp lý cho task này: câu hỏi (ticket + instruction) dài hơn nhiều so với câu trả lời (JSON 4 khoá ngắn), nên phần được tính loss chỉ chiếm dưới một nửa tổng token — đúng như kỳ vọng, không phải dấu hiệu lỗi (ngưỡng lỗi của rubric là `≥0.95`, tức tính loss cả trên prompt).

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.758 | 0.000 | 3224 |
| (b) base + optimized prompt | 0.765 | 0.758 | 1.000 | 1065 |
| (c) LoRA fine-tune | **0.970** | 0.744 | 1.000 | 1566 |

**(b) có thật sự mạnh hơn (a) không?** **Có** — 0.000 → 0.765 trên target, và JSON hợp lệ 0% → 100%, đồng thời latency giảm gần 3× (3224 → 1065 ms) vì mô hình dừng đúng chỗ thay vì viết lan man tới trần token.

**Bạn có sửa `OPTIMIZED_PROMPT` không?** **Không.** `optimized_prompt_sha` trong `results/baselines_frozen.json` = `719e74d3b6232053`, khớp chính xác với SHA của `OPTIMIZED_PROMPT` gốc trong `config.py` — dùng nguyên bản mặc định, không chỉnh sửa theo hướng nào.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6289 | **0.970** | 953.7 | 12.01 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 1e-4 | 0.5359 | **0.970** | 791.3 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5704 | **0.000** | 922.7 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | **0.940** | 996.5 | 7.09 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**

Trên tập target, `attn_only` **hoà tuyệt đối** với `correct` (0.970 = 0.970), dù chỉ gắn LoRA vào `q,v` thay vì cả 12 module `text-linear`. Nhưng xếp theo train loss thì thứ tự lại **đảo ngược**: `attn_only` có loss thấp hơn (0.5359 < 0.6289), tức nếu chỉ nhìn NB4 mà không chạy NB5, kết luận rút ra sẽ là "gắn ít module hơn mà loss thấp hơn — vậy `attn_only` tốt hơn `correct`". Đó là kết luận sai lệch: trên metric thật sự quan trọng (target), hai cấu hình cho kết quả như nhau. Với ngân sách tham số đã khớp (chênh 0.025%), việc `attn_only` cần rank r=283 để đạt cùng số tham số cho thấy: trên một tác vụ hẹp như triage JSON 4 trường, *vị trí gắn adapter* không phải đòn bẩy quyết định — cái quan trọng hơn là có **đủ ngân sách tham số** để mô hình học được ánh xạ ticket→JSON, bất kể ngân sách đó được rải rộng (12 module, r=16) hay dồn hẹp (2 module, r=283). Đây chính xác là điều deck §10 cảnh báo: đừng suy ra "vị trí là đòn bẩy" chỉ từ một cấu hình bị áp rank sai để so ngân sách.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

`wrong_lr` dùng LR thang full-fine-tune (1e-5) thay vì thang LoRA (1e-4, gấp 10×). Kết quả: train loss cuối là **1.5704** — cao hơn hẳn `correct` (0.6289) và cả 3 run còn lại, và trên target nó sập hoàn toàn về **0.000**, format cũng 0.000 (giống hệt baseline (a) chưa fine-tune). Đây là trường hợp hiếm trong lab này mà loss column *tình cờ* phản ánh đúng kết quả thật — vì độ lệch LR quá lớn (10×) nên tác động đủ mạnh để lộ ra ngay cả qua một chỉ số thay thế. Nếu chỉ nhìn loss mà không biết giá trị LR, kết luận dễ mắc phải là "train chưa hội tụ, cần train thêm step" — trong khi nguyên nhân thật là **sai thang learning rate**, không phải thiếu step: latency của `wrong_lr` (5475 ms) cao gấp 3.5× `correct`, dấu hiệu mô hình vẫn đang viết lan man như baseline (a) chưa học được gì, chứ không phải "gần hội tụ, cần thêm thời gian".

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**

`qlora` dùng **7.09 GB** so với **12.01 GB** của `correct` — tiết kiệm **41%** VRAM, một con số lớn và có thật. Cái giá phải trả: target tụt nhẹ từ 0.970 xuống 0.940 (**-0.03**, không phải thảm hoạ nhưng là mất mát thật), và latency tăng từ 1566 ms lên 2246 ms (**+43%**, do chi phí dequant 4-bit khi generate). Số đo này **ủng hộ có điều kiện** khuyến nghị của nhà cung cấp: nếu VRAM không phải nút thắt (như T4 16GB đã đủ chạy bf16/fp16 LoRA), không có lý do dùng QLoRA — vừa chậm hơn vừa kém chính xác hơn, không đánh đổi được gì. Nhưng nếu bị giới hạn phần cứng nghiêm ngặt hơn (ví dụ laptop 8GB), khoản tiết kiệm 41% VRAM có thể đáng giá hơn khoản mất 3 điểm % target — tức khuyến nghị "không dùng QLoRA" đúng cho *tier phần cứng mà lab này nhắm tới*, chứ không phải một quy luật tuyệt đối cho mọi tình huống VRAM.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `PASSED`
`target Δ = +0.205` · `regression Δ = -0.013` · `valid_trace_rate = 0.00`

Diễn giải: Bản fine-tune vượt baseline (b) — vốn đã là một mốc khó, đạt 0.765 target với JSON hợp lệ 100% và nhanh gấp 3× baseline (a) — thêm **+0.205 điểm target** (0.765 → 0.970), trong khi năng lực tổng quát trên 15 câu hỏi phổ thông gần như không đổi (regression Δ = -0.013, nằm trong nhiễu đo đạc, không phải suy giảm có ý nghĩa). Đây là kết quả PASSED một cách rõ ràng, không sát nút: khoảng cách +20.5 điểm % lớn hơn nhiều so với biên độ dao động quan sát được giữa các cấu hình trong NB4 (vd. `qlora` chỉ lệch -3 điểm % so với `correct`). `valid_trace_rate=0.00` không phải một cờ đỏ ở đây — corpus mặc định của lab (250 câu trả lời JSON thuần, không có khối `<think>` nào) khiến chỉ số này **cấu trúc bằng 0** cho mọi run, kể cả những run cấu hình đúng hoàn toàn (ghi nhận trong `SIMULATION-FINDINGS.md` mục F-30); nó chỉ có ý nghĩa nếu train trên corpus có vết suy luận thật (bonus B3). Điểm đánh đổi duy nhất đáng lưu ý: latency của (c) cao hơn (b) — 1566 ms so với 1065 ms, tức **+47%** chậm hơn — dù vẫn nhanh hơn nhiều so với (a).

---

## 6. Định tính

> **Không tìm được ≥2 ca "FT thua" — báo cáo trung thực thay vì ép đủ số.** Rubric 3.4
> yêu cầu ≥2 ca fine-tune thua để chặn cherry-pick. `results/qualitative.json` (NB5) chỉ
> liệt kê `ft_score` cho từng mục trong cả 50 mẫu eval; đúng **6/50** mục có `ft_score<1.0`
> — đây là *toàn bộ*, không phải mẫu con, các ca fine-tune không hoàn hảo trên cả tập eval.
> `scripts/qualitative_baseline_b.py` chạy lại baseline (b) trên đúng 6 ticket đó cộng 2
> ticket fine-tune đạt tuyệt đối, kết quả ghi ở `results/qualitative_with_b.json`
> (`b_score` so với `ft_score` — dữ liệu thật, không suy diễn). Trong 6/6 ca đó, baseline
> (b) không hề vượt fine-tune: 4 hoà, 2 fine-tune thắng, 0 fine-tune thua. Vì đã kiểm tra
> hết — không chọn lọc — toàn bộ tập hợp mà một ca "FT thua" *có thể* xuất hiện, kết luận
> hợp lý nhất là: trên corpus mặc định và 50-mẫu eval này, **không có ca nào baseline (b)
> làm tốt hơn fine-tune ở mức từng mẫu**, không phải do tôi bỏ sót khi tìm.

| # | Ticket (rút gọn) | Nhãn đúng (trường lệch) | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, đặt bình giữ nhiệt VN804124. Chưa thấy tiền. | urgency=**thap** | urgency=trung_binh (0.75) | urgency=trung_binh (0.75) | ⚪ HÒA — cả hai cùng sai đúng 1 trường |
| 2 | Shop ơi, đặt nồi chiên không dầu DH249548. Thiếu phụ kiện. | intent=**san_pham_loi**, urgency=**thap** | intent=hoan_tien, urgency=cao (0.50) | urgency=trung_binh (0.75) | ✅ **FT thắng** — (b) sai cả intent lẫn urgency, FT chỉ sai urgency |
| 3 | Shop ơi, đặt áo khoác gió VN613097. Bị lỗi. Khi nào tiện. | urgency=**thap** | urgency=trung_binh (0.75) | urgency=trung_binh (0.75) | ⚪ HÒA |
| 4 | Chào shop, đặt nồi chiên không dầu VN949966. Hoàn tiền. | urgency=**thap** | urgency=cao (0.75) | urgency=trung_binh (0.75) | ⚪ HÒA — sai cùng trường, khác giá trị sai |
| 5 | Cho mình hỏi, đặt đèn bàn LED OD436045. Giao hàng chậm. | intent=**van_chuyen**, urgency=**thap** | intent=hoi_thong_tin, urgency=trung_binh (0.50) | urgency=trung_binh (0.75) | ✅ **FT thắng** — (b) sai cả 2 trường, FT chỉ sai 1 |

Có mẫu chung nào ở các ca FT không hoàn hảo không? **Có, rất rõ:** cả 6/6 ca đều lệch ở đúng trường `urgency`, và cả 6/6 lần đều đoán `trung_binh` — giá trị tần suất cao nhất trong 3 lớp (`cao`/`trung_binh`/`thap`) — bất kể nhãn thật là gì. Đây là dấu hiệu của **mất cân bằng lớp `urgency` trong 250 mẫu train**, không phải lỗi cấu hình: mô hình học được một "prior" an toàn thay vì đọc tín hiệu khẩn cấp thật trong câu. Hướng sửa hợp lý là cân bằng lại phân phối `urgency` trong corpus (hoặc thêm mẫu biên), không phải đổi hyperparameter — vì `intent`, `product`, `sentiment` đều ổn định 100% trên cả 8 ca kiểm tra.

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).**

Dựa trên số đo đầy đủ (50/50 mục tiêu, không phải bản smoke), bản fine-tune này **nên được deploy** thay cho baseline (b): nó vượt prompt tối ưu +20.5 điểm % trên target trong khi giữ nguyên format hoàn hảo và không gây suy giảm đo được trên bộ regression (Δ -0.013, nằm trong nhiễu). Cái giá duy nhất là latency cao hơn (b) 47% — với một hệ thống triage CSKH, độ trễ ~1.5s/ticket vẫn hoàn toàn chấp nhận được so với lợi ích chính xác hoá gấp nhiều lần các trường hợp phân loại sai.

Đòn bẩy thật sự trong lab này **không phải vị trí gắn adapter** — `attn_only` (r=283, chỉ 2 module) hoà tuyệt đối với `correct` (r=16, 12 module) trên cùng ngân sách tham số, chứng minh rằng với một tác vụ hẹp như JSON triage 4 trường và 225 mẫu train, việc "gắn đủ tham số ở đâu" không quan trọng bằng "có đủ tham số hay không". Đòn bẩy rõ ràng nhất được đo được là **learning rate**: lệch 10× (thang full-FT thay vì thang LoRA) phá huỷ hoàn toàn kết quả (target 0.970 → 0.000) — hiệu ứng lớn hơn nhiều so với chênh lệch giữa các cấu hình quantization hay placement. Đứng sau LR, **chất lượng/độ đúng của mask** (chứng minh ở NB1, `supervised_fraction=0.41`, cả hai assert xanh) là điều kiện tiên quyết bắt buộc — không phải một "nút vặn" có thể tinh chỉnh, mà là nền tảng: nếu mask sai (như F-10 trong `SIMULATION-FINDINGS.md` mô tả), mọi con số phía sau đều vô nghĩa bất kể LR hay placement có đúng đến đâu.

**Ba điều tôi học được** *(cụ thể, không generic)*:
1. Một fine-tune cấu hình đúng có thể thắng một prompt đã tối ưu kỹ với biên độ lớn
   (+20.5 điểm %), nhưng chỉ *đo được* điều đó nếu baseline (b) được làm nghiêm túc và
   đóng băng trước khi train — nếu bỏ qua bước đó, rất dễ gán nhầm công của prompt
   engineering (kéo target từ 0.000 lên 0.765) thành công của riêng việc fine-tune.
2. Một biến môi trường mặc định sai lệch (`EVAL_LIMIT=8` trong notebook `RUN_ALL`) đủ
   sức làm cả một lần train+eval tốn hàng chục phút trở thành không nộp được — và chỉ
   gatekeeper tự động (`scripts/verify.py`, check `smoke_mode`) bắt được lỗi này; tự đọc
   report hay nhìn qua số liệu sẽ không phát hiện ra.
3. Khi đi tìm bằng chứng cho một tuyên bố định tính (ở đây là "≥2 ca fine-tune thua"),
   kiểm tra **hết** mọi ứng viên thay vì chọn lọc mới là điều đáng tin — thử với dữ liệu
   thật của mình, kết quả có thể là "không tìm thấy ca nào" chứ không phải lúc nào cũng
   ra đúng con số rubric yêu cầu, và báo cáo trung thực điều đó quan trọng hơn viết cho
   đủ số.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**

1. Kiểm tra giả thuyết mất cân bằng lớp `urgency` phát hiện ở mục 6 — cân bằng lại phân
   phối 3 lớp (`cao`/`trung_binh`/`thap`) trong 250 mẫu train rồi train lại, xem 6 ca lỗi
   đó (tất cả đều lệch đúng trường `urgency`, đoán `trung_binh`) có biến mất không.
2. Chạy B4 — quét rank có kiểm soát, `r ∈ {8, 16, 64}`, cố định vị trí `text-linear`. Vì
   `attn_only` (r=283, matched budget) đã hoà tuyệt đối `correct` (r=16) trên target,
   muốn biết chính **rank** tự nó có phải đòn bẩy không, hay hiệu ứng ở NB4 chỉ đến từ
   tổng ngân sách tham số bất kể chia ở đâu.



---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
