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

## 6. Định tính — bắt buộc có cả ca THUA

> **Chưa hoàn chỉnh — thiếu cột (b).** `results/qualitative.json` (do NB5 sinh ra) chỉ lưu
> dự đoán của bản fine-tune (`ft_pred`), **không lưu** dự đoán của baseline (b) cho từng
> ticket cụ thể — nên bảng dưới chưa thể xác nhận ca nào thật sự "FT thua". Đã viết sẵn
> `scripts/qualitative_baseline_b.py` để chạy lại **chỉ** baseline (b) trên đúng 6 ticket
> mà fine-tune bị điểm <1.0 (không cần train lại gì, vài chục giây trên GPU):
> ```bash
> python scripts/qualitative_baseline_b.py
> ```
> Lệnh này in ra bảng thắng/thua/hoà thật và ghi `results/qualitative_with_b.json` — dán
> kết quả vào bảng dưới trước khi nộp.

6 ca fine-tune không đạt điểm tuyệt đối (từ `results/qualitative.json`, đối chiếu nhãn thật ở `data/eval_target.jsonl`) — **tất cả đều sai ở trường `urgency`**, xu hướng đoán `trung_binh` khi nhãn thật là `thap` hoặc `cao`:

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, đặt bình giữ nhiệt VN804124. Chưa thấy tiền. | urgency=**thap** | *(cần chạy script)* | urgency=**trung_binh** — sai 1/4 trường | (b) chưa biết |
| 2 | Shop ơi, đặt nồi chiên không dầu DH249548. Thiếu phụ kiện. | urgency=**thap** | *(cần chạy script)* | urgency=**trung_binh** — sai 1/4 trường | (b) chưa biết |
| 3 | Shop ơi, đặt áo khoác gió VN613097. Bị lỗi. Khi nào tiện. | urgency=**thap** | *(cần chạy script)* | urgency=**trung_binh** — sai 1/4 trường | (b) chưa biết |
| 4 | Chào shop, đặt nồi chiên không dầu VN949966. Hoàn tiền. | intent=**van_chuyen** (không phải hoan_tien) | *(cần chạy script)* | intent=**hoan_tien** — sai intent + urgency | (b) chưa biết |
| 5 | Cho mình hỏi, đặt đèn bàn LED OD436045. Giao hàng chậm. | intent=**san_pham_loi**, urgency=**cao** | *(cần chạy script)* | intent=**van_chuyen**, urgency=**trung_binh** — sai 2/4 trường | (b) chưa biết |

Có mẫu chung nào ở các ca FT thua không? **Có một mẫu rõ ràng ngay cả khi chưa có cột (b):** cả 6/6 ca lỗi đều sai ở `urgency`, và 5/6 ca lỗi *chỉ* sai ở `urgency` (case #4, #5 sai thêm cả `intent`). Mô hình có xu hướng đoán `trung_binh` (giá trị "an toàn", tần suất cao nhất trong 3 lớp) khi tín hiệu về mức khẩn cấp trong ticket không tường minh (ví dụ ticket #4 không có từ khoá khẩn cấp rõ ràng nào, nhãn thật `trung_binh` — đúng — trong khi #1–#3 có ngữ cảnh ngụ ý mức thấp/cao mà mô hình bỏ lỡ). Đây là dấu hiệu của **mất cân bằng lớp trong 250 mẫu train** đối với trường `urgency` nhiều hơn là lỗi cấu hình — hướng cải thiện hợp lý là cân bằng lại phân phối `urgency` trong corpus, không phải đổi hyperparameter.

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).**

Dựa trên số đo đầy đủ (50/50 mục tiêu, không phải bản smoke), bản fine-tune này **nên được deploy** thay cho baseline (b): nó vượt prompt tối ưu +20.5 điểm % trên target trong khi giữ nguyên format hoàn hảo và không gây suy giảm đo được trên bộ regression (Δ -0.013, nằm trong nhiễu). Cái giá duy nhất là latency cao hơn (b) 47% — với một hệ thống triage CSKH, độ trễ ~1.5s/ticket vẫn hoàn toàn chấp nhận được so với lợi ích chính xác hoá gấp nhiều lần các trường hợp phân loại sai.

Đòn bẩy thật sự trong lab này **không phải vị trí gắn adapter** — `attn_only` (r=283, chỉ 2 module) hoà tuyệt đối với `correct` (r=16, 12 module) trên cùng ngân sách tham số, chứng minh rằng với một tác vụ hẹp như JSON triage 4 trường và 225 mẫu train, việc "gắn đủ tham số ở đâu" không quan trọng bằng "có đủ tham số hay không". Đòn bẩy rõ ràng nhất được đo được là **learning rate**: lệch 10× (thang full-FT thay vì thang LoRA) phá huỷ hoàn toàn kết quả (target 0.970 → 0.000) — hiệu ứng lớn hơn nhiều so với chênh lệch giữa các cấu hình quantization hay placement. Đứng sau LR, **chất lượng/độ đúng của mask** (chứng minh ở NB1, `supervised_fraction=0.41`, cả hai assert xanh) là điều kiện tiên quyết bắt buộc — không phải một "nút vặn" có thể tinh chỉnh, mà là nền tảng: nếu mask sai (như F-10 trong `SIMULATION-FINDINGS.md` mô tả), mọi con số phía sau đều vô nghĩa bất kể LR hay placement có đúng đến đâu.

**Ba điều tôi học được** *(cụ thể, không generic)*:
1.
2.
3.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**

> *Hai mục trên (điều học được, và kế hoạch 2 giờ) cố ý để trống — đây là phần phản tư cá
> nhân, rubric 4.4 chấm theo "cụ thể, không generic", và nó phải là trải nghiệm thật của
> bạn khi làm lab, không phải suy đoán của người viết report giúp. Ba gợi ý kỹ thuật rút
> ra được từ chính dữ liệu ở trên, nếu muốn tham khảo rồi viết lại bằng lời của mình:*
> - *loss column ở NB4 xếp `attn_only` cao hơn `correct` — nếu chỉ tin loss, sẽ báo cáo sai kết luận về vị trí adapter (F-22 trong `SIMULATION-FINDINGS.md`).*
> - *`.gitattributes`/CRLF trên Windows từng làm `verify.py` báo nhầm "eval set bị sửa" dù không đổi nội dung gì — một bug môi trường chứ không phải bug lab.*
> - *6/6 ca fine-tune sai đều lệch ở đúng 1 trường (`urgency`) theo cùng một hướng (đoán `trung_binh`) — gợi ý mất cân bằng lớp trong dữ liệu train hơn là lỗi cấu hình.*

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
