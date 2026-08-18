# Kế hoạch trích scope Paper Rank B từ KLTN
### OT Temporal Alignment cho Audio-Visual Emotion Recognition

## 1. Timeline hội nghị Rank B khả thi

| Hội nghị | Hạn nộp | Trạng thái | Rank | Ghi chú |
|---|---|---|---|---|
| **SAC 2027** (ACM/SIGAPP) | **2/10/2026** | **Còn ~6 tuần — target chính** | CORE B | Cần chọn đúng track (Multimedia/AI-ML), Gwangju Hàn Quốc, 5-9/4/2027 |
| ICASSP 2027 | 16/9/2026 | Còn ~1 tháng — quá gấp | Rank A | Stretch goal, chỉ khả thi nếu code cực nhanh |
| ACII 2027 | ~Q1 2027 (dự kiến) | Còn nhiều tháng | Rank B | Backup an toàn, đúng chuyên đề Affective Computing |
| ICMI 2027 | ~Q2 2027 (dự kiến) | Còn nhiều tháng | Rank B | Backup an toàn, đúng chuyên đề multimodal |
| INTERSPEECH 2027 | 1/4/2027 | Còn nhiều tháng | Rank A | Phù hợp nếu nhấn mạnh phần audio/speech |

> Lưu ý: ICMI/ACII/ACM MM/ACM MM Asia bản 2026 đều đã đóng hạn tại thời điểm hiện tại (19/8/2026). SAC 2027 là lựa chọn cân bằng tốt nhất hiện tại — đủ thời gian hơn ICASSP nhưng sớm hơn nhiều so với chờ ACII/ICMI 2027.

## 2. Lưu ý quan trọng về SAC 2027

- SAC tổ chức theo **track riêng biệt**, không nộp vào "main track" chung — cần chọn đúng track (ví dụ Multimedia and Visual Computing, hoặc AI/Machine Learning) trên [sigapp.org/sac/sac2027](https://www.sigapp.org/sac/sac2027/).
- Mỗi track có thể có program committee và mốc thời gian phụ riêng dù chung khung ngày chính.
- Format: ACM SIG proceedings template, double-blind review, tối đa 8 trang + 2 trang phụ phí.
- Thông báo chấp nhận: 13/11/2026; Camera-ready: 28/11/2026.

## 3. Thu hẹp scope: từ KLTN xuống 1 paper nhỏ

KLTN đầy đủ có thể bao gồm: framework tổng quát, nhiều dataset, nhiều loại desync, so sánh nhiều fusion strategy. Một paper Rank B chỉ cần **một đóng góp rõ ràng, đo được, so sánh công bằng với baseline**.

**Tên paper đề xuất:**
> *Temporal-Aware Optimal Transport for Robust Audio-Visual Fusion under Modality Desynchronization*

**Scope hẹp:**

- Claim duy nhất: thêm temporal-distance term vào cost matrix OT giúp fusion audio-visual bền vững hơn khi 2 stream bị lệch pha (desync), so với OT chỉ dùng feature similarity (kiểu GRACE) và so với cross-attention/MulT.
- KHÔNG cần đề xuất kiến trúc mới phức tạp — chỉ cần 1 module `OTTemporalAlign` cắm vào pipeline chuẩn (đã có sẵn trong codebase), giữ encoder đơn giản.
- Thực nghiệm trọng tâm: **robustness curve** (accuracy vs. mức độ desync tăng dần) — đây là điểm mà CMOT/GRACE/DecAlign đều chưa đo, dễ tạo bảng/biểu đồ thuyết phục reviewer.
- 1 dataset chính (IEMOCAP hoặc CMU-MOSEI) + ablation alpha/beta là đủ, không cần nhiều dataset như KLTN đầy đủ.

## 4. Danh sách task để AI code (theo thứ tự ưu tiên)

| # | Task | Input cần từ bạn | AI code được luôn? |
|---|---|---|---|
| 1 | Viết loader IEMOCAP/CMU-MOSEI thật (thay `synthetic_dataset.py`) | Đường dẫn dataset đã tải + license access | Có, cần bạn cung cấp path/format file |
| 2 | Trích feature audio (wav2vec2) và visual (OpenFace/VideoMAE) | Chọn pretrained model cụ thể | Có, script trích feature offline |
| 3 | Cắm `OTTemporalAlign` vào pipeline, chạy baseline full model | Không cần thêm gì, dùng code đã có | Có, chạy ngay trên Kaggle |
| 4 | Cài đặt baseline so sánh: concat fusion, cross-attention fusion, GRACE-style OT (beta=0) | Không cần thêm | Có |
| 5 | Viết script tạo desync nhân tạo (random shift N frame) trên dataset thật | Không cần thêm | Có |
| 6 | Chạy ablation alpha/beta/eps, xuất bảng kết quả CSV + biểu đồ robustness curve | Không cần thêm | Có |
| 7 | Viết Related Work + Method section dựa trên báo cáo so sánh đã có | Duyệt lại claim novelty có đúng ý bạn không | Có, cần bạn review nội dung khoa học |
| 8 | Viết Experiments + Results section từ số liệu thực chạy | Kết quả thực nghiệm sau khi chạy xong task 6 | Có, sau khi có số liệu thật |
| 9 | Format theo template ACM SIG (SAC 2027), chuẩn bị submission | Chọn đúng track SAC phù hợp | Có |

## 5. Timeline rút gọn 6 tuần (target SAC 2027, hạn 2/10/2026)

- **Tuần 1** (19–25/8): Task 1-2 — chuẩn bị dataset thật + feature extraction, chốt dataset dùng chính thức.
- **Tuần 2** (26/8–1/9): Task 3-4 — chạy baseline, đảm bảo pipeline train ổn định trên Kaggle GPU, so sánh sơ bộ với các fusion baseline.
- **Tuần 3** (2–8/9): Task 5 — hoàn thiện script tạo desync nhân tạo, kiểm tra tính đúng đắn của mức shift.
- **Tuần 4** (9–15/9): Task 6 — chạy toàn bộ ablation alpha/beta/eps, xuất bảng + biểu đồ robustness curve.
- **Tuần 5** (16–22/9): Task 7-8 — viết bản draft đầy đủ dựa trên số liệu thật, review khoa học nội dung.
- **Tuần 6** (23–29/9): Task 9 — format theo template ACM SIG, chọn track, review nội bộ, nộp trước hạn (2/10) ít nhất 2-3 ngày để tránh trục trặc hệ thống submission.

## 6. Bước tiếp theo ngay bây giờ

Để bắt đầu Task 1 ngay trong tuần này, cần bạn cung cấp:
- Dataset nào bạn có sẵn quyền truy cập (IEMOCAP / CMU-MOSEI / MELD / khác).
- Đường dẫn/định dạng dữ liệu (đã tải về Kaggle Dataset, Google Drive, hay local).
- Backbone pretrained bạn muốn dùng cho audio (wav2vec2 / COVAREP) và visual (OpenFace / VideoMAE) ở Task 2.
- Xác nhận track cụ thể trên SAC 2027 bạn định nộp (Multimedia and Visual Computing hoặc AI/Machine Learning) để mình chuẩn bị đúng template và định hướng viết phù hợp với scope track đó.