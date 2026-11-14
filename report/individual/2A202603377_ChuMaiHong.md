# Báo cáo Day 7 — Chu Mai Hong

## Trạng thái các phần bắt buộc

- **Kết nối Day 6:** Hoàn thành. Day 7 dùng adapter tới embedding model, Chroma collection strategy B, chunk, ID, nguồn và metadata hiện có của Day 6. Không sửa bài, dữ liệu, index hoặc mã nguồn Day 6.
- **Thử nghiệm retrieval:** Hoàn thành so sánh dense, lexical và hybrid trên dev và bộ test mới; thử ngân sách context 800/3200 ký tự.
- **Agent:** Hoàn thành vòng search–answer/abstain, lọc section, kiểm tra citation theo context, chống lặp truy vấn và fallback về câu hỏi gốc.
- **Đánh giá:** Đã lưu trace, nhãn người chấm và metrics cho baseline lẫn cấu hình được chọn.
- **Báo cáo/chi phí:** Hoàn thành; chi phí là ước tính từ token usage, không phải hóa đơn thực tế.
- **Thử thách nâng cao:** Không làm theo yêu cầu.

## Dữ liệu và cấu hình

- **Nguồn:** OWASP Top 10 for LLM Applications 2025, [trang OWASP](https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/). PDF có 45 trang vật lý; Day 6 đã chọn 35 trang nội dung (7–13, 15–42), trích khoảng 78 nghìn ký tự. Quyền tái phân phối PDF chưa được xác minh; báo cáo không kèm bản sao.
- **Day 6:** strategy B, 124 chunk theo đoạn (43–799 ký tự; trung vị 705.5), 77,988 ký tự; `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384 chiều; Chroma PersistentClient cosine/HNSW, collection strategy B. Day 7 gọi lại đúng index qua adapter.
- **Agent:** OpenAI-compatible provider, `gpt-4o-mini`; cấu hình đọc từ `.env` cục bộ. API key không được ghi vào báo cáo hay kết quả nộp.
- **Benchmark dev:** 15 câu có đáp án của Day 6 và 2 câu Day 7-only không có bằng chứng (q21–q22).
- **Test cuối độc lập:** 5 câu có đáp án mới (q25–q29) và 2 câu không có bằng chứng (q30–q31), tại `starter/data/final_test_questions.json` trong working copy. Năm đoạn gold đã được đối chiếu với chunk Day 6. q30 hỏi hạn vá theo số ngày, q31 hỏi số lần gọi tool tối đa; rà soát toàn văn và các section liên quan không tìm thấy các ngưỡng này. Các câu q23–q24 và test cũ chỉ là thử nghiệm trước đó, không được dùng làm kết quả test cuối.
- **Định nghĩa retrieval:** Hit@1, Evidence Hit@3 và Context Evidence Rate được tính trên câu có đáp án (15 dev, 5 test); context rate là số câu có đoạn gold trong context cuối / tổng câu có đáp án. Agent grade 0–2 và citation support được chấm thủ công theo nội dung chunk, không chỉ theo ID.

## Thử nghiệm retrieval

Ba phương pháp dùng cùng corpus, top-k=3 và section equality filter khi câu hỏi có filter. Hybrid kết hợp score dense và TF-IDF theo tỷ lệ 0.6/0.4. Ngân sách context tính bằng ký tự.

| Split | Budget | Phương pháp | Hit@1 | Evidence Hit@3 | Context Evidence | Retrieval TB |
|---|---:|---|---:|---:|---:|---:|
| dev | 800 | Dense Day 6 B | 4/15 (26.7%) | 9/15 (60%) | 3/15 (20%) | 694 ms |
| dev | 800 | Lexical | 6/15 (40%) | 8/15 (53.3%) | 2/15 (13.3%) | 6.3 ms |
| dev | 800 | Hybrid | 5/15 (33.3%) | 9/15 (60%) | 3/15 (20%) | 21.0 ms |
| dev | 3200 | Dense Day 6 B | 4/15 (26.7%) | 9/15 (60%) | 9/15 (60%) | 1247 ms |
| dev | 3200 | Lexical | 6/15 (40%) | 8/15 (53.3%) | 8/15 (53.3%) | 13.3 ms |
| dev | 3200 | Hybrid | 5/15 (33.3%) | 9/15 (60%) | 9/15 (60%) | 55.9 ms |
| test cuối | 800 | Dense Day 6 B | 3/5 (60%) | 5/5 (100%) | 1/5 (20%) | 2654 ms |
| test cuối | 800 | Lexical | 2/5 (40%) | 4/5 (80%) | 1/5 (20%) | 1.5 ms |
| test cuối | 800 | Hybrid | 2/5 (40%) | 5/5 (100%) | 1/5 (20%) | 23.2 ms |
| test cuối | 3200 | Dense Day 6 B | 3/5 (60%) | 5/5 (100%) | 5/5 (100%) | 2537 ms |
| test cuối | 3200 | Lexical | 2/5 (40%) | 4/5 (80%) | 4/5 (80%) | 1.5 ms |
| test cuối | 3200 | Hybrid | 2/5 (40%) | 5/5 (100%) | 5/5 (100%) | 23.8 ms |

**Cấu hình được chọn trước khi chạy test cuối:** Dense Day 6 B, top-k=3, section filter khi có, context budget=3200 ký tự, tối đa 4 lượt search. Trên dev, tăng budget làm dense context rate tăng từ 3/15 lên 9/15 mà không đổi thứ hạng top-k. Trên test cuối, context rate tăng từ 1/5 lên 5/5. Retrieval latency dense test cao do khởi tạo model trong tiến trình riêng; đây là thời gian retrieval, không gồm thời gian gọi agent/API và không nên so trực tiếp giữa các lần chạy có cold start khác nhau.

## Agent và đánh giá thủ công

Agent dùng cấu hình dense đã chọn. Citation chỉ được chấp nhận nếu ID nằm trong context truy xuất; câu trả lời phải có citation inline. Prompt nhấn mạnh grounding, độ chính xác của số liệu và abstain khi không có bằng chứng. Nếu model lặp truy vấn, agent thử một lần với câu hỏi gốc rồi dừng, tránh vòng lặp vô hạn. Token usage và thời gian mỗi câu được lưu trong trace.

| Split / cấu hình | Câu | Grade 0/1/2 trên câu có đáp án | Mean /2 | Citation hỗ trợ | Answer có citation | Abstain đúng (no-evidence) | False abstain | Search TB/câu | Thời gian TB/câu | Chi phí ước tính |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev / 800 | 17 | 9 / 2 / 4 | 0.67 | 7/7 | 7/7 | 2/2 | 8/15 | 2.06 | 6.72 s | $0.00455 |
| dev / 3200 | 17 | 5 / 2 / 8 | 1.20 | 11/12 | 10/10 | 2/2 | 5/15 | 1.71 | 6.57 s | $0.00675 |
| test cuối / 800 | 7 | 4 / 0 / 1 | 0.40 | 1/1 | 1/1 | 2/2 | 4/5 | 1.86 | 8.11 s | $0.00154 |
| test cuối / 3200 | 7 | 1 / 1 / 3 | 1.40 | 4/4 | 4/4 | 2/2 | 1/5 | 1.71 | 10.85 s | $0.00300 |

Chi phí dùng snapshot giá chuẩn `gpt-4o-mini` **$0.15/1M input token, $0.075/1M cached input token, $0.60/1M output token**, theo [bảng giá OpenAI](https://developers.openai.com/api/docs/pricing/) (xem ngày 2026-11-14). Đây là ước tính, không phải số tiền hóa đơn; trace dev cũ không lưu riêng cached input nên tính toàn bộ input ở mức chuẩn, có thể cao hơn thực tế. Tổng 4 lượt benchmark khoảng **$0.01583**; không bao gồm lần gọi q31 lỗi ban đầu vì provider không trả usage.

### Nhận xét các câu test cuối ở cấu hình được chọn

| Query | Nhãn | Nhận xét |
|---|---:|---|
| q25 | 0 | False abstention dù context có đoạn SBOM trả lời trực tiếp. |
| q26 | 2 | Nêu đúng parameterized queries / prepared statements; citation hỗ trợ. |
| q27 | 1 | Nêu quyền đọc bảng `products`; câu trả lời chưa nói rõ cấm bảng khác và quyền ghi/xóa. Citation hỗ trợ phần đã trả lời. |
| q28 | 2 | Nêu đúng giới hạn $5000/ngày và tổng khoản vay $10,000; citation hỗ trợ. |
| q29 | 2 | Nêu đúng gián đoạn dịch vụ, thiệt hại kinh tế, model theft và service degradation; citation hỗ trợ. |
| q30 | Abstain đúng | Không tìm thấy hạn vá tối đa theo ngày trong tài liệu. |
| q31 | Abstain đúng | Không tìm thấy số lần gọi tool tối đa trong tài liệu. Lần chạy đầu bị `Connection error`; một retry giới hạn thời gian thành công và được dùng trong metrics. |

### Giới hạn và lỗi còn lại

- Dù retrieval test cuối đưa evidence vào context cho 5/5 câu có đáp án, agent vẫn từ chối nhầm q25. Đây là lỗi agent/answer generation, không phải thiếu đoạn gold trong retrieval context.
- Cấu hình 3200 cải thiện điểm agent trên test nhỏ (mean 0.40 lên 1.40; false abstention 4/5 xuống 1/5), nhưng chỉ có 5 câu có đáp án nên chưa đủ để kết luận chất lượng tổng quát.
- Dev selected có một citation không hỗ trợ (q06) và hai câu trả lời một phần; citation validity không đồng nghĩa toàn bộ câu trả lời chính xác.
- Latency và API cost phụ thuộc provider, cold start và cache. Chi phí trong báo cáo chỉ là ước tính token-based.

## Kết quả và lệnh tái chạy

Các file benchmark và trace/metrics chi tiết nằm trong `starter/data/` và `starter/results/` của working copy. Hai thư mục này bị `.gitignore` đánh dấu là dữ liệu cục bộ/đầu ra sinh tự động, nên không được đưa vào lần push này; các kết quả tổng hợp cần nộp được ghi trong báo cáo.

- Retrieval dev: `../../starter/results/retrieval-dev-budget800-cached/` và `../../starter/results/retrieval-dev-budget3200/`.
- Retrieval test cuối: `../../starter/results/retrieval-test-final-budget800/` và `../../starter/results/retrieval-test-final-budget3200/`.
- Agent dev: `../../starter/results/agent-dev-baseline800-final/` và `../../starter/results/agent-dev-final-v2/` (mỗi thư mục có `cases.json`, `traces.json`, `reviews.json`, `metrics.json`).
- Agent test cuối: `../../starter/results/agent-test-final-budget800/` và `../../starter/results/agent-test-final-budget3200/`.
- Log lần chạy lại q31: `../../starter/results/agent-test-final-budget3200-q31-retry/attempts.json`.
- Bộ test độc lập: `../../starter/data/final_test_questions.json`.

Ví dụ lệnh (đặt các biến môi trường trong `.env`, không commit API key):

```powershell
python starter/retrieval_experiments.py --split dev --budget 3200 --output starter/results/retrieval-dev-budget3200
python starter/retrieval_experiments.py --benchmark starter/data/final_test_questions.json --split test --budget 3200 --output starter/results/retrieval-test-final-budget3200
python starter/run_agent_benchmark.py --benchmark starter/data/final_test_questions.json --split test --budget 3200 --output starter/results/agent-test-final-budget3200
python -m unittest discover -s tests -v
```

## Kết luận

Các phần bắt buộc của Day 7 đã hoàn thành và có kết quả lưu kèm. Day 7 tái sử dụng nguyên trạng dữ liệu/index Day 6; không sửa bài Day 6. Dense Day 6 B với context budget 3200 là cấu hình được chọn trên dev và giữ nguyên cho test cuối. Retrieval đưa evidence tốt vào context, nhưng agent vẫn có false abstention và một citation không hỗ trợ, nên chưa nên xem là hệ thống trả lời đáng tin cậy để triển khai. Thử thách nâng cao được để lại đúng theo yêu cầu.
