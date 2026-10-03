# Báo cáo Lab 16 — AWS CPU + LightGBM

1. Tôi triển khai trên AWS tại `us-east-1`, dùng compute node `t3.medium`, Bastion `t3.micro` và source commit `55539f6`.
2. Dataset Credit Card Fraud Detection có 284.807 dòng, 31 cột và phân bố `Class` gồm 284.315 giao dịch thường, 492 giao dịch gian lận; dữ liệu được chia train/validation/test theo tỷ lệ 60/20/20, stratified với seed 42.
3. Thời gian đọc dữ liệu là 2,108 giây; thời gian training là 2,137 giây; early stopping chọn best iteration bằng 1.
4. Trên tập test, mô hình đạt AUC-ROC 0,938095; Accuracy 0,999070; F1 0,764444; Precision 0,677165 và Recall 0,877551.
5. Median latency dự đoán một dòng qua 100 lần đo là 1,348 ms; throughput batch 1.000 dòng qua 10 lần lặp là 615.134,25 dòng/giây. Phép đo chỉ gồm `predict_proba` khi dữ liệu đã nằm trong RAM.
6. CPU/RAM/Network được quan sát sau benchmark vào khoảng 00:14–00:16 ngày 04/10/2026 (UTC+7); ảnh bằng chứng nằm trong `screenshots/`.
7. AWS Billing tại 00:17 ngày 04/10/2026 (UTC+7) chưa cập nhật dữ liệu chi phí; ảnh dashboard đúng tháng nằm trong `screenshots/`. Chi phí thực tế có thể xuất hiện muộn sau khi tài nguyên đã được xóa.
8. Tôi đã tải `benchmark.py`, `benchmark_result.json` và terminal log về laptop; tài nguyên được `terraform destroy` lúc 00:57 04/10/2026 (UTC+7), xác nhận bằng ảnh `screenshots/aws-destroy-complete.png` và `terraform state list` không còn tài nguyên.
