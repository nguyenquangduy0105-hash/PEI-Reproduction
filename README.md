# PEI-Reproduction
- Currently having weaker metrics count compare to original paper, looking into model's behavior, dataloader and trainer's logic

- Chỉ cần chạy Trainer, các metadata đã có đầy đủ.
  
- Có nhiều file .csv là do toàn bộ các file data đều được chứa ở thư mục cấp 1, bao gồm data gốc,data đã encode bằng vocab cũ từ .csv của repo và data encode bằng vocab mới

- Nếu muốn sử dụng ERROR GT với correct=0, incorrect=1 thì giữ nguyên code, nếu mong muốn ngược lại thì đọc từ error_gt cột thứ 3, và vào ERROR_GT.py sửa CORRECT thành 1 và ERROR thành 0

- Hiện tại code đang sử dụng hàm build_decode, lấy từ thư viện pyctcdecode đã cũ và không thể decode ra sequence, có thể sử dụng beam search tự code (Trong Repo gôc sử dụng build_decode là beam search) hoặc sử dụng greedy decode

# ĐÁNH GIÁ METRICS :
- Trong kết quả Reproduce (Chạy bằng phiên bảo beam search tự code), model trả ra kết quả FRR, PER, DER cao hơn nhiều so với Paper gốc, F1 thấp hơn paper gốc 6% và FAR lại thấp hơn nhiều so với paper gốc. Có thể model bias theo canonical sequence ít hơn, nhưng hallucinate nhiều hơn, hoặc model đang trả ra rất nhiều phoneme sai
- Link đến bảng so sánh chi tiết : https://docs.google.com/document/d/1duOFT_tasYRWPHhkrJbxzOI_m7DY7o5-ACOSbzFchmI/edit?usp=sharing


