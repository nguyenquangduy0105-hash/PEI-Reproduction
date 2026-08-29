# PEI-Reproduction
Currently having weaker metrics count compare to original paper, looking into model's behavior, dataloader and trainer's logic

Chỉ cần chạy Trainer, các metadata đã có đầy đủ.
Có nhiều file .csv là do toàn bộ các file data đều được chứa ở thư mục cấp 1, bao gồm data gốc,data đã encode bằng vocab cũ từ .csv của repo và data encode bằng vocab mới

Nếu muốn sử dụng ERROR GT với correct=0, incorrect=1 thì giữ nguyên code, nếu mong muốn ngược lại thì đọc từ error_gt cột thứ 3, và vào ERROR_GT.py sửa CORRECT thành 1 và INCORRECT thành 0
