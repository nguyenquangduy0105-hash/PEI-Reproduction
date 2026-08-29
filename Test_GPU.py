import torch
print("="*40)
print(" KẾT NỐI VAST.AI THÀNH CÔNG ")
print("="*40)
print("PyCharm đang dùng Python tại:", torch.__file__)
print("Hệ thống có nhận diện GPU không:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Tên dòng card đồ họa của server:", torch.cuda.get_device_name(0))
    print("Phiên bản CUDA hiện tại:", torch.version.cuda)
print("="*40)