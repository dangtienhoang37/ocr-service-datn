import logging
import os
import cv2
from fastapi import FastAPI, File, UploadFile,Form
from io import BytesIO
from ultralytics import YOLO
from paddleocr import PaddleOCR
from datetime import datetime
import numpy as np

# Tắt log debug của PaddleOCR
logging.getLogger('ppocr').setLevel(logging.ERROR)

# Tạo ứng dụng FastAPI
app = FastAPI()

# Load mô hình YOLO đã huấn luyện
model = YOLO('E:/datn-AI-module/be-ocr/best.pt')

# Tạo instance OCR với mô hình lightweight
ocr = PaddleOCR(
    det_model_dir='PaddleOCR/inference/ch_ppocr_mobile_v2.0_det_infer',
    rec_model_dir='PaddleOCR/inference/ch_ppocr_mobile_v2.0_rec_infer',
    use_angle_cls=False,   # Tắt phân loại góc ảnh để tăng tốc độ
    lang='en'              # Chỉ nhận diện các ký tự tiếng Anh
)

# Tạo thư mục "img" nếu chưa tồn tại
img_folder = 'E:/datn-AI-module/be-ocr/img'
os.makedirs(img_folder, exist_ok=True)

@app.post("/upload/")
async def upload_image(id: str = Form(...), file: UploadFile = File(...)):
    # Đọc file hình ảnh từ request
    image_bytes = await file.read()

    # Chuyển đổi hình ảnh từ bytes thành array
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Phát hiện biển số xe với YOLO
    results = model.predict(source=image, save=False, imgsz=640)

    # Lưu hình ảnh vào thư mục img với tên là ID
    img_path = os.path.join(img_folder, f"{id}.png")
    cv2.imwrite(img_path, image)
    print(f"{datetime.now()} - Hình ảnh đã lưu tại: {img_path}")

    # Biến để lưu kết quả OCR nhận diện nếu có nhiều dòng
    license_plate_text = ""

    # Duyệt qua tất cả bounding box của YOLO
    for i, box in enumerate(results[0].boxes):
        # Lấy tọa độ bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box [x_min, y_min, x_max, y_max]

        # Cắt vùng biển số
        cropped_image = image[y1:y2, x1:x2]

        # Lưu ảnh biển số đã cắt vào thư mục "crop" với tên giữ nguyên và thêm chỉ số
        crop_path = os.path.join(img_folder, f"{id}_crop_{i}.jpg")
        cv2.imwrite(crop_path, cropped_image)
        print(f"{datetime.now()} - Biển số đã cắt lưu tại: {crop_path}")

        # Sử dụng PaddleOCR để nhận diện văn bản trong ảnh cắt
        result = ocr.ocr(crop_path, rec=True)

        # Duyệt qua các dòng văn bản đã nhận diện
        for line in result[0]:
            plate_text = line[1][0]  # Lấy văn bản nhận diện từ mỗi dòng
            license_plate_text += plate_text + " "  # Nối các phần lại với nhau

    # In kết quả OCR và thời gian hiện tại
    if license_plate_text.strip():  # Nếu có kết quả nhận diện
        print(f"{datetime.now()} - Biển số OCR nhận diện: {license_plate_text.strip()}")

    return {"message": "OCR processing complete", "license_plate": license_plate_text.strip()}

