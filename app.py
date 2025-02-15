import logging
import os
import redis
import json
import cv2
from fastapi import FastAPI, File, UploadFile,Form,BackgroundTasks
import paho.mqtt.client as mqtt
import time
from threading import Thread
from io import BytesIO
from ultralytics import YOLO
from paddleocr import PaddleOCR
from datetime import datetime
import numpy as np
import cloudinary
import cloudinary.api
import cloudinary.uploader
from cloudinary.uploader import upload as cloudinary_upload

# Tắt log debug của PaddleOCR
logging.getLogger('ppocr').setLevel(logging.ERROR)

# Tạo ứng dụng FastAPI
app = FastAPI()
is_model_ready = False
#mqtt config 
def connect_mqtt():
    client = mqtt.Client()
    client.connect("test.mosquitto.org", 1883, 60)  # Thay đổi broker và port nếu cần
    print(f"config mqtt sucess")
    return client
# Kết nối Redis
redis_client = redis.StrictRedis(host='redis-service', port=6379, db=0,password='1', decode_responses=True)

# Hàm lưu biển số vào Redis
def save_license_plate_to_redis(id: str, license_plate: str):
    # Sửa key thành ocrCache:{id} để lưu vào cache với tên ocrCache
    key = f"ocrCache:{id}"
    
    # Lưu biển số vào Redis với key là ocrCache:{id}
    redis_client.set(key, license_plate)
    print(f"License plate {license_plate} saved to Redis with key {key}")

# Hàm push message vào MQTT topic
import json

def push_mqtt_message_ocr_done(id, parkingId):
    client = connect_mqtt()
    topic = "ocr:" + parkingId  # Topic mà bạn muốn push message

    # Tạo payload dưới dạng chuỗi
    payload = f"{id}"

    # Gửi message vào topic 'ocr'
    client.publish(topic, payload)
    print(f"Sent message: {payload}")


# Tạo một hàm tải mô hình OCR trong background
def load_ocr_model():
    print("Đang tải mô hình OCR... Vui lòng chờ...")
    global ocr
    ocr = PaddleOCR(
        det_model_dir='PaddleOCR/inference/ch_ppocr_mobile_v2.0_det_infer',
        rec_model_dir='PaddleOCR/inference/ch_ppocr_mobile_v2.0_rec_infer',
        use_angle_cls=False,   # Tắt phân loại góc ảnh để tăng tốc độ
        lang='en'              # Chỉ nhận diện các ký tự tiếng Anh
    )
    print("Mô hình OCR đã sẵn sàng!")

# Hàm tải mô hình YOLO
def load_yolo_model():
    global model
    print("Đang tải mô hình YOLO... Vui lòng chờ...")
    model = YOLO('./best.pt')
    print("Mô hình YOLO đã sẵn sàng!")

# Cloudinary config
cloudinary.config(
    cloud_name="dll0eotnd",
    api_key="284869121846298",
    api_secret="A1UJDl8OPaOUPA4jCQdeF2MAuqQ",
    secure=True,
)
# Hàm ping Cloudinary
def ping_cloudinary():
    while True:
        try:
            # Thực hiện ping Cloudinary
            cloudinary.api.ping()
            print("Kết nối Cloudinary đã sẵn sàng")
        except Exception as e:
            print(f"Không thể kết nối với Cloudinary: {str(e)}")
        time.sleep(30)  # Chờ 30 giây trước khi ping lại

# Hàm kiểm tra mô hình YOLO và OCR
def test_models():
    # Kiểm tra mô hình YOLO
    try:
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)  # Tạo ảnh giả để kiểm tra YOLO
        results = model.predict(source=dummy_image, save=False, imgsz=640)
        print(f"YOLO Model Test Successful! Detected objects: {len(results[0].boxes)}")
    except Exception as e:
        print(f"YOLO Model Test Failed: {str(e)}")
    
    # Kiểm tra mô hình OCR
    try:
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)  # Tạo ảnh giả để kiểm tra OCR
        result = ocr.ocr(dummy_image, rec=True)
        print(f"OCR Model Test Successful! OCR Result: {result}")
    except Exception as e:
        print(f"OCR Model Test Failed: {str(e)}")

@app.on_event("startup")
async def startup():
    # Khởi chạy mô hình OCR và YOLO trong background
    thread_ocr = Thread(target=load_ocr_model, daemon=True)
    thread_yolo = Thread(target=load_yolo_model, daemon=True)

    thread_ocr.start()
    thread_yolo.start()
    # Đảm bảo rằng mô hình đã sẵn sàng trước khi kiểm tra
    thread_ocr.join()  # Đợi mô hình OCR tải xong
    thread_yolo.join()  # Đợi mô hình YOLO tải xong
    
    # Đánh dấu mô hình đã sẵn sàng
    global is_model_ready
    is_model_ready = True
    # Chạy hàm ping_cloudinary trong một thread riêng để không làm block FastAPI
    thread = Thread(target=ping_cloudinary, daemon=True)
    thread.start()
    # Kiểm tra mô hình ngay khi khởi động
    test_models()

# Tạo thư mục "img" nếu chưa tồn tại
# img_folder = 'E:/datn-AI-module/be-ocr/img'
# os.makedirs(img_folder, exist_ok=True)


@app.post("/upload")
async def upload_image(background_tasks: BackgroundTasks,id: str = Form(...),parkingid: str = Form(...), file: UploadFile = File(...)):
    # Đọc file hình ảnh từ request
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise ValueError("Received empty image data")

    # Chuyển đổi hình ảnh từ bytes thành array
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Phát hiện biển số xe với YOLO
    results = model.predict(source=image, save=False, imgsz=640)

    # Lưu hình ảnh vào thư mục img với tên là ID
    # img_path = os.path.join(img_folder, f"{id}.png")
    # cv2.imwrite(img_path, image)
    # print(f"{datetime.now()} - Hình ảnh đã lưu tại: {img_path}")

    # Biến để lưu kết quả OCR nhận diện nếu có nhiều dòng
    license_plate_text = ""
    # Chọn bounding box có độ tin cậy cao nhất
    best_box = None
    best_confidence = 0

    for box in results[0].boxes:
        confidence = float(box.conf[0])  # Lấy độ tin cậy của YOLO
        if confidence > best_confidence:
            best_confidence = confidence
            best_box = box

    # Nếu tìm thấy một biển số xe có độ tin cậy cao
    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])  # Lấy tọa độ của box tốt nhất
        cropped_image = image[y1:y2, x1:x2]  # Cắt ảnh biển số có độ tin cậy cao nhất

        # Nhận diện biển số với PaddleOCR
        result = ocr.ocr(cropped_image, rec=True)

        # Duyệt qua các dòng văn bản đã nhận diện
        license_plate_text = " ".join([line[1][0] for line in result[0]])



    # Duyệt qua tất cả bounding box của YOLO
    # for i, box in enumerate(results[0].boxes):
    #     confidence = box.conf[0].item()  # Lấy độ tin cậy của bounding box
    #     if confidence > best_confidence:
    #         best_confidence = confidence
    #         best_box = box
    #     # Lấy tọa độ bounding box
    #     x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box [x_min, y_min, x_max, y_max]

    #     # Cắt vùng biển số
    #     cropped_image = image[y1:y2, x1:x2]

    #     # # Lưu ảnh biển số đã cắt vào thư mục "crop" với tên giữ nguyên và thêm chỉ số
    #     # crop_path = os.path.join(img_folder, f"{id}_crop_{i}.jpg")
    #     # cv2.imwrite(crop_path, cropped_image)
    #     # print(f"{datetime.now()} - Biển số đã cắt lưu tại: {crop_path}")

    #     # Sử dụng PaddleOCR để nhận diện văn bản trong ảnh cắt
    #     result = ocr.ocr(cropped_image, rec=True)

    #     # Duyệt qua các dòng văn bản đã nhận diện
    #     for line in result[0]:
    #         plate_text = line[1][0]  # Lấy văn bản nhận diện từ mỗi dòng
    #         license_plate_text += plate_text + " "  # Nối các phần lại với nhau

    # In kết quả OCR và thời gian hiện tại
    if license_plate_text.strip():  # Nếu có kết quả nhận diện
        print(f"{datetime.now()} - Biển số OCR nhận diện: {license_plate_text.strip()}")
        # Lưu biển số vào Redis trước khi tiếp tục
        save_license_plate_to_redis(id, license_plate_text.strip())
        print("pass redis")
        # Push MQTT message vào topic 'ocr'
        push_mqtt_message_ocr_done(id, parkingid)

        # Thực hiện upload ảnh lên Cloudinary trong nền (background task)
    background_tasks.add_task(upload_image_to_cloudinary, image_bytes, id)

    return {"message": "OCR processing complete", "license_plate": license_plate_text.strip()}

# Hàm upload ảnh lên Cloudinary trong nền
def upload_image_to_cloudinary(image_bytes, id):
    try:
        cloudinary_response = cloudinary_upload(image_bytes, folder="ocr_images", public_id=id)
        cloudinary_url = cloudinary_response.get("url")
        print(f"{datetime.now()} - Ảnh đã tải lên Cloudinary: {cloudinary_url}")
    except Exception as e:
        print(f"{datetime.now()} - Lỗi khi tải ảnh lên Cloudinary: {str(e)}")