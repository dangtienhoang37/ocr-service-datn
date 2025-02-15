# # Stage 1: Build stage
# FROM python:3.10-slim 

# # Set working directory
# WORKDIR /app

# # Copy requirements file
# COPY requirements.txt .

# # Install dependencies
# RUN apt-get update && apt-get install --no-install-recommends -y \
#     libglib2.0-0 libgl1-mesa-glx \
#     && pip install --no-cache-dir -r requirements.txt \
#     && rm -rf /var/lib/apt/lists/*



# # Expose the application port
# EXPOSE 8000

# # Run the FastAPI application with Uvicorn
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
# Sử dụng image Python nhỏ
FROM python:3.10-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Thiết lập biến môi trường
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Copy file requirements.txt
COPY requirements.txt .

# Cài đặt các thư viện cần thiết
RUN apt-get update && apt-get install --no-install-recommends -y \
    libglib2.0-0 libgl1-mesa-glx \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove \
    && rm -rf /var/lib/apt/lists/*

# Copy toàn bộ mã nguồn vào container
COPY . .

# Mở cổng ứng dụng
EXPOSE 8000

# Lệnh chạy ứng dụng
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
