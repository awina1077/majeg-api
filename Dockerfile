# Pakai Python versi 3.10
FROM python:3.10

# Set folder kerja
WORKDIR /code

# Install library sistem yang dibutuhkan OpenCV/YOLO
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Copy file requirements dulu (biar cache jalan)
COPY ./requirements.txt /code/requirements.txt

# Install library Python
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy semua file project kamu ke dalam server
COPY . /code

# Beri izin akses file (Penting buat Hugging Face)
RUN chmod -R 777 /code

# Jalankan aplikasi di port 7860 (Port wajib Hugging Face)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]