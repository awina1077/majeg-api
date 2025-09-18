# --- Tahap 1: Tahap Build ---
# Menggunakan image Python lengkap untuk instalasi paket.
FROM python:3.13 AS build

WORKDIR /app

# Salin file requirements.txt
COPY requirements.txt .

# Instal semua dependensi
RUN pip install --no-cache-dir -r requirements.txt

# --- Tahap 2: Tahap Final ---
# Menggunakan image Python yang lebih kecil (slim) sebagai image akhir.
FROM python:3.13-slim

WORKDIR /app

# Salin hanya paket yang sudah terinstal dari tahap build.
COPY --from=build /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages

# Salin file proyek Anda ke dalam image.
COPY . .

# Konfigurasi untuk Uvicorn
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]