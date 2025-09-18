# Gunakan base image Python yang stabil dan ringan.
FROM python:3.13-slim

# Tetapkan direktori kerja di dalam container.
WORKDIR /app

# Salin file requirements.txt ke dalam container.
# Ini memungkinkan caching Docker untuk mempercepat build berikutnya.
COPY requirements.txt .

# Instal semua dependensi dari requirements.txt.
# Build akan memakan waktu di sini, tapi akan di-cache untuk deployment berikutnya.
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file dari proyek lokal ke dalam container.
COPY . .

# Beri tahu Docker bahwa container akan mendengarkan port 8000.
EXPOSE 8000

# Perintah untuk menjalankan aplikasi saat container diluncurkan.
# Pastikan main:app sesuai dengan nama file dan objek FastAPI Anda.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]