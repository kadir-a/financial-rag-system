# 1. Hafif ve güvenli bir Python tabanı seçiyoruz
FROM python:3.10-slim

# 2. Çevresel değişkenler (Logların anında düşmesi için)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Konteyner içindeki çalışma masamız
WORKDIR /app

# 4. İşletim sistemi seviyesinde gerekli C derleyicileri (ChromaDB sorun çıkarmaz)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Sadece kütüphane listesini kopyala ve kur (Önbellek optimizasyonu)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Tüm proje dosyalarını kopyala
COPY . .

# 7. Streamlit portunu dışarı aç
EXPOSE 8501

# 8. Konteyner ayağa kalktığında çalıştırılacak komut
CMD ["python", "-m", "streamlit", "run", "ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]