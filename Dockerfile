# 1. Base image: lightweight and secure Python environment
FROM python:3.10-slim

# 2. Environment variables for optimal logging and performance
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Install essential OS-level dependencies (required for ChromaDB compilation)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy requirements and install dependencies (optimized for Docker cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the entire project payload
COPY . .

# 7. Expose the default Streamlit port
EXPOSE 8501

# 8. Define the default execution command for the container
CMD ["python", "-m", "streamlit", "run", "ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]