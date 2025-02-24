# 使用基礎映像檔
FROM python:3.10-slim

# 設定環境變數
ENV TZ=Asia/Taipei \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1

# 設置工作目錄
WORKDIR /app

# 複製專案檔案到容器
COPY . /app

# 安裝必要工具和套件
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libssl-dev \
    libffi-dev \
    libopencv-dev \
    tzdata \
    ffmpeg \
    curl && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 安裝 Python 依賴
RUN pip3 install --no-cache-dir -r requirements.txt

# 暴露服務端口
EXPOSE 8080

# 設定容器啟動時執行的命令
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8080", "app:app"]




