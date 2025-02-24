from io import BytesIO
from datetime import datetime, timedelta
from PIL import Image, ImageDraw
from google.cloud import storage
from google.oauth2 import service_account

# 初始化 GCS 客戶端
def get_gcs_client_from_key(key_path):
    """
    使用服務帳戶金鑰文件初始化 GCS 客戶端。

    Args:
        key_path (str): 服務帳戶金鑰的 JSON 文件路徑。

    Returns:
        google.cloud.storage.Client: GCS 客戶端實例。
    """
    credentials = service_account.Credentials.from_service_account_file(key_path)
    client = storage.Client(credentials=credentials, project=credentials.project_id)
    return client

#上傳圖片到GCS
def upload_image_to_gcs(image, user_id, timestamp: float = None,
                        bucket_name="map-storage-20250106", folder_name="static", expiration_minutes=60):
    """
    將圖片上傳到 Google Cloud Storage，並使用 user_id 和時間戳生成檔案名稱。

    Args:
        image (numpy.ndarray): OpenCV 圖像數據。
        bucket_name (str): GCS 存儲桶名稱。
        timestamp (int): 時間戳（默認為當前時間）。
        user_id (str): 用戶 ID。
        folder_name (str): GCS 中的目標資料夾名稱（默認為 static）。
    Returns:
        str: 上傳到 GCS 的檔案完整路徑。
    """
    # 生成時間戳
    if not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    else:
        timestamp = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S") # 將時間戳轉換為格式化時間字符串


    # 動態生成 blob_name
    file_name = f"{user_id}_{timestamp}.png"
    print(f"上傳圖片至 GCS，檔案名稱: {file_name}")
    blob_name = f"{folder_name}/{file_name}"
    print(f"上傳圖片至 GCS，檔案名稱: {blob_name}")

    # 初始化 GCS 客戶端
    client = get_gcs_client_from_key("gcskey.json")
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # 將 Pillow 圖片保存到內存中的二進制數據
    buffer = BytesIO()
    image.save(buffer, format="PNG")  # 保存為 PNG 格式
    buffer.seek(0)  # 將指針重置到內存的起始位置

    # 上傳圖片到 GCS
    blob.upload_from_file(buffer, content_type="image/png")

    # 生成簽名 URL
    expiration_time = timedelta(minutes=expiration_minutes)
    signed_url = blob.generate_signed_url(expiration=expiration_time, method="GET")
    # print(f"圖片已上傳到 GCS，簽名 URL: {signed_url}")
    return signed_url