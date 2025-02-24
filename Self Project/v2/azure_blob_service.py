# services/azure_blob_service.py

from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
import time
from configparser import ConfigParser
from datetime import datetime, timedelta, timezone
config = ConfigParser()
config.read('config.ini')

connection_string = config.get("Azure", "AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER_NAME = "static"

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
blob_container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)

def upload_file_to_blob(file_content: bytes, user_id: str, file_ext: str) -> str:
    """
    將指定的 local_file_path 上傳到 Azure Blob，
    檔名：{user_id}_{timestamp}.{副檔名}。
    回傳 blob 的公開 URL。
    """
    try:
        timestamp = int(time.time())
        blob_filename = f"{user_id}_{timestamp}.{file_ext}"

        # 上傳檔案到 Azure Blob
        blob_client = blob_container_client.get_blob_client(blob_filename)
        blob_client.upload_blob(file_content, overwrite=True)

        # 生成 SAS Token，有效時間 1 小時
        sas_token = generate_blob_sas(
            account_name=blob_service_client.account_name,
            container_name=BLOB_CONTAINER_NAME,
            blob_name=blob_filename,
            account_key=blob_service_client.credential.account_key,
            permission=BlobSasPermissions(read=True),  # 允許讀取
            expiry = datetime.now(timezone.utc) + timedelta(hours=1)  # 設定 1 小時有效期
        )

        # 生成 Blob 的公開 URL
        blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{BLOB_CONTAINER_NAME}/{blob_filename}?{sas_token}"
        print('檔案上傳成功')
        return blob_url
    except Exception as e:
        print(f"Failed to upload to Azure Blob Storage: {e}")
        raise

if __name__ == "__main__":
    # Test upload_file_to_blob
    with open("test.png", "rb") as f:
        image_bytes = f.read()
        upload_file_to_blob(image_bytes, "test", "jpg")
