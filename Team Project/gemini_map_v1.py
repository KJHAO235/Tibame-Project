import os
import json
import logging
import requests
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai

import matplotlib
matplotlib.use("Agg")

from io import BytesIO
from configparser import ConfigParser
from datetime import datetime, timedelta
from PIL import Image, ImageDraw
from google.cloud import storage
from google.oauth2 import service_account

try:
    config = ConfigParser()
    config.read("config.ini", encoding="utf-8")
    GEMINI_API_KEY = config["Google"]["GEMINI_API_KEY"]
    if not GEMINI_API_KEY:
        raise FileNotFoundError("Error reading config.ini")
except Exception as e:
    logging.error(f"Error reading config.ini: {e}")
    try:
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    except Exception as e:
        logging.error(f"Error reading environment variables: {e}")
        raise FileNotFoundError("Error reading environment variables")


#------------------------------------以下為初始化設定及基本功能------------------------------------
# 設定gemini模型
def set_gemini_model():

    logging.info("配置 Gemini API")
    genai.configure(api_key=GEMINI_API_KEY)

    # 設定gemini-1.5-flash模型
    logging.info("設定 Gemini 模型")
    model = genai.GenerativeModel('gemini-2.0-flash-exp',
                generation_config={
                "temperature": 0,
                "response_mime_type":"application/json"
                })
    return model

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
def upload_image_to_gcs(image, user_id, bucket_name="map-storage-20250106", folder_name="static", expiration_minutes=60):
    """
    將圖片上傳到 Google Cloud Storage，並使用 user_id 和時間戳生成檔案名稱。

    Args:
        image (numpy.ndarray): OpenCV 圖像數據。
        bucket_name (str): GCS 存儲桶名稱。
        user_id (str): 用戶 ID。
        folder_name (str): GCS 中的目標資料夾名稱（默認為 static）。
    Returns:
        str: 上傳到 GCS 的檔案完整路徑。
    """
    # 生成時間戳
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # 動態生成 blob_name
    file_name = f"{user_id}_{timestamp}.png"
    blob_name = f"{folder_name}/{file_name}"

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

#------------------------------------圖片前處理（繪製網格）------------------------------------
# 讀取標記圖片（從 URL）
def read_marker_images_from_url(start_size=(50, 50),end_size=(33, 48)):
    try:
        # 從 URL 讀取起點標記圖片
        start_marker_url = 'https://storage.googleapis.com/map_startend_storage-20250112/start.png'
        start_response = requests.get(start_marker_url)
        start_response.raise_for_status()  # 檢查請求是否成功
        start_marker = Image.open(BytesIO(start_response.content)).convert("RGBA")
        
        # 從 URL 讀取終點標記圖片
        end_marker_url = 'https://storage.googleapis.com/map_startend_storage-20250112/end.png'
        end_response = requests.get(end_marker_url)
        end_response.raise_for_status()  # 檢查請求是否成功
        end_marker = Image.open(BytesIO(end_response.content)).convert("RGBA")
        
        # 調整標記圖片大小
        start_marker = start_marker.resize(start_size)
        end_marker = end_marker.resize(end_size)
        
        return start_marker, end_marker
    except Exception as e:
        logging.error(f"無法讀取標記圖片: {e}")
        return None, None

# 從 URL 讀取圖片並轉換為 RGB 格式
def read_and_convert_image(image_url):
    try:
        # 發送 GET 請求下載圖片
        response = requests.get(image_url)
        response.raise_for_status()  # 確保請求成功，否則拋出 HTTPError
        
        # 使用 PIL 打開圖片並轉換為 RGB 格式
        image = Image.open(BytesIO(response.content))
        # image = Image.open(image_url)
        image = image.convert("RGB")  # 確保轉換為 RGB 格式
        logging.info(f"圖片已成功從 URL 加載並轉換為 RGB 格式")
        return image
    except Exception as e:
        logging.error(f"無法從 URL 讀取或轉換圖片: {e}")
        return None

# 繪製網格
def draw_grid(image, grid_size=50):
    """
    在圖片上繪製網格，回傳繪製後的圖片及圖片尺寸。

    Args:
        image (PIL.Image.Image): 原始圖片。
        grid_size (int): 網格的間隔大小，預設為 50。

    Returns:
        PIL.Image.Image: 繪製網格後的圖片。
        int: 圖片的寬度。
        int: 圖片的高度。
    """
    logging.info("開始繪製網格")
    # 獲取圖片 DPI，如果沒有，預設為 72
    dpi = image.info.get('dpi', (72, 72))[0]

    # 使用 Matplotlib 繪製網格
    fig, ax = plt.subplots(figsize=(image.width / dpi, image.height / dpi), dpi=dpi)
    ax.imshow(image)

    ax.set_xlim(0, image.width)
    ax.set_ylim(image.height, 0)  # 設置 Y 軸從下到上遞增
    ax.set_xticks(np.arange(0, image.width, grid_size))  # 設置 X 軸刻度
    ax.set_yticks(np.arange(0, image.height, grid_size))  # 設置 Y 軸刻度
    ax.grid(True, which="both", color="black", linestyle="--", linewidth=0.5)  # 顯示網格
    ax.axis('on')  # 顯示座標軸

    # 將 Matplotlib 圖片保存到內存
    buf = BytesIO()
    plt.savefig(buf, format='PNG', bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close(fig)
    buf.seek(0)

    # 讀取內存中的圖片，轉換為 PIL.Image 對象
    grid_image = Image.open(buf)

    logging.info("網格繪製完成")
    return grid_image, image.width, image.height


#------------------------------------調用Gemini及處理格式------------------------------------
# 調用 Gemini API 獲取建議
def call_gemini_for_suggestions(map_image, max_x, max_y, start, end):
    logging.info("開始發送 Gemini 請求")
    user_input = f"""
            你是一位專業的展覽導遊，正在幫助一位遊客找到他們想要參觀的店家。
            請參考圖片網格x軸y軸座標數據，辨識start所對應的座標點。
            
            圖中的網格被視為一個平面直角坐標系，參考座標位於x軸及y軸上，其中：

            規則：
            最左上角的點視為原點座標 (0, 0)。
            X 軸：從 (0, 0) 向右延伸為正，最大值為 {max_x}
            Y 軸：從 (0, 0) 向下延伸為正，最大值為 {max_y}
            
            每個攤位的編號與其對應的座標需要通過圖中的邊界區塊位置來判斷，而不是直接從編號推導出座標。

            請提供:
            {start}精確的區塊中心座標start_coordinate，不要給我{start}。
            {end}精確的區塊中心座標end_coordinate，不要給我{end}。
            格式如下:
            "start_coordinate": {{"x": "num","y": "num"}},
            "end_coordinate": {{"x": "num","y": "num"}}

            依據你所提供的座標，遵循以下規則:
            1.避開除了start與goal的區塊邊界，途中個別有編號區塊絕對不可穿越，要繞過，不要讓路徑穿過其他區塊。
            2.依據格式提供route_coordinate，不要提供其他格式。
            3.轉彎時應該會多一個座標點，這是正常的。
            4.格式: "route_coordinate": [{{"x": "num","y": "num"}},{{"x": "num","y": "num"}},...]
            """
    try:
        # map_image = Image.open(map_image)
        # logging.info("圖片已加載")

        # # 將圖片轉換為二進制數據
        # buffer = BytesIO()
        # map_image.save(buffer, format="PNG")
        # buffer.seek(0)
        # binary_image = buffer.read()

        model = set_gemini_model()
        response = model.generate_content([map_image, user_input], generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0  # 設置溫度參數為 0
        ))
        suggestions = response._result.candidates[0].content.parts[0].text

        if hasattr(response, 'text'):
            suggestions = response.text.replace("*", "")
            logging.info(f"獲取到的建議: {suggestions}")
        else:
            suggestions = "無法獲取有效的Gemini建議。"
            logging.warning(suggestions)
        return suggestions
    except Exception as e:
        logging.error(f"無法獲取Gemini建議: {e}")
        return "無法獲取Gemini建議。" + str(e)

# 轉為字典格式
def gen_dict(suggestions):
    logging.info("開始解析建議為字典格式")
    if isinstance(suggestions, str):
        try:
            suggestions = json.loads(suggestions)
            logging.info("建議已解析為字典格式")
        except json.JSONDecodeError as e:
            logging.error(f"無法解析建議為 JSON: {e}")
            return {}
    return suggestions


#------------------------------------繪製路徑------------------------------------
# 繪製路徑
def draw_path_on_image(user_id, image, START_Name, END_Name):
    # 設置日誌配置
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    image = read_and_convert_image(image)
    if image is None:
        raise FileNotFoundError("無法讀取或轉換圖片，程式終止。")
    
    draw = ImageDraw.Draw(image)
    
    logging.info("開始繪製網格")
    preprocessed_image, max_x, max_y = draw_grid(image)
    
    # 讀取標記圖片並調整大小
    start_marker, end_marker = read_marker_images_from_url()
    if start_marker is None or end_marker is None:
        raise FileNotFoundError("無法讀取標記圖片，程式終止。")

    suggestion = call_gemini_for_suggestions(preprocessed_image, max_x, max_y, START_Name, END_Name)
    suggestions = gen_dict(suggestion)
    
    logging.info("開始在圖片上繪製標記和路徑")
    # 給定文字中心座標和文字信息
    start_coordinate = suggestions.get('start_coordinate', {})
    end_coordinate = suggestions.get('end_coordinate', {})
    route_coordinates = suggestions.get('route_coordinate', [])

    # 繪製路徑
    if route_coordinates:
        for i in range(len(route_coordinates) - 1):
            start = (int(route_coordinates[i]['x']), int(route_coordinates[i]['y']))
            end = (int(route_coordinates[i + 1]['x']), int(route_coordinates[i + 1]['y']))
            draw.line([start, end], fill="green", width=5)
        logging.info("路徑已繪製")

    # 繪製起點標記
    if 'x' in start_coordinate and 'y' in start_coordinate:
        start_x = int(start_coordinate['x'])
        start_y = int(start_coordinate['y'])
        logging.info(f"起點座標: ({start_x}, {start_y})")
        image.paste(start_marker, (start_x - start_marker.width // 2, start_y - start_marker.height // 2), start_marker)
    else:
        logging.error("無法獲取起點座標")

    # 繪製終點標記
    if 'x' in end_coordinate and 'y' in end_coordinate:
        end_x = int(end_coordinate['x'])
        end_y = int(end_coordinate['y'])
        logging.info(f"終點座標: ({end_x}, {end_y})")
        image.paste(end_marker, (end_x - end_marker.width // 2, end_y - end_marker.height // 2), end_marker)
    else:
        logging.error("無法獲取終點座標")

    # 將處理後的圖片上傳至 GCS
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    public_url = upload_image_to_gcs(image, user_id)
    print("公開訪問的圖片網址:", public_url)
    return public_url# 保存結果圖片


# logging.info("開始繪製圖片")
if __name__ == "__main__":
    image = "test4.jpg"
    user_id = "test123"
    map = draw_path_on_image(user_id, image, "馬祖酒廠", "陽明")