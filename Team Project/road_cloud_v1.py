# Description: 這是一個展覽地圖路徑規劃的程式，可以根據展覽地圖和攤位框生成的遮罩圖像，計算兩個攤位之間的最短路徑。
# pip install opencv-python
import cv2
import requests
import numpy as np
import networkx as nx

import os
from configparser import ConfigParser
from datetime import datetime, timedelta

# pip install azure-core
from azure.core.credentials import AzureKeyCredential
# pip install azure-ai-vision-imageanalysis
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures

from google.cloud import storage
from google.oauth2 import service_account

#------------------------------------以下為初始化設定及基本功能------------------------------------

# 初始化 Azure Computer Vision 客戶端
def az_cv_set():
    config = ConfigParser()
    config.read("config.ini", encoding="utf-8")

    # Azure Computer Vision 設定
    CV_KEY = config.get('Azure', 'CV_KEY')
    CV_ENDPOINT = config.get('Azure', 'CV_ENDPOINT')

    # CV_KEY = os.envget("CV_KEY")
    # CV_ENDPOINT = os.envget('CV_ENDPOINT')

    # Create an Image Analysis client
    client = ImageAnalysisClient(
        endpoint=CV_ENDPOINT,
        credential=AzureKeyCredential(CV_KEY)
    )
    return client

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

    # 將 OpenCV 圖像保存為內存中的二進制數據
    _, buffer = cv2.imencode('.png', image)
    blob.upload_from_string(buffer.tobytes(), content_type='image/png')

    # 生成簽名 URL
    expiration_time = timedelta(minutes=expiration_minutes)
    signed_url = blob.generate_signed_url(expiration=expiration_time, method="GET")
    print(f"圖片已上傳到 GCS，簽名 URL: {signed_url}")
    return signed_url

#從 URL 讀取圖片並轉換為 OpenCV 格式
def read_image_from_url(url):
    """
    從 URL 讀取圖片並轉換為 OpenCV 格式。

    Args:
        url (str): 圖片的 URL。

    Returns:
        numpy.ndarray: OpenCV 格式的圖像數據。
    """
    response = requests.get(url, verify=False)
    response.raise_for_status()  # 確保請求成功

    # 將下載的內容轉換為 NumPy 數組
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)

    # 解碼為 OpenCV 格式的圖片
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("無法從 URL 解碼圖片。")
    
    return image

#根據使用者輸入的 MongoDB Atlas 格式 _id 查詢展覽標題  
def find_exhibition_title_by_id(documents, input_id):
    """
    根據使用者輸入的 MongoDB Atlas 格式 _id 查詢展覽標題。

    Args:
        documents (list): 包含展覽數據的 Document 列表。
        input_id (str): 使用者輸入的 _id 字串。

    Returns:
        str: 對應的展覽標題，若未找到則返回提示訊息。
    """
    for document in documents:
        # 提取 _id 的 $oid 值
        # document_id = document.metadata["_id"].get("$oid") if isinstance(document.metadata["_id"], dict) else None
        # if document_id == input_id:
        #     return document.metadata.get("title", "未找到標題")
        if str(document.metadata["_id"]) == input_id:
            return document.metadata.get("title", "未找到標題")
        print(f"未找到對應的展覽，請檢查 _id 是否正確。")
    return "未找到對應的展覽"

#------------------------------------以下為展覽地圖路徑規劃的程式------------------------------------

# 提取文字中心點
def calculate_center(boundingPolygon):
    """
    計算給定四邊形的中心點。

    Args:
        boundingPolygon (list): 四個頂點的座標 [{'x': x1, 'y': y1}, {'x': x2, 'y': y2}, ...]

    Returns:
        dict: 中心點的座標 {'x': cx, 'y': cy'}
    """
    # 計算中心點
    x_coords = [point['x'] for point in boundingPolygon]
    y_coords = [point['y'] for point in boundingPolygon]
    center_x = round(sum(x_coords) / len(x_coords))
    center_y = round(sum(y_coords) / len(y_coords))
    return {'x': center_x, 'y': center_y}

def extract_centers(data):
    """
    從給定資料中提取每個詞語的中心點並整理為 JSON 格式。

    Args:
        data (dict): JSON 格式的資料，包含 'blocks' -> 'lines' -> 'boundingPolygon'.

    Returns:
        list: 包含詞語和中心點的 JSON 資料 [{'text': '詞語', 'center': {'x': cx, 'y': cy}}]
    """
    result = []
    for block in data.get('blocks', []):
        for line in block.get('lines', []):
            text = line['text']
            boundingPolygon = line['boundingPolygon']
            center = calculate_center(boundingPolygon)
            result.append({'text': text, 'center': center})
    return result

def extract_text_from_image(image_path):
    """
    從圖像中提取文字。

    Args:
        image_path (str): 圖像路徑。

    Returns:
        dict: 包含文字和中心點的 JSON 資料 [{'text': '詞語', 'center': {'x': cx, 'y': cy}}]
    """
    if image_path.startswith("http://") or image_path.startswith("https://"):
        # 如果是 URL，使用 requests 下載圖片
        response = requests.get(image_path, verify=False)
        if response.status_code != 200:
            raise FileNotFoundError(f"無法從 URL 讀取圖像: {image_path}")
        image_data = response.content

    # client.analyze_from_url
    result = az_cv_set().analyze(
        image_data=image_data,
        visual_features=[VisualFeatures.READ]
    )

    # 提取中心點
    words = extract_centers(result.read)
    return words

#處理展覽地圖，生成攤位框和道路遮罩
def process_map_with_mask(image_path):
    """
    處理展覽地圖，生成攤位框和道路遮罩。

    Args:
        image_path (str): 輸入圖片路徑。
        output_path_mask (str): 保存道路遮罩的路徑。
        output_path_boxes (str): 保存攤位框的路徑。
    """
    # 讀取圖像
    # image = cv2.imread(image_path)
    image = read_image_from_url(image_path)
    if image is None:
        raise FileNotFoundError(f"無法讀取圖像: {image_path}")

    # 轉換為灰度圖像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 邊緣檢測
    edges = cv2.Canny(gray_image, 30, 100, apertureSize=3)

    # 霍夫直線檢測（調整效率！）
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=30, maxLineGap=20)

    # 創建一個空白遮罩圖像
    mask = np.ones_like(gray_image) * 255  # 初始為全白（道路）
    boxes_image = image.copy()  # 用於顯示攤位框的圖像

    vertical_lines = []
    horizontal_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 判斷是垂直線還是水平線
            if abs(x1 - x2) < 10:  # 垂直線
                vertical_lines.append((x1, y1, x2, y2))
            elif abs(y1 - y2) < 10:  # 水平線
                horizontal_lines.append((x1, y1, x2, y2))

    # 確定網格的交點
    vertical_lines = sorted(vertical_lines, key=lambda x: x[0])  # 按 x 坐標排序
    horizontal_lines = sorted(horizontal_lines, key=lambda x: x[1])  # 按 y 坐標排序

    # 遍歷網格框
    for i in range(len(vertical_lines) - 1):
        for j in range(len(horizontal_lines) - 1):
            # 確定網格邊界
            x1 = vertical_lines[i][0]
            x2 = vertical_lines[i + 1][0]
            y1 = horizontal_lines[j][1]
            y2 = horizontal_lines[j + 1][1]

            # 提取網格內部的區域
            grid_region = gray_image[y1:y2, x1:x2]
            avg_intensity = np.mean(grid_region)  # 計算網格內像素的平均亮度

            # 判斷是否為攤位（根據亮度）
            if avg_intensity < 200:  # 假設暗區域是攤位
                # 填充攤位區域為黑色（遮罩）
                mask[y1:y2, x1:x2] = 0
                # 在框圖上標記攤位框
                cv2.rectangle(boxes_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 綠色框標記攤位

    # 再次檢測封閉區域，處理非矩形區域（如梯形）
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # 計算輪廓的面積，過濾過小的區域
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        # 檢查是否為接近整張圖的外框
        if (x <= 5 or y <= 5 or x + w >= gray_image.shape[1] - 5 or y + h >= gray_image.shape[0] - 5) and area > 0.9 * gray_image.shape[0] * gray_image.shape[1]:
            continue  # 忽略接近整張圖大小的外框

        if area > 500:  # 可根據需求調整面積閾值
            # 判斷輪廓是否為封閉區域
            perimeter = cv2.arcLength(contour, True)
            if cv2.isContourConvex(contour) or (perimeter > 0 and area / perimeter > 5):
                # 填充封閉區域為黑色
                cv2.fillPoly(mask, [contour], 0)

    return mask

#繪製路線圖
def find_connected_region(mask, start_point):
    """
    找到遮罩圖中指定起始點所屬的黑色區域。

    Args:
        mask (numpy.ndarray): 遮罩圖像。
        start_point (tuple): 起始點座標 (x, y)。

    Returns:
        list: 黑色區域中的所有像素座標列表。
    """
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    region = []
    stack = [start_point]

    while stack:
        x, y = stack.pop()
        if visited[y, x]:
            continue
        visited[y, x] = True
        region.append((x, y))
        # 檢查四周像素
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx_, ny_ = x + dx, y + dy
            if 0 <= nx_ < w and 0 <= ny_ < h and mask[ny_, nx_] == 0 and not visited[ny_, nx_]:
                stack.append((nx_, ny_))
    return region

def find_boundary_point(mask, region):
    """
    找到黑色區域中與白色道路鄰接的第一個像素。

    Args:
        mask (numpy.ndarray): 遮罩圖像。
        region (list): 黑色區域中的所有像素座標。

    Returns:
        tuple: 與道路鄰接的第一個白色像素座標。
    """
    h, w = mask.shape
    for x, y in region:
        # 檢查四周像素
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx_, ny_ = x + dx, y + dy
            if 0 <= nx_ < w and 0 <= ny_ < h and mask[ny_, nx_] == 255:
                return (nx_, ny_)
    return None

def update_point_to_boundary(mask, point):
    """
    將給定點更新到攤位區域與道路的鄰接點。

    Args:
        mask (numpy.ndarray): 遮罩圖像。
        point (tuple): 原始點座標 (x, y)。

    Returns:
        tuple: 更新後的點座標（鄰接白色道路的點）。
    """
    # 檢查點是否在黑色區域
    if mask[point[1], point[0]] != 0:
        raise ValueError(f"點 {point} 不在黑色區域內。")

    # 找到該點所屬的黑色區域
    region = find_connected_region(mask, point)

    # 找到與道路鄰接的第一個點
    boundary_point = find_boundary_point(mask, region)
    if boundary_point is None:
        raise ValueError(f"無法找到攤位區域與道路鄰接的點。")
    
    return boundary_point

def smooth_path(path):
    """
    對路徑進行平滑處理。

    Args:
        path (list): 路徑點列表。

    Returns:
        list: 平滑後的路徑。
    """
    smoothed_path = []
    for i in range(1, len(path) - 1):
        x_prev, y_prev = path[i - 1]
        x_curr, y_curr = path[i]
        x_next, y_next = path[i + 1]
        smoothed_x = (x_prev + x_curr + x_next) // 3
        smoothed_y = (y_prev + y_curr + y_next) // 3
        smoothed_path.append((smoothed_x, smoothed_y))
    return [path[0]] + smoothed_path + [path[-1]]

def calculate_shortest_path(img_path, start_booth, end_booth, user_id):
    """
    根據展覽地圖遮罩圖計算兩點之間的最短路徑。

    Args:
        mask_path (str): 遮罩圖的路徑。
        start_point (tuple): 起點座標 (x, y)。
        end_point (tuple): 終點座標 (x, y)。
        output_path (str): 保存結果圖像的路徑。
    """
    # 提取文字與攤位位置
    word_location = extract_text_from_image(img_path)

    # 將攤位列表轉換為字典格式
    booth_dict = {item['text']: (item['center']['x'], item['center']['y']) for item in word_location}

    if start_booth in booth_dict and end_booth in booth_dict:
        start_point = booth_dict[start_booth]
        end_point = booth_dict[end_booth]

        # 讀取遮罩圖
        mask = process_map_with_mask(img_path)
        # 讀取原圖
        original_img = read_image_from_url(img_path)
        if original_img is None:
            raise FileNotFoundError(f"無法讀取原始圖像: {img_path}")

        # 計算距離轉換
        dist_transform = cv2.distanceTransform((mask == 255).astype(np.uint8), cv2.DIST_L2, 5)
        max_dist = np.max(dist_transform)  # 獲取距離轉換的最大值

        # 更新起點和終點到鄰接道路的白色像素
        start_point = update_point_to_boundary(mask, start_point)
        end_point = update_point_to_boundary(mask, end_point)

        # 構建圖
        graph = nx.Graph()
        rows, cols = mask.shape
        for y in range(rows):
            for x in range(cols):
                if mask[y, x] == 255:  # 僅將白色像素視為道路
                    neighbors = [(y + dy, x + dx) for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
                    for ny, nx_ in neighbors:
                        if 0 <= ny < rows and 0 <= nx_ < cols and mask[ny, nx_] == 255:
                            weight = max_dist - dist_transform[ny, nx_]  # 轉為正值權重
                            graph.add_edge((x, y), (nx_, ny), weight=weight)

        # 計算最短路徑
        try:
            path = nx.shortest_path(graph, source=start_point, target=end_point, weight='weight')
        except nx.NetworkXNoPath:
            raise ValueError("無法找到兩點之間的路徑。")

        # 平滑路徑
        path = smooth_path(path)

        # 在遮罩圖上繪製路徑
        result_image = original_img.copy()
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            cv2.line(result_image, start, end, (0, 0, 255), thickness=5)  # 繪製紅線，線條加粗

        # 將結果上傳到 GCS
        public_url = upload_image_to_gcs(result_image, user_id)
        print("公開訪問的圖片網址:", public_url)
        return public_url
    else:
        print("輸入的攤位名稱無效，請檢查後重新輸入喔。")
        return None
    
if __name__ == "__main__":
    user_id = "user123"
    start_point = "西4"
    end_point = "中4C"
    url = 'https://media.huashan1914.com/WebUPD/huashan1914/map/0106-0119-05_25010617504300886.jpg'
    calculate_shortest_path(url, start_point, end_point, user_id)