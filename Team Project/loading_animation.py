'''
這個檔案是用來發送 LINE 載入動畫的範例程式碼
https://developers.line.biz/en/docs/messaging-api/use-loading-indicator/
'''
import requests

def send_loading_animation(user_id, CHANNEL_ACCESS_TOKEN, loadingSeconds=5):
    url = "https://api.line.me/v2/bot/chat/loading/start"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {CHANNEL_ACCESS_TOKEN}",
    }
    
    # 呼叫所代的參數
    data = {
        "chatId": user_id, # 使用者 ID，不可為群組或聊天室 ID
        "loadingSeconds": loadingSeconds  # 可不加這行，將預設為 20 秒
        # 可設定 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 或 60 秒
    }

    # 發送 POST 請求到 LINE API
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code == 202:
        print("Loading animation sent successfully")
    else:
        print(f"Error: {response.status_code}, {response.text}")