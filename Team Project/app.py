import os
import io
from configparser import ConfigParser
from flask import Flask, request, abort

from linebot.v3 import (
    WebhookHandler
)
from linebot.v3.exceptions import (
    InvalidSignatureError
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    MessagingApiBlob,
    TextMessage,
)
from linebot.v3.webhooks import (
    MessageEvent,
    PostbackEvent,
    TextMessageContent,
    ImageMessageContent,
)
from loading_animation import send_loading_animation


# # JSON
# import json
# from start_estab import read_user_history, start_user_history, write_user_history
# from user_reply import user_reply
# from postback_reply import postback_reply

# MongoDB
from start_estab_mgdb import read_user_info_mgdb, write_user_event_mgdb, write_user_info_mgdb
from user_reply_mgdb import user_reply_mgdb, user_image_reply_mgdb
from postback_reply_mgdb import postback_reply_mgdb

print("=====================================")
print("這裡是 app.py")
print("=====================================")

# 載入環境變數
try:
    config = ConfigParser()
    config.read('config.ini', encoding='utf-8')
    CHANNEL_ACCESS_TOKEN = config.get('LINE', 'CHANNEL_ACCESS_TOKEN')
    CHANNEL_SECRET = config.get('LINE', 'CHANNEL_SECRET')
    if not CHANNEL_ACCESS_TOKEN or not CHANNEL_SECRET:
        raise ValueError("Config file not set properly")
except Exception as e:
    print(f'Failed to load config file: {e}')
    # Use environment variables
    try:
        CHANNEL_ACCESS_TOKEN = os.getenv('CHANNEL_ACCESS_TOKEN')
        CHANNEL_SECRET = os.getenv('CHANNEL_SECRET')
        if not CHANNEL_ACCESS_TOKEN or not CHANNEL_SECRET:
            raise ValueError("Environment variables not set properly")
    except Exception as env_e:
        print(f'Failed to load environment variables: {env_e}')
        abort(500)


app = Flask(__name__)

try:
    configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
    handler = WebhookHandler(CHANNEL_SECRET)
except Exception as e:
    print(f"Error initializing configuration or handler: {e}")
    abort(500)


@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    print("這裡是 body：", body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.info("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_id = event.source.user_id
    send_loading_animation(user_id, CHANNEL_ACCESS_TOKEN, 60)

    # # JSON
    # # 讀取 user_history.json
    # user_history = read_user_history()
    # # 開始記錄使用者歷史資料，並建立相關資料欄位
    # start_user_history(user_history, event)
    # # 處理使用者輸入得到回覆訊息
    # reply_message, user_history = user_reply(user_history, event, configuration)
    # # 更新 user_history.json
    # write_user_history(user_history)


    # MongoDB
    # 讀取 user_info
    user_info = read_user_info_mgdb(user_id)
    # 寫入使用者事件至 collection
    write_user_event_mgdb(event)
    # 處理使用者輸入得到回覆訊息
    reply_message, user_info = user_reply_mgdb(user_info, event, configuration)
    # 更新 user_info
    write_user_info_mgdb(user_info)



    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=reply_message
            )
        )


@handler.add(PostbackEvent)
def handle_postback(event: PostbackEvent):
    print("這裡是 event.postback：", event.postback)
    user_id = event.source.user_id
    send_loading_animation(user_id, CHANNEL_ACCESS_TOKEN, 60)

    # # JSON
    # # 讀取 user_history.json
    # user_history = read_user_history()
    # # 開始記錄使用者歷史資料，並建立相關資料欄位
    # start_user_history(user_history, event)
    # # 處理 postback 得到回覆訊息
    # reply_message, user_history = postback_reply(event, user_history)
    # # 更新 user_history.json
    # write_user_history(user_history)


    # MongoDB
    # 讀取 user_info
    user_info = read_user_info_mgdb(user_id)
    # 寫入使用者事件至 collection
    write_user_event_mgdb(event)
    # 處理 postback 得到回覆訊息
    reply_message, user_info = postback_reply_mgdb(event, user_info)
    # 更新 user_info
    write_user_info_mgdb(user_info)


    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=reply_message
            )
        )


# 豪 + 軍
@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event):
    user_id = event.source.user_id
    # message: ImageMessageContent
    # type='image' id='544043007037407903' 
    # content_provider=ContentProvider(type='line', original_content_url=None, preview_image_url=None) 
    # image_set=None ："Image set ID. Only included when multiple images are sent simultaneously."

    # 取得圖片
    with ApiClient(configuration) as api_client:
        line_bot_blob_api = MessagingApiBlob(api_client)
        message_content = line_bot_blob_api.get_message_content(message_id=event.message.id)

    # MongoDB
    # 讀取 user_info
    user_info = read_user_info_mgdb(user_id)
    # 處理使用者輸入得到回覆訊息
    reply_message, user_info, img_url = user_image_reply_mgdb(user_info, event, message_content)
    # 寫入使用者事件至 collection
    write_user_event_mgdb(event, img_url)
    # 更新 user_info
    write_user_info_mgdb(user_info)


    # # 設定用戶狀態並保存到 Firestore
    # user_state = {
    #     "state": "waiting_for_start_point",
    #     "img_url": img_url
    # }
    # save_user_state_to_firestore(user_id, user_state)


    with ApiClient(configuration) as api_client:
        messaging_api = MessagingApi(api_client)
        messaging_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=reply_message
            )
        )



if __name__ == "__main__":
    app.run(port=5001)