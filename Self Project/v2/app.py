import configparser
from flask import Flask, request, abort
from linebot.v3 import (
    WebhookHandler
)
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    AudioMessageContent,
    ImageMessageContent,
    PostbackEvent,
    UserSource
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
)

from message_process import process_message

# Config Parser
config = configparser.ConfigParser()
config.read('config.ini')

# 初始化 Line Messaging API 客戶端 get
try:
    LINE_CHANNEL_ACCESS_TOKEN = config.get('Line', 'CHANNEL_ACCESS_TOKEN')
    LINE_CHANNEL_SECRET = config.get('Line', 'CHANNEL_SECRET')
    # 初始化 Messaging API 客戶端
    line_config = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
    # 初始化 Webhook Handler
    handler = WebhookHandler(LINE_CHANNEL_SECRET)
except configparser.NoSectionError:
    print("Please provide Line section in config.ini")
    exit()

# 初始化 Flask 應用程式
app = Flask(__name__)

@app.route("/callback", methods=['POST'])
def callback():
    # 取得 X-Line-Signature 標頭
    signature = request.headers.get('X-Line-Signature', '')

    # 取得請求內容
    body = request.get_data(as_text=True)
    app.logger.info(f"Request body: {body}")

    # 驗證並處理 Webhook
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

# 處理文字訊息
@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event: MessageEvent):
    # 取得使用者 ID
    user_id = event.source.user_id
    reply = process_message(event, user_id)
    with ApiClient(line_config) as api_client:
        messaging_api = MessagingApi(api_client)
    messaging_api.reply_message(reply)

# 處理影像訊息
@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event: MessageEvent):
    
    # 取得使用者 ID
    user_id = event.source.user_id
    reply = process_message(event, user_id)
    with ApiClient(line_config) as api_client:
        messaging_api = MessagingApi(api_client)
    messaging_api.reply_message(reply)

# 處理 Postback 事件
@handler.add(PostbackEvent)
def handle_postback(event):
    user_id = event.source.user_id
    reply = process_message(event, user_id)
    with ApiClient(line_config) as api_client:
        messaging_api = MessagingApi(api_client)
    messaging_api.reply_message(reply)
    

