import os
import io
import sys
import json
import wave
import tempfile
import configparser
import typing_extensions as typing
import google.generativeai as genai
import azure.cognitiveservices.speech as speechsdk
from PIL import Image
from typing import List
from flask import Flask, request, abort
from azure.storage.blob import BlobServiceClient
from linebot.v3 import (
    WebhookHandler
)
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    AudioMessageContent,
    ImageMessageContent,
    PostbackEvent
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    MessagingApiBlob,
    ReplyMessageRequest,
    TextMessage,
    StickerMessage,
    AudioMessage,
    QuickReply,
    QuickReplyItem,
    MessageAction,
    FlexMessage,
    FlexContainer,
    CameraAction,
    CameraRollAction,
)

# Config Parser
config = configparser.ConfigParser()
config.read('config.ini')

# 初始化 Line Messaging API 客戶端 get
LINE_CHANNEL_ACCESS_TOKEN = config['Line']['CHANNEL_ACCESS_TOKEN']
LINE_CHANNEL_SECRET = config['Line']['CHANNEL_SECRET']
if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    print("Channel Secret 或 Access Token 未設置")
    sys.exit(1)

# 初始化 Messaging API 客戶端
api_config = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)

# 初始化 Webhook Handler
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# 設定 Google Generative AI
genai.configure(api_key=config['Google']['GEMINI_API_KEY'])
model = genai.GenerativeModel('gemini-1.5-flash')

# 設定 Azure Speech
speech_key = config['Azure']['AZURE_SPEECH_KEY']
speech_region = config['Azure']['AZURE_REGION']
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
speech_config.speech_synthesis_voice_name = "zh-TW-HsiaoChenNeural" 

# 設定 Azure Blob Service
# connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
connection_string = config['Azure']['AZURE_STORAGE_CONNECTION_STRING']
BLOB_CONTAINER_NAME = "static-tmp"
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
blob_container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)

# 初始化 Flask 應用程式
app = Flask(__name__)

# 初始化全域變數來保存（針對不同使用者在後端資料庫做不同設定！）
interface_language = ""
detected_language = ""
language_isornot = None
function_type = ""
function_isornot = None

# 建立class 規格
class text_translation(typing.TypedDict):
        original_text: str
        translation_text: str

class food_content(typing.TypedDict):
        food_original_language_name: str
        food_translation_name: str
        ingredients: list[str]

class language_is(typing.TypedDict):
        isornot: bool

# 切換語言quick reply
language_quick_reply = QuickReply(
                            items=[
                                QuickReplyItem(
                                    action=MessageAction(
                                        label="繁體中文",
                                        text="interface_language:繁體中文"
                                        )
                                ),
                                QuickReplyItem(
                                    action=MessageAction(
                                        label="English",
                                        text="interface_language:English"
                                        )
                                ),
                            ]
                        )

# 設定檔案暫存資料夾
# static_tmp_path = os.path.join(os.path.dirname(__file__), 'static', 'tmp')
# 創建暫存檔資料夾
# def make_static_tmp_dir():
#     try:
#         os.makedirs(static_tmp_path)
#     except OSError as exc:
#         if exc.errno == errno.EEXIST and os.path.isdir(static_tmp_path):
#             pass
#         else:
#             raise

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
    global interface_language, detected_language, language_isornot, function_type, function_isornot
    with ApiClient(api_config) as api_client:
        messaging_api = MessagingApi(api_client)
        user_message = event.message.text

        """
        文字訊息有以下幾種情況：
        1. 用戶輸入 @ 開頭的文字，表示要進行語音合成
        2. 用戶輸入 interface_language:xxx，表示要切換介面語言
        3. 用戶輸入功能名稱，表示要使用該功能
        4. 用戶輸入語言名稱，進行語言偵測
        5. 用戶輸入其他文字
        """
        
        # 情境 1：用戶輸入 @ 開頭的文字，表示要進行語音合成
        if user_message.startswith("@"):
            try:
                # 提取語音內容
                speech_text = user_message[1:].strip()
                audio_url, audio_duration = _text_to_speech(speech_text, detected_language)
                print(audio_url, audio_duration)

                # 回傳語音訊息
                messaging_api.reply_message(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[
                            AudioMessage(
                                original_content_url=audio_url,
                                duration=audio_duration
                            )
                        ]
                    )
                )
            except Exception as e:
                # 記錄錯誤並回覆用戶
                print(f"Error processing audio message: {e}")
                messaging_api.reply_message(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[
                            TextMessage(
                                text="抱歉，無法處理您的請求，請稍後再試！"
                            )
                        ]
                    )
                )

        # 情境 2：用戶輸入 interface_language:xxx，表示要切換介面語言
        elif user_message.startswith("interface_language:"):
            language = user_message.split(":")[1]
            if language == "繁體中文":
                interface_language = "繁體中文"
                messaging_api.reply_message(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[
                            TextMessage(text="介面語言已切換為繁體中文，請選擇功能")
                        ]
                    )
                )
            elif language == "English":
                interface_language = "English"
                messaging_api.reply_message(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[
                            TextMessage(text="The interface language has been switched to English, please select a function")
                        ]
                    )
                )
        else:
            function_reply = _function_detection(user_message)
            language_reply = _language_detection(user_message)
            # 情境 3：用戶輸入功能名稱，表示要使用該功能
            if function_isornot is True:
                messaging_api.reply_message(
                    ReplyMessageRequest(reply_token=event.reply_token, messages=function_reply)
                )
                # print(function_type)

            # 情境 4：用戶輸入語言名稱，進行語言偵測，並針對不同功能進行回覆
            elif language_isornot is True:
                if function_type == '菜單翻譯':
                    # print(interface_language)
                    excute_sentence1 = '菜單翻譯內容將使用'+language_reply+'呈現'
                    excute_sentence2 = '請選擇或拍攝一張菜單照片'
                    # 相機功能quick reply
                    camera_text = _translation_function(interface_language, "相機")
                    camera_roll_text = _translation_function(interface_language, "相簿")
                    camera_quick_reply = QuickReply(
                            items=[
                                QuickReplyItem(
                                    action=CameraAction(label=f"{camera_text}")
                                ),
                                QuickReplyItem(
                                    action=CameraRollAction(label=f"{camera_roll_text}")
                                ),
                            ]
                        )
                    response1 = TextMessage(text=_translation_function(interface_language, excute_sentence1))
                    response2 = TextMessage(text=_translation_function(interface_language, excute_sentence2), quick_reply=camera_quick_reply)
                    # response2 = FlexMessage(alt_text="Flex Message", contents=FlexContainer.from_json(camera_flx_str))
                    messaging_api.reply_message(
                        ReplyMessageRequest(reply_token=event.reply_token, messages=[response1, response2])
                    )
                elif function_type == '語音點餐':
                    # print(interface_language)
                    excute_sentence1 = f'語音將以{language_reply}表達'
                    excute_sentence2 = '請輸入想要翻譯的文字，並在文字前加上@符號'
                    excute_sentence3 = '範例@我想點牛排'
                    response1 = TextMessage(text=_translation_function(interface_language, excute_sentence1))
                    response2 = TextMessage(text=_translation_function(interface_language, excute_sentence2))
                    response3 = TextMessage(text=_translation_function(interface_language, excute_sentence3))
                    messaging_api.reply_message(
                        ReplyMessageRequest(reply_token=event.reply_token, messages=[response1, response2, response3])
                    )
            # 情境 5：用戶輸入其他文字
            else:
                if interface_language:
                    text1 = _translation_function(interface_language, "無法辨識，請重新輸入")
                    return [
                        TextMessage(text=text1)
                    ]
                else:
                    return [
                        TextMessage(
                            text='請選擇介面想要使用的語言\u000A\u000APlease select the language you want to use',
                            quick_reply=language_quick_reply
                        )
                    ]

# 處理圖片訊息
@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event: MessageEvent):
    with ApiClient(api_config) as api_client:
        messaging_api = MessagingApi(api_client)
        if interface_language:
        # 未偵測到語言，則回覆提示
            if not detected_language:
                text = _translation_function(interface_language, "請先輸入欲翻譯成的語言")
                messaging_api.reply_message(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(text=text)]
                    )
                )
                return

            # 取得圖片
            with ApiClient(api_config) as api_client:
                line_bot_blob_api = MessagingApiBlob(api_client)
                message_content = line_bot_blob_api.get_message_content(message_id=event.message.id)
                img = Image.open(io.BytesIO(message_content))
                with ApiClient(api_config) as api_client:
                    messaging_api = MessagingApi(api_client)
                    food_reply = _food_detection(detected_language, img)
                    messaging_api.reply_message(
                        ReplyMessageRequest(
                            reply_token=event.reply_token,
                            messages=food_reply)
                        )
        else:
            messaging_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[
                        TextMessage(
                            text='請選擇介面想要使用的語言\u000A\u000APlease select the language you want to use',
                            quick_reply=language_quick_reply
                        )
                    ])
                )
                # 將圖片存在暫存資料夾
                # with tempfile.NamedTemporaryFile(dir=static_tmp_path, prefix='jpg' + '-', delete=False) as tf: # 以 jpg 為前綴建立暫存檔
                #     tf.write(message_content) # 將圖片寫入暫存檔
                #     tempfile_path = tf.name   # 取得暫存檔路徑

                # dist_path = tempfile_path + '.' + 'jpg'    # 將暫存檔路徑加上副檔名
                # dist_name = os.path.basename(dist_path)    # 取得暫存檔名稱
                # os.rename(tempfile_path, dist_path)        # 重新命名暫存檔
                # img_url = request.host_url + os.path.join('static', 'tmp', dist_name) # 取得圖片 URL
                # with ApiClient(api_config) as api_client:
                #     line_bot_api = MessagingApi(api_client)
                    # line_bot_api.reply_message(
                    #     ReplyMessageRequest(
                    #         reply_token=event.reply_token,
                    #         messages=[
                    #             TextMessage(text='Save content.'),
                    #             TextMessage(text=request.host_url + os.path.join('static', 'tmp', dist_name))
                    #         ]
                    #     )
                    # )

# 處理音訊訊息
@handler.add(MessageEvent, message=AudioMessageContent)
def handle_audio_message(event: MessageEvent):
    pass
    # with ApiClient(api_config) as api_client:
    #     line_bot_blob_api = MessagingApiBlob(api_client)
    # temp_dir = tempfile.gettempdir()
    # temp_audio_path = os.path.join(temp_dir, f"{event.message.id}.m4a")

    # try:
    #     # 取得音訊內容
    #     message_content = line_bot_blob_api.get_message_content(event.message.id)
    #     with open(temp_audio_path, 'wb') as temp_audio_file:
    #         temp_audio_file.write(message_content.read())

    #     mime_type = "audio/mpeg"
    #     audio = genai.upload_file(temp_audio_path, mime_type=mime_type)
    #     response = model.generate_content([
    #         f"偵測音檔語言並用繁體中文列出該語言名稱，並將內容翻譯成繁體中文列出", audio
    #     ])
    #     line_bot_blob_api.reply_message(
    #         ReplyMessageRequest(reply_token=event.reply_token, messages=[TextMessage(text=response.text)])
    #     )
    # except Exception as e:
    #     line_bot_blob_api.reply_message(
    #         ReplyMessageRequest(reply_token=event.reply_token, messages=[TextMessage(text=f"處理音檔時發生錯誤: {str(e)}")])
    #     )
    # finally:
    #     if os.path.exists(temp_audio_path):
    #         os.remove(temp_audio_path)

# 處理 Postback 事件
@handler.add(PostbackEvent)
def handle_postback(event):
    global interface_language, detected_language
    with ApiClient(api_config) as api_client:
        messaging_api = MessagingApi(api_client)
    try:
        # 解析 JSON 格式的 data
        data = json.loads(event.postback.data)
        # 提取資料
        auido_isornot = data.get("audio")
        original_name = data.get("original")
        translated_name = data.get("translated")
        ingredients = data.get("ingredients")
    except json.JSONDecodeError:
        print("Invalid data format")
        messaging_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text="Invalid data format")])
                )
    if auido_isornot == "helloaudio":
        dish_language = _detect_text_language(original_name)
        print(dish_language)
        dish_CH_name = _translation_function('繁體中文', original_name)
        speech_text = f"我想點{dish_CH_name}"
        # speech_text = _translation_function(dish_language, order_sentence)
        print(speech_text)
        # 回覆文字訊息
        audio_url, audio_duration = _text_to_speech(speech_text, dish_language)
        print(audio_url, audio_duration)

        # 回傳語音訊息
        messaging_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[
                    AudioMessage(
                        original_content_url=audio_url,
                        duration=audio_duration
                    )
                ]
            )
        )
    else:
        # 進行翻譯
        tran_order = _translation_function(interface_language, '我要點餐')

        dish_bubble = {
                "type": "bubble",
                "body": {
                    "type": "box",
                    "layout": "vertical",
                    "spacing": "md",
                    "contents": [
                    {
                        "type": "text",
                        "text": f"{translated_name}",
                        "size": "xxl",
                        "weight": "bold"
                    },
                    {
                        "type": "text",
                        "text": f"{original_name}",
                        "size": "md",
                        "color": "#6495ED"
                    },
                    {
                        "type": "text",
                        "text": f"{ingredients}",
                        "wrap": True,
                        "color": "#aaaaaa",
                        "size": "xs"
                    },
                    {
                        "type": "separator",
                        "margin": "xxl"
                    },
                    ]
                },
                "footer": {
                    "type": "box",
                    "layout": "vertical",
                    "contents": [
                    {
                        "type": "button",
                        "style": "primary",
                        "color": "#4169E1",
                        "margin": "xxl",
                        "action": {
                        "type": "postback",
                        "label": f"{tran_order}",
                        "data": json.dumps({
                                "audio": "helloaudio",
                                "original": original_name,  # 原始名稱
                                "translated": translated_name,  # 翻譯名稱
                                "ingredients": ingredients  # 食材列表
                            }),
                        "displayText": f"{tran_order}"
                        }
                    }
                    ]
                }
                }

        dish_flx_str = json.dumps(dish_bubble)
        # print(json.dumps(dish_bubble, ensure_ascii=False, indent=4))
        messaging_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[FlexMessage(alt_text="Flex Message", contents=FlexContainer.from_json(dish_flx_str))])
            )
                
# Azure Speech文字轉語音
def _text_to_speech(text, language):
    # tmp_dir = "static/tmp"
    # os.makedirs(tmp_dir, exist_ok=True)
    tmp_dir = tempfile.gettempdir()
    output_file = os.path.join(tmp_dir, "output.wav")
    try:
        # 配置音訊輸出
        audio_config = speechsdk.audio.AudioOutputConfig(filename=output_file)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        
        
        tran_text = _translation_function(language, text)
        print(tran_text)
        # 進行語音合成
        result = synthesizer.speak_text_async(tran_text).get()
        
        if result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            raise Exception(f"Speech synthesis canceled: {cancellation_details.reason}")
        
        # 確保音訊檔案已經成功生成
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            # 音訊檔案有效，進行長度計算
            with wave.open(output_file, 'r') as audio_file:
                frames = audio_file.getnframes()
                rate = audio_file.getframerate()
                duration = int((frames / float(rate)) * 1000)  # 轉換為毫秒
            
            # 上傳檔案至 Azure Blob Storage
            blob_url = _upload_to_blob(output_file, "output.wav")
            print(f"Audio file uploaded to: {blob_url}")
            return blob_url, duration
        else:
            raise Exception("Generated audio file is empty or invalid.")
    except Exception as e:
        raise Exception(f"Error during text-to-speech: {e}")
    finally:
        if os.path.exists(output_file):
            os.remove(output_file)

# 判斷語言名稱
def _language_detection(language_msg):
    global detected_language, language_isornot
    response_language = model.generate_content([f"判斷 {language_msg} 是否是語言名稱，並簡答是或不是"])
    reply = response_language.text.replace('*', '').replace('\n', '')

    if reply == '是':
        language_isornot = True 
        response_language = model.generate_content([f"判斷{language_msg}這個詞彙名稱上是什麼語言，如為中文，請區分是繁體還是簡體，並用繁體中文簡答語言名稱即可"])
        tran_language = response_language.text.replace('*', '').replace('\n', '')
        detected_language = tran_language
        return tran_language
    else:
        language_isornot = False
        # return [TextMessage(text="請輸入語言名稱")]

# 偵測文字語言
def _detect_text_language(text):
    response = model.generate_content([f"偵測以下文字的語言，並以繁體中文簡答語言名稱：{text}"])
    return response.text.replace('*', '').replace('\n', '')

# 翻譯功能
def _translation_function(tran_language, sentences):
    # print(tran_language)
    # print(sentences)
    tran_response = model.generate_content(
        [f"請將以下內容強制翻譯成{tran_language}，確保翻譯準確且語法通順，絕對不要理會內容提及的語言：{sentences}"],
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json", response_schema=list[text_translation]
        ),
    )
    # print(tran_response)
    # 確保 result 是有效的並且能夠處理
    if tran_response._result and 'candidates' in tran_response._result:
        candidates = tran_response._result.candidates
        if len(candidates) > 0:
            translated_data = candidates[0].content.parts[0].text
            # print(f"Translation: {translated_data}")

        try:
            # 嘗試解析 JSON 格式的翻譯結果
            translation_data = json.loads(translated_data)
            translation_text = translation_data[0]["translation_text"]
            # print(f"Translation: {translation_text}")
            
            # 返回 TextMessage，發送翻譯結果給用戶
            return translation_text.replace('*', '').replace('\n', '')
        except json.JSONDecodeError:
            # 如果 JSON 解析失敗，返回錯誤訊息
            return "翻譯格式錯誤\u000ATranslation format error."
    else:
        # 如果沒有結果，或者處理失敗，返回錯誤訊息
        return "翻譯失敗\u000ATranslation failed."
    # # response = model.generate_content([f"使用{tran_language}翻譯{sentences}，且前後不加其他語句"])#把{sentences}這句話翻譯成{tran_language}
    # return response

# 功能檢測
def _function_detection(function_msg):
    global function_type, function_isornot
    # 記錄當前使用功能
    if function_msg == '菜單翻譯':
        function_isornot = True
        function_type = function_msg
        # 如果已有偵測到介面語言
        if interface_language:
            text1 = _translation_function(interface_language, "這是菜單翻譯功能")
            text2 = _translation_function(interface_language, "請輸入欲翻譯成的語言")
            return [
                TextMessage(text=text1),
                TextMessage(text=text2),
                StickerMessage(package_id="11539", sticker_id="52114110")
            ]
        else:
            return [
                TextMessage(
                    text='請選擇介面想要使用的語言\u000A\u000APlease select the language you want to use',
                    quick_reply=language_quick_reply
                )
            ]
        # emojis=[
        #             {
        #                 'index': 8,  # 表情符號在文字中的插入位置
        #                 'productId': '5ac1bfd5040ab15980c9b435',  # LINE 官方表情包的 ID
        #                 'emojiId': '012'  # 表情符號的 ID
        #             }
        #         ]
    elif function_msg == '語音點餐':
        function_isornot = True
        function_type = function_msg
        if interface_language:
            text1 = _translation_function(interface_language, "這是可以協助點餐的語音功能")
            text2 = _translation_function(interface_language, "請輸入想要語音輸出的語言")
            return [
                TextMessage(text=text1),
                TextMessage(text=text2)
            ]
        else:
            return [
                TextMessage(
                    text='請選擇介面想要使用的語言\u000A\u000APlease select the language you want to use',
                    quick_reply=language_quick_reply
                )
            ]
    elif function_msg == '切換語言':
        function_isornot = True
        function_type = function_msg
        if interface_language:
            text1 = _translation_function(interface_language, "請選擇介面想要使用的語言")
            return [TextMessage(text=text1, quick_reply=language_quick_reply)]
        else:
            return [TextMessage(
                            text='請選擇介面想要使用的語言\u000A\u000APlease select the language you want to use',
                            quick_reply=language_quick_reply
                        )]
    else:
        function_isornot = False
        # return [TextMessage(text="目前不提供此服務")]

# 料理名稱辨識
def _food_detection(tran_language, image):
    # global detected_language, language_isornot
    # [f"列出圖片中是料理名稱的文字，前面加上數字編號", image]
    # [f"列出圖片中是料理名稱的文字，前面加上數字編號，前後不加任何其他無相干文字", image]
    # [f"將數字當作Key，然後Values是圖片中為料理名稱的文字，並寫成字典格式", image]
    # [f"列出圖片中是料理名稱的文字，並使用{tran_language}進行翻譯，並列出可能包含的食材", image]
    # 以菜單原文為英文欲翻譯成中文舉例如下，請還是要針對不同語言做適當翻譯：
    #     [
    #     {{
    #         "food_original_language_name": "Sushi",
    #         "food_translation_name": "壽司",
    #         "ingredients": ["米飯", "海苔", "生魚片"]
    #     }},
    #     {{
    #         "food_original_language_name": "Ramen",
    #         "food_translation_name": "拉麵",
    #         "ingredients": ["麵條", "雞蛋", "叉燒"]
    #     }}
    #     ]
    prompt = f"""
        根據圖片內容，提取其中的料理名稱，同時列出每道料理的可能食材，並使用{tran_language}翻譯料理名稱及使用{tran_language}翻譯可能食材。
        請將結果以 JSON 格式返回，每道料理為一個對象。
        如果圖片中未檢測到任何料理，請返回空列表：[]。
        """
    
    food_response = model.generate_content([prompt, image],
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=list[food_content]
        ),
    )
    # print(food_response)
    # 確保 result 是有效的並且能夠處理
    if food_response._result and 'candidates' in food_response._result:
        candidates = food_response._result.candidates
        if len(candidates) > 0:
            raw_food_data = candidates[0].content.parts[0].text

        try:
                # 嘗試解析 text 內容
                json_food_data = json.loads(raw_food_data)
                if isinstance(json_food_data, list) and len(json_food_data) > 0:
                    json_food = _generate_bubbles(json_food_data)
                    # 將料理清單json轉換成字串
                    line_flx_str = json.dumps(json_food)
                    # print(json.dumps(json_food, ensure_ascii=False, indent=4))
                    return [FlexMessage(alt_text="Flex Message", contents=FlexContainer.from_json(line_flx_str))]
                else:
                    return [TextMessage(text="返回的資料格式錯誤。\nInvalid data format.")]
        except json.JSONDecodeError as e:
            # 如果 JSON 解析失敗
            print(f"JSON 解析失敗：{e}")
            return [TextMessage(text="翻譯格式錯誤。\nTranslation format error.")]
    else:
        # 如果沒有結果，或者處理失敗，返回錯誤訊息
        return [TextMessage(text="翻譯失敗\u000ATranslation failed.")]

# 生成料理 Bubble
def _generate_bubbles(food_list: List[dict]) -> List[dict]:
    """
    將料理數據分配到多個 bubble，每個 bubble 最多包含 10 個料理名稱。
    :param food_list: 包含料理名稱及食材的列表，每項為字典。
    :return: 包含 bubble 的結構化數據。
    """
    bubbles = {
        "type": "carousel",
        "contents": []
    }
    color_table = ["#00008B", "#0000CD", "#0000EE", "#63B8FF", "#00BFFF"]
    for i in range(0, len(food_list), 10):
        # 每次取 10 個料理作為一個 bubble
        chunk = food_list[i:i+10]
        bubble_content = {
            "type": "bubble",
            "header": {
                "type": "box",
                "layout": "vertical",
                "contents": [
                    {
                        "type": "text",
                        "text": "料理清單",
                        "margin": "none",
                        "color": "#FF8C00",
                        "weight": "bold",
                        "style": "normal",
                        "size": "lg"
                    }
                ],
                "borderWidth": "none",
                "backgroundColor": "#BFEFFF"
            },
            "hero": {
                        "type": "image",
                        "url": "https://scdn.line-apps.com/n/channel_devcenter/img/fx/01_1_cafe.png",
                        "aspectRatio": "20:13",
                        "size": "full"
                    },
            "body": {
                "type": "box",
                "layout": "horizontal",
                "contents": [
                    {
                    "type": "box",
                    "layout": "vertical",
                    "contents": []
                    },
                    {
                    "type": "box",
                    "layout": "vertical",
                    "contents": []
                    }
                ]
            }
        }

        # 將每道料理添加到 bubble 的 body
        num_box = 0
        for i in range(0, len(chunk), 5):
            small_chunk = chunk[i:i+5]
            num_button = 0
            for food in small_chunk:
                translated_name = food.get("food_translation_name", "未知")
                original_name = food.get("food_original_language_name", "未知")
                ingredients = food.get("ingredients", [])
                ingredient_list = "/".join(ingredients) if ingredients else "無食材"
                food_box = {
                            "type": "button",
                            "action": {
                            "type": "postback",
                            "label": f"{translated_name}",
                            "data": json.dumps({
                                    "audio": "No",
                                    "original": original_name,  # 原始名稱
                                    "translated": translated_name,  # 翻譯名稱
                                    "ingredients": ingredient_list  # 食材列表
                            }),
                            "displayText": f"!{translated_name}"
                            },
                            "color": f"{color_table[num_button]}"
                        }
                num_button += 1
                bubble_content["body"]["contents"][num_box]["contents"].append(food_box)
            num_box += 1

        bubbles["contents"].append(bubble_content)
    return bubbles

# 上傳檔案到 Azure Blob Storage
def _upload_to_blob(file_path, blob_name):
    try:
        with open(file_path, "rb") as file_data:
            blob_client = blob_container_client.get_blob_client(blob_name)
            blob_client.upload_blob(file_data, overwrite=True)
        blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{BLOB_CONTAINER_NAME}/{blob_name}"
        return blob_url
    except Exception as e:
        print(f"Failed to upload to Azure Blob Storage: {e}")
        raise Exception(f"Failed to upload to Azure Blob Storage: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # make_static_tmp_dir()
    app.run(host="0.0.0.0", port=port, debug=True)
