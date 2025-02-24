import user_state
import io
from azure_blob_service import upload_file_to_blob
from azure_service import text_to_speech
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApiBlob,
    ReplyMessageRequest,
    TextMessage,
    StickerMessage,
    AudioMessage,
    FlexMessage,
    FlexContainer,
)
import configparser
import user_state
from gemini_service import food_detection, translation_function
from set_structure import change_language, camera, dish_bubble, speech_language_detection
from PIL import Image
import json

# Config Parser
config = configparser.ConfigParser()
config.read('config.ini')

# 初始化 Line Messaging API 客戶端 get
try:
    LINE_CHANNEL_ACCESS_TOKEN = config.get('Line', 'CHANNEL_ACCESS_TOKEN')
    LINE_CHANNEL_SECRET = config.get('Line', 'CHANNEL_SECRET')
    # 初始化 Messaging API 客戶端
    line_config = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
except configparser.NoSectionError:
    print("Please provide Line section in config.ini")
    exit()

def process_message(event, user_id):
    # 確保用戶狀態存在
    user_info = user_state.ensure_user_state_in_firestore(user_id)
    
    # 處理訊息
    if event.type == "message":
        
        # 處理文字訊息
        if event.message.type == "text":
            text = event.message.text
            if user_info.get("state") == "speech_translator":
                voice_type = user_info.get("speech_info")[1]
                language = user_info.get("speech_info")[0]
                translation_text = translation_function(language, text)
                audio_url, audio_duration = text_to_speech(translation_text, voice_type, user_id)
                audio_message = AudioMessage(original_content_url=audio_url, duration=audio_duration)
                return ReplyMessageRequest(reply_token=event.reply_token,messages=[audio_message])
            else:
                response = TextMessage(text='不接受此服務')
                return ReplyMessageRequest(reply_token=event.reply_token, messages=[response])
        
        # 處理影像訊息
        elif event.message.type == "image":
            user_state.update_user_state_in_firestore(user_id, {"state": "menu_translator"})
            # 處理影像訊息
            with ApiClient(line_config) as api_client:
                line_bot_blob_api = MessagingApiBlob(api_client)
                message_content = line_bot_blob_api.get_message_content(message_id=event.message.id)
                # 讀取影像的二進制內容
                image_bytes = io.BytesIO(message_content)  # 使用 .read() 提取資料
                image_bytes.seek(0)
                # 上傳影像到 Azure Blob
                try:
                    blob_url = upload_file_to_blob(image_bytes, user_id, "jpg")
                except Exception as e:
                    print(f"Failed to upload image: {e}")
                    response = TextMessage(text='圖片無法辨識，請重新上傳')
                    return ReplyMessageRequest(reply_token=event.reply_token, messages=[response])
                user_state.update_user_state_in_firestore(user_id, {"image_url": blob_url})
                # try:
                img = Image.open(io.BytesIO(message_content))
                print(user_info.get("image_info"))
                food_reply = food_detection(img, user_id, user_info.get("image_info"))
                return ReplyMessageRequest(reply_token=event.reply_token, messages=food_reply)
                # except Exception as e:
                #     print(f"Failed to detect food: {e}")
                #     response = TextMessage(text='請重新上傳圖片')
                #     return ReplyMessageRequest(reply_token=event.reply_token, messages=[response])
                
        # 處理垃圾訊息
        else:
            raise ValueError("Unsupported message type")
        
    # 處理 Postback 訊息
    elif event.type == "postback":
        postback_data = event.postback.data

        # 菜單翻譯功能
        if postback_data == 'menu_translator':
            user_state.update_user_state_in_firestore(user_id, {"state": "menu_translator"})
            # print(user_info)
            if not user_info.get("image_info"):
                user_state.update_user_state_in_firestore(user_id, {"image_info": '繁體中文'})
            language = user_info.get("image_info")
            camera_quick_reply = camera()
            response = TextMessage(text=f'翻譯語言已設定為{language}，請選擇或拍攝一張菜單', quick_reply=camera_quick_reply)
            return ReplyMessageRequest(reply_token=event.reply_token, messages=[response])
        
        # 語音翻譯功能
        elif postback_data == 'speech_translator':
            user_state.update_user_state_in_firestore(user_id, {"state": 'speech_translator'})
            pass
            speech_language_quick_reply = change_language()
            response = TextMessage(text='請選擇語音輸出的語言', quick_reply=speech_language_quick_reply)
            return ReplyMessageRequest(reply_token=event.reply_token, messages=[response])
        
        # 語言切換功能
        elif postback_data == 'change_language':
            if user_info.get("state") == "":
                response = TextMessage(text='請先選擇其他功能')
            change_language_quick_reply = change_language()
            response = TextMessage(text='請選擇語言', quick_reply=change_language_quick_reply)
            return ReplyMessageRequest(reply_token=event.reply_token, messages=[response])
        
        # 語音輸出語言設定
        elif postback_data.startswith("change_language="):
            if user_info.get("state") == "speech_translator":
                language = postback_data.split("=")[1]
                voice_type = postback_data.split("=")[2]
                user_state.update_user_state_in_firestore(user_id, {"speech_info": [language, voice_type]})
                response = TextMessage(text=f'語音輸出語言已設定為{language}，請輸入文字')
                return ReplyMessageRequest(reply_token=event.reply_token, messages=[response])
            elif user_info.get("state") == "menu_translator":
                language = postback_data.split("=")[1]
                user_state.update_user_state_in_firestore(user_id, {"image_info": language})
                camera_quick_reply = camera()
                response = TextMessage(text=f'翻譯語言已設定為{language}，請選擇或拍攝一張菜單', quick_reply=camera_quick_reply)
                return ReplyMessageRequest(reply_token=event.reply_token, messages=[response])
            else:
                response = TextMessage(text='請重新選擇功能')
                return ReplyMessageRequest(reply_token=event.reply_token, messages=[response])
            
        # 料理選單
        elif postback_data.startswith("food_button="):
            food_id = postback_data.split("=")[1]
            print(food_id)
            food_list = user_info.get("food_list")
            result = next((item for item in food_list if item["food_id"] == food_id), None)
            if result:
                ingredients = result.get("ingredients", [])
                ingredient_list = "/".join(ingredients) if ingredients else "無食材"
                translated_name = result.get("food_translation_name", "未知")
                original_name = result.get("food_original_language_name", "未知")
                language = user_info.get("image_info")
                order_word = translation_function(language, '我要點餐')
                dish = dish_bubble(original_name, translated_name, ingredient_list, order_word, food_id)
                dish_flx_str = json.dumps(dish)
                content = FlexContainer.from_json(dish_flx_str)
                return ReplyMessageRequest(reply_token=event.reply_token,messages=[FlexMessage(alt_text="Flex Message", contents=content)])
            else:
                response = TextMessage(text='未找到料理')
                return ReplyMessageRequest(reply_token=event.reply_token, messages=[response])
        
        # 料理點餐
        elif postback_data.startswith("order_button="):
            food_id = postback_data.split("=")[1]
            food_list = user_info.get("food_list")
            result = next((item for item in food_list if item["food_id"] == food_id), None)
            if result:
                original_name = result.get("food_original_language_name", "未知")
                # original_language = detect_text_language(original_name)
                language_msg = user_info.get("menu_language")
                original_language = speech_language_detection(language_msg)
                food_ch_name = translation_function('繁體中文', original_name)
                ch_order_word = '我想點' + food_ch_name
                if original_language[0] == None:
                    response = TextMessage(text=f'{original_language[1]}')
                    return ReplyMessageRequest(reply_token=event.reply_token, messages=[response])
                else:
                    voice_order_word = translation_function(original_language[1], ch_order_word)
                    audio_url, audio_duration = text_to_speech(voice_order_word, original_language[0], user_id)
                    audio_message = AudioMessage(original_content_url=audio_url, duration=audio_duration)
                    return ReplyMessageRequest(reply_token=event.reply_token,messages=[audio_message])
            else:
                response = TextMessage(text='未找到料理')
                return ReplyMessageRequest(reply_token=event.reply_token, messages=[response])
            