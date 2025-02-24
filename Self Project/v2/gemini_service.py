import google.generativeai as genai
import configparser
import json
from typing import List
import typing_extensions as typing
from linebot.v3.messaging import (
    TextMessage,
    FlexMessage,
    FlexContainer,
)
from set_structure import menu_bubble
from azure_service import language_detection
import user_state 
import uuid  # 用於生成唯一 ID

# Config Parser
config = configparser.ConfigParser()
config.read('config.ini')

# 設定 Google Generative AI
genai.configure(api_key=config.get('Google', 'GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')

class Food_content(typing.TypedDict):
        food_original_language_name: str
        food_translation_name: str
        ingredients: list[str]

class Text_translation(typing.TypedDict):
        original_text: str
        translation_text: str

# 料理名稱辨識
def food_detection(image, user_id, tran_language):
    
    # prompt = f"""
    #     根據圖片內容，提取其中的料理名稱，同時列出每道料理的可能食材，並使用{tran_language}翻譯料理名稱及使用{tran_language}翻譯可能食材。
    #     請將結果以 JSON 格式返回，每道料理為一個對象。
    #     如果圖片中未檢測到任何料理，請返回空列表：[]。
    #     """
    
    prompt = f"""
                請分析提供的圖片，準確提取其中的料理名稱，並列出每道料理的可能食材。請確保：
                1. **料理名稱需同時包含原始語言與 {tran_language} 的翻譯**。
                2. **可能食材列表需包含翻譯後的食材名稱**。
                3. **結果必須以 JSON 格式返回，每道料理應符合以下結構**：
                [
                    {{
                        "food_original_language_name": "原始語言的料理名稱",
                        "food_translation_name": "翻譯後的料理名稱",
                        "ingredients": ["翻譯後的食材1", "翻譯後的食材2", ...]
                    }},
                    ...
                ]
                4. **若圖片中未檢測到任何料理，請返回空列表：[]**。
                5. **確保輸出格式為有效的 JSON，避免額外的註解或非 JSON 內容**。

                請務必嚴格遵守上述規則，確保輸出格式符合標準。
            """
    
    food_response = model.generate_content([prompt, image],
        generation_config=genai.GenerationConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=list[Food_content]
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
                # 生成唯一 food_id
                for food in json_food_data:
                    food["food_id"] = _generate_food_id(food["food_original_language_name"]) 
                menu_language = _detect_menu_language(json_food_data)
                if menu_language[0] == None:
                    user_state.update_user_state_in_firestore(user_id, {"food_list": json_food_data, "menu_language": menu_language[0]})
                # print(json_food_data, menu_language)
                user_state.update_user_state_in_firestore(user_id, {"food_list": json_food_data, "menu_language": menu_language[0]})
                print("成功儲存料理列表")
                json_food = _generate_bubbles(user_id)
                # 將料理清單json轉換成字串
                line_flx_str = json.dumps(json_food)
                # print(json.dumps(json_food, ensure_ascii=False, indent=4))
                return [FlexMessage(alt_text="Flex Message", contents=FlexContainer.from_json(line_flx_str))]
            else:
                return [TextMessage(text="返回的資料格式錯誤。")]
        except json.JSONDecodeError as e:
            # 如果 JSON 解析失敗
            print(f"JSON 解析失敗：{e}")
            return [TextMessage(text="翻譯格式錯誤。")]
    else:
        # 如果沒有結果，或者處理失敗，返回錯誤訊息
        return [TextMessage(text="翻譯失敗")]
    
# 翻譯功能
def translation_function(tran_language, sentences):
    # print(tran_language)
    # print(sentences)
    tran_response = model.generate_content(
        [f"請將以下內容強制翻譯成{tran_language}，確保翻譯準確且語法通順，絕對不要理會內容提及的語言：{sentences}"],
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json", response_schema=list[Text_translation]
        ),
    )
    # print(tran_response)
    # 確保 result 是有效的並且能夠處理
    if tran_response._result and 'candidates' in tran_response._result:
        candidates = tran_response._result.candidates
        if len(candidates) > 0:
            translated_data = candidates[0].content.parts[0].text

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

# 生成料理 Bubble
def _generate_bubbles(user_id) -> List[dict]:
    """
    將料理數據分配到多個 bubble，每個 bubble 最多包含 10 個料理名稱。
    :param food_list: 包含料理名稱及食材的列表，每項為字典。
    :return: 包含 bubble 的結構化數據。
    """
    user_info = user_state.ensure_user_state_in_firestore(user_id)
    food_list = user_info.get("food_list")
    menu_url = user_info.get("image_url")

    bubbles = {
        "type": "carousel",
        "contents": []
    }
    color_table = ["#00008B", "#0000CD", "#0000EE", "#63B8FF", "#00BFFF"]
    for i in range(0, len(food_list), 10):
        # 每次取 10 個料理作為一個 bubble
        chunk = food_list[i:i+10]
        bubble_content = menu_bubble(menu_url)

        # 將每道料理添加到 bubble 的 body
        num_box = 0
        for i in range(0, len(chunk), 5):
            small_chunk = chunk[i:i+5]
            num_button = 0
            for food in small_chunk:
                food_id = food.get("food_id")
                translated_name = food.get("food_translation_name", "未知")
                food_box = {
                            "type": "button",
                            "action": {
                            "type": "postback",
                            "label": f"{translated_name}",
                            "data": "food_button=" + f"{food_id}",
                            "displayText": f"{translated_name}"
                            },
                            "color": f"{color_table[num_button]}"
                        }
                num_button += 1
                bubble_content["body"]["contents"][num_box]["contents"].append(food_box)
            num_box += 1

        bubbles["contents"].append(bubble_content)
    return bubbles

# 生成唯一 food_id
def _generate_food_id(food_name):
    """ 產生唯一 food_id，可用 UUID 或哈希 """
    return str(uuid.uuid4())[:8]  # 取前 8 碼的 UUID

# 判斷文字語言
def _detect_text_language(text):
    prompt = f"請判斷以下文字的主要語言，並僅返回語言名稱（如：中文、英文、法文、日文等），不得包含其他內容。\n\n{text}"
    
    response = model.generate_content(
        [prompt],
        generation_config=genai.GenerationConfig(
            temperature=0.0,  # 使輸出穩定不變
            max_output_tokens=10  # 限制輸出長度，避免回傳額外內容
        )
    )

    # 清理並確保輸出格式
    detected_language = response.text.strip().replace('*', '').split("\n")[0]

    # 若偵測結果異常，則回傳 "未知"
    return detected_language if detected_language else "未知"

# 判斷菜單語言
def _detect_menu_language(food_list):
    # 取前五個料理名稱，並判斷菜單語言 
    first_five_foods = food_list[:5]
    food_names = [food.get("food_original_language_name", "未知") for food in first_five_foods]
    foodstr = ", ".join(food_names)
    print(type(foodstr), foodstr)
    ln_name, ln_code = language_detection([foodstr])
    if ln_name == "unknown" or ln_code == "unknown":
        print("無法判斷語言，返回預設值")
        return ["unknown", "unknown"]
    return [ln_code, ln_name]