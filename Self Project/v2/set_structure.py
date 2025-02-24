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
    PostbackAction
)

# 切換語言
def change_language():
    # inputOption="openKeyboard"
    return QuickReply(
                items=[
                    QuickReplyItem(
                        action=PostbackAction(label="繁體中文",
                                              data="change_language=繁體中文=zh-TW-HsiaoChenNeural",)
                    ),
                    QuickReplyItem(
                        action=PostbackAction(label="英文",
                                              data="change_language=英文=en-US-AvaNeural",)
                    ),
                    QuickReplyItem(
                        action=PostbackAction(label="日文",
                                              data="change_language=日文=ja-JP-NanamiNeural",)
                    ),
                    QuickReplyItem(
                        action=PostbackAction(label="西班牙文",
                                              data="change_language=西班牙文=es-ES-ElviraNeural",)
                    ),
                    QuickReplyItem(
                        action=PostbackAction(label="法文",
                                              data="change_language=法文=fr-FR-DeniseNeural",)
                    ),
                    QuickReplyItem(
                        action=PostbackAction(label="德文",
                                              data="change_language=德文=de-DE-KatjaNeural",)
                    ),
                    QuickReplyItem(
                        action=PostbackAction(label="韓文",
                                              data="change_language=韓文=ko-KR-SunHiNeural",)
                    ),
                ]
            )

# 語言判斷
def speech_language_detection(language_msg):
    lang_map = {
        "zh": ["zh-TW-HsiaoChenNeural", "繁體中文"],
        "en": ["en-US-AvaNeural", "英文"],
        "ja": ["ja-JP-NanamiNeural", "日文"],
        "es": ["es-ES-ElviraNeural", "西班牙文"],
        "fr": ["fr-FR-DeniseNeural", "法文"],
        "de": ["de-DE-KatjaNeural", "德文"],
        "ko": ["ko-KR-SunHiNeural", "韓文"],
    }
    return lang_map.get(language_msg[:2], [None, "目前不支援該語言語音服務"])  # 預設英文


# 相機相簿
def camera():
    return QuickReply(
                items=[
                    QuickReplyItem(
                        action=CameraAction(label='相機')
                    ),
                    QuickReplyItem(
                        action=CameraRollAction(label='相簿')
                    ),
                ]
            )

# menu_bubble
def menu_bubble(menu_url):
    return {
            "type": "bubble",
            "hero": {
                        "type": "image",
                        "url": f"{menu_url}",
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
# "header": {
#                 "type": "box",
#                 "layout": "vertical",
#                 "contents": [
#                     {
#                         "type": "text",
#                         "text": "料理清單",
#                         "margin": "none",
#                         "color": "#FF8C00",
#                         "weight": "bold",
#                         "style": "normal",
#                         "size": "lg"
#                     }
#                 ],
#                 "borderWidth": "none",
#                 "backgroundColor": "#BFEFFF"
#             },

# dish_bubble
def dish_bubble(original_name, translated_name, ingredients, order_word, food_id):
    return {
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
                        "label": f"{order_word}",
                        "data": "order_button=" + f"{food_id}",
                        "displayText": f"{order_word}"
                        }
                    }
                    ]
                }
                }
# f"{tran_order}"