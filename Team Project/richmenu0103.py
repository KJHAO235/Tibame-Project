
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    MessagingApiBlob,
    RichMenuRequest,
    RichMenuArea,
    RichMenuSize,
    RichMenuBounds,
    URIAction,
    RichMenuSwitchAction,
    CreateRichMenuAliasRequest,
    PostbackAction,
    TextMessage,
    MessageAction,
)


def rich_menu_object_json():
    return {
        "size": {
            "width": 2500,
            "height": 1686
        },
        "selected": False,
        "name": "功能在這裡喔",
        "chatBarText": "你好？",
        "areas": [
            {   # 左下：搜尋
                "bounds": {
                    "x": 0,
                    "y": 844,
                    "width": 833,
                    "height": 843
                },
                "action": {
                    "type": "postback",
                    "data": "search",
                    "inputOption": "openKeyboard"
                }
            },
            {   # 中下：推薦
                "bounds": {
                    "x": 834,
                    "y": 844,
                    "width": 833,
                    "height": 843
                },
                "action": {
                    "type": "postback",
                    "data": "recommend",
                }
            },
            {  # 右下角：「可刪除」我的最愛
                "bounds": {
                    "x": 2261,
                    "y": 1447,
                    "width": 240,
                    "height": 240
                },
                "action": {
                    "type": "postback",
                    "data": "show_del_both",
                }
            },
            {  # 右下：我的最愛
                "bounds": {
                    "x": 1667,
                    "y": 844,
                    "width": 833,
                    "height": 843
                },
                "action": {
                    "type": "postback",
                    "data": "show_both",
                }
            },
            {  # 右上邊：管理者面板
                "bounds": {
                    "x": 2400,
                    "y": 0,
                    "width": 100,
                    "height": 100
                },
                "action": {
                    "type": "message",
                    # "richMenuAliasId": "richmenu-alias-admin",
                    # "data": "richmenu-changed-to-admin"
                    "text": "Administrator"
                }
            },
            {  # 上：展覽
                "bounds": {
                    "x": 0,
                    "y": 0,
                    "width": 2500,
                    "height": 843
                },
                "action": {
                    "type": "postback",
                    "data": "quick",
                    "inputOption": "closeRichMenu"
                }
            },
        ]
    }


def rich_menu_object_admin_json():
    return {
        "size": {
            "width": 2500,
            "height": 1686
        },
        "selected": False,
        "name": "這是管理介面",
        "chatBarText": "你好？",
        "areas": [
            {   # 左：
                "bounds": {
                    "x": 0,
                    "y": 0,
                    "width": 833,
                    "height": 1686
                },
                "action": {
                    "type": "postback",
                    "data": "search",
                }
            },
            {   # 中：
                "bounds": {
                    "x": 834,
                    "y": 0,
                    "width": 833,
                    "height": 1686
                },
                "action": {
                    "type": "postback",
                    "data": "recommend",
                }
            },
            {  # 右下：
                "bounds": {
                    "x": 1667,
                    "y": 0,
                    "width": 833,
                    "height": 1686
                },
                "action": {
                    "type": "postback",
                    "data": "show_both",
                }
            }
        ]
    }


def create_action(action):
    # 根據 action 的 type，創造對應的 Action 物件
    if action['type'] == 'uri':
        return URIAction(uri=action.get('uri'))
    elif action['type'] == 'postback':
        return PostbackAction(data=action.get('data'))
    elif action['type'] == 'message':
        return MessageAction(text=action.get('text'))
    else:
        return RichMenuSwitchAction(
            rich_menu_alias_id=action.get('richMenuAliasId'),
            data=action.get('data')
        )

# 設定預設的 RichMenu
def rich_menu_set(configuration):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_blob_api = MessagingApiBlob(api_client)

        # 呼叫模板
        rich_menu_object = rich_menu_object_json()

        # 建立 RichMenu 裡的 Area 物件，將要放入 RichMenuRequest 物件中
        areas = [
            RichMenuArea(
                bounds=RichMenuBounds(
                    x=info['bounds']['x'],
                    y=info['bounds']['y'],
                    width=info['bounds']['width'],
                    height=info['bounds']['height']
                ),
                action=create_action(info['action'])
            ) for info in rich_menu_object['areas']
        ]

        # 建立 RichMenuRequest 物件
        rich_menu_to_a_create = RichMenuRequest(
            size=RichMenuSize(width=rich_menu_object['size']['width'],
                                height=rich_menu_object['size']['height']),
            selected=rich_menu_object['selected'],
            name=rich_menu_object['name'],
            chat_bar_text=rich_menu_object['name'],
            areas=areas
        )

        # 創造 RichMenu，並取得 rich_menu_id
        rich_menu_a_id = line_bot_api.create_rich_menu(
            rich_menu_request=rich_menu_to_a_create
        ).rich_menu_id

        # 上傳 RichMenu 的圖片，根據 rich_menu_id
        with open('rich01_240110.png', 'rb') as image:
            line_bot_blob_api.set_rich_menu_image(
                rich_menu_id=rich_menu_a_id,
                body=bytearray(image.read()),
                _headers={'Content-Type': 'image/png'}
            )

        # 設定預設的 RichMenu
        line_bot_api.set_default_rich_menu(rich_menu_id=rich_menu_a_id)

        # 創造 RichMenu 別名
        # alias_a = CreateRichMenuAliasRequest(
        #     rich_menu_alias_id='richmenu-alias-a',
        #     rich_menu_id=rich_menu_a_id
        # )
        # line_bot_api.create_rich_menu_alias(alias_a)


def link_rich_menu_to_user(configuration, user_id, rich_menu_id):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.link_rich_menu_id_to_user(user_id=user_id,
                                                rich_menu_id=rich_menu_id)