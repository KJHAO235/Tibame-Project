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
    MessageAction
)
import configparser

# Config Parser
config = configparser.ConfigParser()
config.read('config.ini')

# 初始化 Line Messaging API 客戶端 get
try:
    LINE_CHANNEL_ACCESS_TOKEN = config.get('Line', 'CHANNEL_ACCESS_TOKEN')
    LINE_CHANNEL_SECRET = config.get('Line', 'CHANNEL_SECRET')
    # 初始化 Messaging API 客戶端
    configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
except configparser.NoSectionError:
    print("Please provide Line section in config.ini")
    exit()

def rich_menu_object_a_json():
    return {
        "size": {
        "width": 2500,
        "height": 1686
    },
    "selected": False,
    "name": "richmenu-a",
    "chatBarText": "功能選單:)",
    "areas": [
        {
            "bounds": {
                "x": 0,
                "y": 0,
                "width": 1666,
                "height": 1686
            },
            "action": {
                "type": "postback",
                "label": "Tap area A(left)",
                "data": "menu_translator",
            }
        },
        {
            "bounds": {
                "x": 1667,
                "y": 0,
                "width": 834,
                "height": 843
            },
            "action": {
                "type": "postback",
                "label": "Tap area B(up right)",
                "data": "speech_translator",
                "inputOption": "openKeyboard"
            }
        },
        {
            "bounds": {
                "x": 1667,
                "y": 844,
                "width": 834,
                "height": 843
            },
            "action": {
                "type": "postback",
                "label": "Tap area C(lower right)",
                "data": "change_language"
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
    
def main():
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_blob_api = MessagingApiBlob(api_client)

        # 2. Create rich menu A (richmenu-a)
        rich_menu_object_a = rich_menu_object_a_json()
        areas = [
            RichMenuArea(
                bounds=RichMenuBounds(
                    x=info['bounds']['x'],
                    y=info['bounds']['y'],
                    width=info['bounds']['width'],
                    height=info['bounds']['height']
                ),
                action=create_action(info['action'])
            ) for info in rich_menu_object_a['areas']
        ]

        rich_menu_to_a_create = RichMenuRequest(
            size=RichMenuSize(width=rich_menu_object_a['size']['width'],
                              height=rich_menu_object_a['size']['height']),
            selected=rich_menu_object_a['selected'],
            name=rich_menu_object_a['name'],
            chat_bar_text=rich_menu_object_a['chatBarText'],
            areas=areas
        )

        rich_menu_a_id = line_bot_api.create_rich_menu(
            rich_menu_request=rich_menu_to_a_create
        ).rich_menu_id

        # 3. Upload image to rich menu A
        with open('richmenu.png', 'rb') as image:
            line_bot_blob_api.set_rich_menu_image(
                rich_menu_id=rich_menu_a_id,
                body=bytearray(image.read()),
                _headers={'Content-Type': 'image/png'}
            )

        # 4. Set rich menu A to default
        line_bot_api.set_default_rich_menu(rich_menu_id=rich_menu_a_id)

        # 5. Create rich menu alias
        # alias_a = CreateRichMenuAliasRequest(
        #     rich_menu_id=rich_menu_a_id,
        #     alias_name='richmenu-a-alias'
        # )
        # line_bot_api.create_rich_menu_alias(alias_a)

        print("Rich menu created successfully")

if __name__ == "__main__":
    main()