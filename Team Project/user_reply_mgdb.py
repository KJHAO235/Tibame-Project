import os
import io
import json
from PIL import Image
from configparser import ConfigParser
from linebot.v3.messaging import (
    FlexMessage,
    FlexContainer,
    PostbackAction,
    QuickReply,
    QuickReplyItem,
    TextMessage,
    ImageMessage
)

from start_estab_mgdb import add_admin, remove_admin, is_admin, admin_collection
from qr_type import exhibitions, make_quick_reply_exhi, recommend_like_or_not
# from show_exhi import show_exhi, show_list
# from show_firm import show_firm, show_list_firm, show_both
from show_exhi_mgdb import show_exhi_mgdb, show_list_exhi_mgdb, show_list_exhi_del_mgdb
from show_firm_mgdb import (show_firm_mgdb, show_list_firm_mgdb, show_both_mgdb, 
                            show_list_firm_del_mgdb, show_both_del_mgdb)
from search_exhib import search_exhib_2
from richmenu0103 import rich_menu_set

# 推薦廠商
from show_recom import show_recom_firm_mgdb

# 上傳圖片到 GCS。by 豪
from upload2GCS import upload_image_to_gcs
from gemini_map_v1 import draw_path_on_image

# chikawa
from show_exhi_mgdb import show_exhi_mgdb_chikawa, chikawa
from road_cloud_v1 import calculate_shortest_path


# 載入環境變數
try:
    config = ConfigParser()
    config.read('config.ini', encoding='utf-8')
    ADMINISTRATOR_SECRET = config['LINE']['ADMINISTRATOR_SECRET']
except:
    print('載入config ADMINISTRATOR 密碼失敗')
    try:
        ADMINISTRATOR_SECRET = os.getenv('ADMINISTRATOR_SECRET')
    except:
        print('載入環境變數 ADMINISTRATOR 密碼失敗')





def user_reply_mgdb(user_info, event, configuration):
    user_id = event.source.user_id
    user_input = event.message.text.strip()

    user_status = user_info["status"]

    if user_status == "want_to_be_Admin":
        # 如果使用者輸入的是 Administrator 密碼
        if user_input == ADMINISTRATOR_SECRET:
            add_admin(user_id)
            reply_message = [TextMessage(text="您已進入管理模式。")]
        else:
            reply_message = [TextMessage(text="密碼錯誤！")]
        user_info["status"] = None


    elif user_status == "search_exhibition_by_keyword":
        # 搜尋展覽
        if user_input == "吉伊卡哇" or user_input == "ちいかわ":
            user_info["status"] = "吉伊卡哇"
            exhib_id = "677fa88a853cb68988c7f4e4"
            reply_message = [
                FlexMessage(alt_text="ちいかわ", 
                            contents=FlexContainer.from_dict(show_exhi_mgdb_chikawa([exhib_id]))),
                            ]
        else:
            # 搜尋展覽
            user_info["status"] = None
            # 使用 search_exhib_2(user_input)
            reply_message = [
                FlexMessage(alt_text="猜你想趣", 
                            contents=FlexContainer.from_dict(show_exhi_mgdb(search_exhib_2(user_input))),
                            quick_reply=recommend_like_or_not())
                            ]
    

    elif user_status is not None:
        user_status_code = user_status.split("@@@@@")[0]

        if user_status_code == "bring_me_to_01":
            # 已接收 user_input = 起點，接下來要傳送地圖
            if user_input == "取消" or user_input == "cancel":
                user_info["status"] = None
                reply_message = [TextMessage(text="帶我趣已取消。")]
            else:
                frim_id = user_status.split("@@@@@")[1]
                user_info["status"] = None # 結束導航，重置使用者狀態
                reply_message = [TextMessage(text=f"你的起點是：{user_input}\n你的目的地是：{frim_id}\n\n導航功能尚未開放！")]
        
        
        elif user_status_code == "show_recom_shop_01":
            # 已接收 user_input = 關於某展覽感興趣的關鍵字，接下來要傳送推薦店家資訊
            if user_input == "取消" or user_input == "cancel":
                user_info["status"] = None
                reply_message = [TextMessage(text="店家推薦取消。")]
            else:
                # user_info["status"] = "show_recom_shop_01" + "@@@@@" + data_data + "@@@@@" + exhi_name # 廠商id + 展覽名稱
                exhi_id = user_status.split("@@@@@")[1]
                exhi_name = user_status.split("@@@@@")[2]
                result_dict = show_recom_firm_mgdb(exhi_id, user_input)
                if result_dict['type'] == "text":
                    reply_message = [TextMessage(text=result_dict['text'], quoteToken=event.message.quote_token),
                                    TextMessage(text=f"請重新輸入關於「{exhi_name}」您感興趣的關鍵字：\n離開請輸入「取消」。")]
                else:
                    reply_message = [FlexMessage(alt_text=f"關於 {exhi_name} 的推薦店家", contents=FlexContainer.from_dict(result_dict))]
                    user_info["status"] = None


        elif user_status_code == "waiting_for_start_point":
            # 豪 + 軍，接收起點
            if user_input == "取消" or user_input == "cancel":
                user_info["status"] = None
                reply_message = [TextMessage(text="帶我趣已取消。")]
            else:
                img_url = user_status.split("@@@@@")[1]
                user_info["status"] = "waiting_for_end_point" + "@@@@@" + img_url + "@@@@@" + user_input
                reply_message = [TextMessage(text=f"起點已設定為：{user_input}\n請輸入您的目的地：")]

        elif user_status_code == "waiting_for_end_point":
            # 豪 + 軍，接收目的地
            if user_input == "取消" or user_input == "cancel":
                user_info["status"] = None
                reply_message = [TextMessage(text="帶我趣已取消。")]
            else:
                img_url = user_status.split("@@@@@")[1]
                start_point = user_status.split("@@@@@")[2]
                end_point = user_input
                user_info["status"] = None
                # 假設 draw_path_on_image 是一個處理圖片的函數
                result_image_url = draw_path_on_image(user_id, img_url, start_point, end_point)

                reply_message = [
                            TextMessage(text=f"已處理完成！起點：{start_point}，終點：{end_point}"),
                            ImageMessage(originalContentUrl=result_image_url, previewImageUrl=result_image_url)
                        ]
        
        
        elif user_status_code == "chikawa_01":
            # 已接收 user_input = 起點，接下來要傳送地圖
            if user_input == "取消":
                user_info["status"] = None
                reply_message = [TextMessage(text="帶我趣已取消！")]
            else:
                frim_id = user_status.split("@@@@@")[1]
                print("這裡是 frim_id：", frim_id)
                chikawa_dict = chikawa(frim_id)
                chikawa_map = calculate_shortest_path(chikawa_dict['map'], user_input, chikawa_dict['location'], user_id)
                reply_message = [ImageMessage(original_content_url=chikawa_map, preview_image_url=chikawa_map)]
                user_info["status"] = None # 結束導航，重置使用者狀態

    elif user_input == 'Administrator':
        # want to be Administrator
        admin_collection()
        if is_admin(user_id):
            remove_admin(user_id)
            reply_message = [TextMessage(text="您已退出管理模式。")]
        elif is_admin(user_id, "super") and not is_admin(user_id):
            add_admin(user_id)
            reply_message = [TextMessage(text="您已進入管理模式。")]
        elif not is_admin(user_id):
            user_info["status"] = "want_to_be_Admin"
            reply_message = [TextMessage(text="請輸入密碼：")]

    elif user_input == '查看 event':
        # 查看 line event
        reply_message = list()
        for item in list(event)[0:-1]:
            reply_message.append(item[0] + ":\n" + str(item[1]))
            print(item[0], item[1], sep=":\n")
        reply_message = "\n\n".join(reply_message)
        reply_message += "\n\nmessage:"
        for item in event.message:
            reply_message += f"\n\n{item[0]}: {str(item[1])}"
            print(item[0], item[1], sep=":\n")
        reply_message = [TextMessage(text=reply_message)]


    elif user_input == 'exh2':
        # 測試展覽2

        reply_message = [
            TextMessage(text="來趣看展覽！\n隨機挑選\n請選擇您感興趣的展覽："),
            FlexMessage(alt_text="隨機展覽", contents=FlexContainer.from_dict(show_exhi_mgdb()))
        ]


    elif user_input == 'show love exh':
        # 顯示我的最愛展覽

        love_exhi = user_info["love_exhi"]
        if len(love_exhi) == 0:
            reply_message = [TextMessage(text="您尚未加入任何展覽至我的最愛！")]
        else:
            reply_message = [
            TextMessage(text="這些是您感興趣的展覽："),
            FlexMessage(alt_text="我的最愛展覽", contents=FlexContainer.from_dict(show_exhi_mgdb(love_exhi)))
        ]
    

    elif user_input == 'show love exh list':
        # 顯示我的最愛展覽 - 列表

        love_exhi = user_info["love_exhi"]
        if len(love_exhi) == 0:
            reply_message = [TextMessage(text="您尚未加入任何展覽至我的最愛！")]
        else:
            reply_message = [
                FlexMessage(alt_text="我的最愛展覽清單", contents=FlexContainer.from_dict(show_exhi_mgdb(love_exhi)))
            ]


    elif user_input == 'firms':
        # 測試商家 - 隨機挑選 5 家
        
        reply_message = [FlexMessage(alt_text="隨機廠商資料", contents=FlexContainer.from_dict(show_firm_mgdb()))]


    elif user_input == 'love firms':
        # 顯示我的最愛廠商
        love_firms = user_info["love_firm"]
        
        reply_message = [FlexMessage(alt_text="我的最愛廠商資料", contents=FlexContainer.from_dict(show_firm_mgdb(love_firms)))]


    elif user_input == 'show love firm list':
        # 顯示我的最愛廠商 - 列表

        love_firm = user_info["love_firm"]
        if len(love_firm) == 0:
            reply_message = [TextMessage(text="您尚未加入任何廠商至我的最愛！")]
        else:
            reply_message = [
                FlexMessage(alt_text="我的最愛展覽清單", contents=FlexContainer.from_dict(show_list_firm_mgdb(love_firm)))
            ]


    elif user_input == 'show both':
        # 顯示我的最愛展覽、廠商

        love_exhi = user_info["love_exhi"]
        love_firm = user_info["love_firm"]

        if len(love_exhi) == 0 and len(love_firm) == 0:
            reply_message = [TextMessage(text="您尚未加入任何展覽或廠商至我的最愛！")]
        elif len(love_exhi) == 0:
            reply_message = [FlexMessage(alt_text="我的最愛廠商清單", contents=FlexContainer.from_dict(show_list_firm_mgdb(love_firm)))]
        elif len(love_firm) == 0:
            reply_message = [FlexMessage(alt_text="我的最愛展覽清單", contents=FlexContainer.from_dict(show_list_exhi_mgdb(love_exhi)))
            ]
        else:
            reply_message = [
                # FlexMessage(alt_text="我的最愛展覽清單", contents=FlexContainer.from_dict(show_list(love_exhi))),
                # FlexMessage(alt_text="我的最愛廠商清單", contents=FlexContainer.from_dict(show_list_firm(love_firm)))
                FlexMessage(alt_text="我的最愛展覽、廠商清單", contents=FlexContainer.from_dict(show_both_mgdb(love_exhi, love_firm)))
            ]

    elif user_input == 'del love firm':
        # 顯示我的最愛廠商
        love_firms = user_info["love_firm"]
        
        reply_message = [FlexMessage(alt_text="我的最愛廠商資料", contents=FlexContainer.from_dict(show_list_firm_del_mgdb(love_firms)))]

    elif user_input == 'del both':
        # 顯示我的最愛展覽、廠商

        love_exhi = user_info["love_exhi"]
        love_firm = user_info["love_firm"]

        if len(love_exhi) == 0 and len(love_firm) == 0:
            reply_message = [TextMessage(text="您尚未加入任何展覽或廠商至我的最愛！")]
        elif len(love_exhi) == 0:
            reply_message = [FlexMessage(alt_text="編輯我的最愛廠商清單", contents=FlexContainer.from_dict(show_list_firm_del_mgdb(love_firm)))]
        elif len(love_firm) == 0:
            reply_message = [FlexMessage(alt_text="編輯我的最愛展覽清單", contents=FlexContainer.from_dict(show_list_exhi_del_mgdb(love_exhi)))
            ]
        else:
            reply_message = [
                # FlexMessage(alt_text="我的最愛展覽清單", contents=FlexContainer.from_dict(show_list(love_exhi))),
                # FlexMessage(alt_text="我的最愛廠商清單", contents=FlexContainer.from_dict(show_list_firm(love_firm)))
                FlexMessage(alt_text="編輯我的最愛展覽、廠商清單", contents=FlexContainer.from_dict(show_both_del_mgdb(love_exhi, love_firm)))
            ]


    elif user_input == 'quick':
        # 進入 quick reply 選擇「喜歡展覽種類」

        fav_exhi = user_info["favorite"]
        # print("這裡是 fav_exhi：", fav_exhi)

        if len(fav_exhi) == len(exhibitions):
            quick_reply = QuickReply(items=[
                QuickReplyItem(
                    action=PostbackAction(label="重新選擇", data="remove_all_fav"))]
                )
        else:
            quick_reply = make_quick_reply_exhi(fav_exhi)

        reply_message = [TextMessage(text="來趣看展覽！\n請選擇您感興趣的展覽種類：", 
                                    quick_reply=quick_reply)]
    
    elif user_input == '重製quick':
        user_info["favorite"] = []
        quick_reply = make_quick_reply_exhi()
        reply_message = [TextMessage(text="已重製您的展覽選擇！\n請選擇您感興趣的展覽種類：", 
                                    quick_reply=quick_reply)]
        

    elif user_input.strip() == 'richmenuset':
        # 設定 Rich Menu
        if is_admin(user_id):
            rich_menu_set(configuration)
            reply_message = [TextMessage(text="Richmenu set!")]
        else:
            reply_message = [TextMessage(text="You are not Administrator!")]
        user_info["status"] = None

    elif user_input == '熊大在嗎？':
        # 測試用
        user_info["status"] = None
        reply_message = [TextMessage(text="我在，有什麼事？", 
                                    sender={"name": "熊大", 
                                            "iconUrl": "https://4.bp.blogspot.com/-x-fowV0tZAo/WohU6444rGI/AAAAAAAAFGk/_0_40Tx3-hc1yQLlwIKJ19cJDlRwcL6MgCKgBGAs/s1600/03.jpg"},
                                    quoteToken=event.message.quote_token,
                                    )]
    
    elif user_input == '熊大？':
        # 測試用
        user_info["status"] = None
        reply_message = [TextMessage(text="我在？", 
                                        sender={"name": "熊大", 
                                                "iconUrl": "https://4.bp.blogspot.com/-x-fowV0tZAo/WohU6444rGI/AAAAAAAAFGk/_0_40Tx3-hc1yQLlwIKJ19cJDlRwcL6MgCKgBGAs/s1600/03.jpg"},
                                        quoteToken=event.message.quote_token
                                        )]

    else:
        # personalized_message = f"Hello {user_id},\nyou said:\n{user_input}"
        personalized_message = "不支援此功能！"
        reply_message = [TextMessage(text=personalized_message)]
    
    return reply_message, user_info


def user_image_reply_mgdb(user_info, event, message_content):
    user_id = event.source.user_id
    # user_input = event.message.text.strip()
    # user_status = user_info["status"]

    try: 
        img = Image.open(io.BytesIO(message_content))
    except Exception as e:
        print("Error:", e)
        return [TextMessage(text="圖片讀取失敗！")], user_info, None

    try:
        # 上傳圖片到 GCS 並取得圖片 URL
        img_url = upload_image_to_gcs(img, user_id, event.timestamp/1000)
    except Exception as e:
        print("Error:", e)
        return [TextMessage(text="圖片上傳失敗！")], user_info, None

    if img_url:
        user_info["status"] = "waiting_for_start_point"  + "@@@@@" + img_url
        return [TextMessage(text="圖片已接收，請輸入起點：")], user_info, img_url
    else:
        return [TextMessage(text="圖片上傳失敗！")], user_info, None