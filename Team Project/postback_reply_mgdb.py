from linebot.v3.messaging import (
    FlexMessage,
    FlexContainer,
    TextMessage,
    QuickReply,
    QuickReplyItem,
    PostbackAction,
)

from qr_type import exhibitions, make_quick_reply_exhi, recommend_like_or_not
# JSON
# from show_exhi import show_exhi, show_list
# from show_firm import show_firm, show_list_firm, show_both

# MongoDB
from show_exhi_mgdb import show_exhi_mgdb, show_list_exhi_mgdb, show_list_exhi_del_mgdb
from show_firm_mgdb import show_firm_mgdb, show_list_firm_mgdb, show_both_mgdb, show_list_firm_del_mgdb, show_both_del_mgdb

# 推薦展覽
from show_recom import show_recom_exhi_mgdb



def postback_reply_mgdb(event, user_info: dict):
    """
    Postback 回覆
    param event: PostbackEvent
    param user_info: dict
    return reply_message: list
    """

    # 如果 data 有 @@@@@，則分開 data 和 data_data
    data = event.postback.data
    data_data = data.split("@@@@@")[1] if "@@@@@" in data else data


    if data.startswith("show_both"):
        # 顯示我的最愛展覽、廠商

        love_exhi = user_info["love_exhi"]
        love_firm = user_info["love_firm"]
        user_info["status"] = None

        if len(love_exhi) == 0 and len(love_firm) == 0:
            reply_message = [TextMessage(text="您尚未加入任何展覽或廠商至我的最愛！")]
        elif len(love_exhi) == 0:
            reply_message = [FlexMessage(alt_text="我的最愛廠商清單",
                                        contents=FlexContainer.from_dict(show_list_firm_mgdb(love_firm)))]
        elif len(love_firm) == 0:
            reply_message = [FlexMessage(alt_text="我的最愛展覽清單",
                                        contents=FlexContainer.from_dict(show_list_exhi_mgdb(love_exhi)))]
        else:
            reply_message = [
                FlexMessage(alt_text="我的最愛展覽、廠商清單", contents=FlexContainer.from_dict(show_both_mgdb(love_exhi, love_firm)))
            ]


    elif data.startswith("show_del_both"):
        # 顯示刪除我的最愛展覽、廠商

        love_exhi = user_info["love_exhi"]
        love_firm = user_info["love_firm"]
        user_info["status"] = None

        if len(love_exhi) == 0 and len(love_firm) == 0:
            reply_message = [TextMessage(text="您尚未加入任何展覽或廠商至我的最愛！")]
        elif len(love_exhi) == 0:
            reply_message = [FlexMessage(alt_text="編輯我的最愛廠商清單",
                                        contents=FlexContainer.from_dict(show_list_firm_del_mgdb(love_firm)))]
        elif len(love_firm) == 0:
            reply_message = [FlexMessage(alt_text="編輯我的最愛展覽清單",
                                        contents=FlexContainer.from_dict(show_list_exhi_del_mgdb(love_exhi)))]
        else:
            reply_message = [
                FlexMessage(alt_text="編輯我的最愛展覽、廠商清單", contents=FlexContainer.from_dict(show_both_del_mgdb(love_exhi, love_firm)))
            ]


    elif data.startswith("add_love_firm"):
        # 將廠商加入我的最愛
        love_firm = user_info["love_firm"]
        # add_time = event.timestamp # 加入時間，之後 list 變成 dict 再加入時間
        user_info["status"] = None
        if data_data not in love_firm:
            love_firm.append(data_data)
            reply_message = [TextMessage(text="已加入我的最愛！")]
        else:
            # 將已經在我的最愛裡的廠商移到 list 最後面
            love_firm.remove(data_data)
            love_firm.append(data_data)
            reply_message = [TextMessage(text="再加入一次！")]
    

    elif data.startswith("show_firm"):
        # 顯示廠商詳細資訊
        user_info["status"] = None
        print("這裡是 data：", data)
        reply_message = [FlexMessage(alt_text="廠商資訊", contents=FlexContainer.from_dict(show_firm_mgdb([data_data])))]
    

    elif data.startswith("add_love_exhib"):
        # 將展覽加入我的最愛
        love_exhi = user_info["love_exhi"]
        # add_time = event.timestamp # 加入時間，之後 list 變成 dict 再加入時間
        user_info["status"] = None
        if data_data not in love_exhi:
            love_exhi.append(data_data)
            reply_message = [TextMessage(text="已加入我的最愛！")]
        else:
            # 將已經在我的最愛裡的展覽移到 list 最後面
            love_exhi.remove(data_data)
            love_exhi.append(data_data)
            reply_message = [TextMessage(text="再加入一次！")]
    
    elif data.startswith("del_exhib"):
        # 將展覽從我的最愛移除
        # "data": "del_exhib@@@@@" + exhi['id_add'] + "@@@@@" + exhi['title']
        love_exhi = user_info["love_exhi"]
        user_info["status"] = None
        exhib_title = data.split("@@@@@")[2]
        if data_data in love_exhi:
            love_exhi.remove(data_data)
            reply_message = [TextMessage(text=f"已將「{exhib_title}」移除我的最愛！")]
        else:
            reply_message = [TextMessage(text="該展覽已不在我的最愛裡，\n請重新查看清單。")]

    elif data.startswith("del_firm"):
        # 將廠商從我的最愛移除
        # "data": "del_firm@@@@@" + firm['id_add'] + "@@@@@" + firm['name']
        love_firm = user_info["love_firm"]
        user_info["status"] = None
        firm_name = data.split("@@@@@")[2]
        if data_data in love_firm:
            love_firm.remove(data_data)
            reply_message = [TextMessage(text=f"已將「{firm_name}」移除我的最愛！")]
        else:
            reply_message = [TextMessage(text="該廠商已不在我的最愛裡，\n請重新查看清單。")]


    elif data.startswith("bring_me_to"):
        # 帶我去攤位
        user_info["status"] = "bring_me_to_01" + "@@@@@" + data_data # 廠商id
        reply_message = [TextMessage(text="帶我趣～～\n請輸入你現在附近的攤位名稱：\n離開請輸入「取消」。")]
    

    elif data.startswith("chikawa"):
        # 帶我去吉伊卡哇
        user_info["status"] = "chikawa_01" + "@@@@@" + data_data # 吉伊卡哇展覽id
        reply_message = [TextMessage(text="帶我趣～～\n請輸入你現在附近的攤位名稱：\n離開請輸入「取消」。")]


    elif data.startswith("show_exhib"):
        # 顯示展覽詳細資訊
        print("這裡是 show_exhib 的 data：", data)
        reply_message = [FlexMessage(alt_text="展覽資訊", contents=FlexContainer.from_dict(show_exhi_mgdb([data_data])))]
        user_info["status"] = None

    elif data.startswith("show_recom_shop"):
        # 廠商bubble中的店家推薦
        # data = f"show_recom_shop@@@@@{exhi_id}@@@@@{exhi_name}"
        exhi_name = data.split("@@@@@")[2]
        user_info["status"] = "show_recom_shop_01" + "@@@@@" + data_data + "@@@@@" + exhi_name # 廠商id + 展覽名稱
        reply_message = [TextMessage(text=f"請輸入關於「{exhi_name}」您感興趣的關鍵字：\n離開請輸入「取消」。")] 



    elif data == 'recommend':
        # 中下，推薦展覽
        # reply_text = "這裡是推薦展覽的功能！"
        user_info["status"] = None
        if user_info["love_exhi"] == []:
            exhib_id = ""
        else:
            exhib_id = user_info["love_exhi"][-1]
        
        reply_message = [
            FlexMessage(alt_text="猜你想趣", 
                        contents=FlexContainer.from_dict(show_recom_exhi_mgdb(exhib_id)),
                        quick_reply=recommend_like_or_not(),)
                        ]


    elif data == 'search':
        # 左下，搜尋展覽
        reply_text = "請輸入您想搜尋展覽的一個關鍵字："
        user_info["status"] = "search_exhibition_by_keyword"
        reply_message = [TextMessage(text=reply_text)]


    elif data == 'quick':
        # 進入 quick reply 選擇「喜歡展覽種類」

        print("進入 quick")
        user_info["status"] = None
        fav_exhi = user_info["favorite"]

        if len(fav_exhi) == len(exhibitions):
            quick_reply = QuickReply(items=[
                QuickReplyItem(
                    action=PostbackAction(label="重新選擇", data="remove_all_fav"))]
                )
        else:
            quick_reply = make_quick_reply_exhi(fav_exhi)

        reply_message = [TextMessage(text="來趣看展覽！\n請選擇您感興趣的展覽種類：", 
                                    quick_reply=quick_reply)]
    
    elif data == "remove_all_fav":
        # 展覽種類重置
        user_info["favorite"] = []
        reply_text = "已重製您的展覽選擇！\n請選擇您感興趣的展覽種類："
        reply_message = [TextMessage(text=reply_text, 
                                    quick_reply=make_quick_reply_exhi())]
        user_info["status"] = None
        
    elif data == "like" or data == "dislike":
        # 推薦展覽是否喜歡
        reply_message = [TextMessage(text="謝謝您的回饋！")]
        user_info["status"] = None


    elif data in exhibitions:
        user_info["status"] = None
        # 選擇展覽種類

        reply_text = f"你喜歡{exhibitions[data]}！"
        print("這裡是 reply_text：", reply_text)
        if data not in user_info["favorite"]:
            user_info["favorite"].append(data)
        
        if len(user_info["favorite"]) == len(exhibitions):
            reply_text += "\n你已經選擇了所有展覽！"
            reply_message = [
                FlexMessage(alt_text="展覽資訊", contents=FlexContainer.from_dict(show_exhi_mgdb(type=data))),
                TextMessage(text=reply_text)
                ]
        else:
            reply_message = [
                FlexMessage(alt_text="展覽資訊", contents=FlexContainer.from_dict(show_exhi_mgdb(type=data))),
                TextMessage(text=reply_text,
                            # quick_reply=make_quick_reply_exhi(user_info["favorite"])
                            )
                            ]
    
    
    else:
        user_info["status"] = None
        reply_text = "這個 postback 目前不存在！"
        reply_text += f"\n以下是 data 部分：\n{data}"
        print("這裡是 reply_text：", reply_text)
        reply_message = [TextMessage(text=reply_text)]
    
    return reply_message, user_info
