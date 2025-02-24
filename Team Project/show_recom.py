import json
import random
import urllib.parse
from pymongo import MongoClient
from bson import ObjectId
import time

from langchain_exhitbition_mf_V4_250109 import user_interaction, initialize_environment_exh
# from langchain_exhitbition_mf_V5_250110_ui import backend_initialize, user_interaction
from langchain_store_mf_V4_250108 import user_interaction_store, initialize_environment_store

# 設定 MongoDB 連接
# client = MongoClient("mongodb+srv://everybody:gac244@cluster0.mh9yw.mongodb.net/") # 鎮均
client = MongoClient("mongodb+srv://jenhao:abcd1234@cluster0.0vluw.mongodb.net/") # 人豪
exhi_db = client['exhibitionDB'] # db 只有在有資料寫入時才會建立
firm_db = client['companyDB'] # db 只有在有資料寫入時才會建立

# 初始化
print("=====================================")
print("這裡是 show_recom.py")
print("=====================================")
# print("正在初始化展覽：backend_initialize()")
# start = time.time()
# db = backend_initialize() #初始化執行時間約為 8 秒
# end1 = time.time()
# print("=====================================")
# print("正在初始化廠商：backend_initialize_store()")
# start2 = time.time()
# vector_stores, processed_documents_dict = backend_initialize_store()
# end2 = time.time()
# print("=====================================")
# print(f"初始化展覽時間：{end1 - start : .2f} 秒")
# print(f"初始化廠商時間：{end2 - start2 : .2f} 秒")
# print(f"總共初始化時間：{end2 - start : .2f} 秒")
# print("=====================================")

from langchain_exhitbition_mf_V4_250109 import initialize_environment_exh
db = initialize_environment_exh()


def bubb_temp(img_url, exhi_url, exhi_name, exhi_loca, exhi_date, exhi_id, highlight):
    # 對中文 exhi_loca 進行 URL 編碼
    encoded_exhi_loca = urllib.parse.quote(exhi_loca.replace(' ', '').replace('、', ''))

    bubble = {
        "type": "bubble",
        "hero": {
            "type": "image",
            "url": img_url,
            "size": "full",
            "aspectRatio": "20:13",
            "aspectMode": "cover",
            "action": {
                "type": "uri",
                "uri": exhi_url.replace(' ', '')
            }
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "text",
                    "text": exhi_name,
                    "weight": "bold",
                    "size": "xl"
                },
                {
                    "type": "box",
                    "layout": "vertical",
                    "margin": "lg",
                    "spacing": "sm",
                    "contents": [
                        {
                            "type": "box",
                            "layout": "baseline",
                            "spacing": "sm",
                            "contents": [
                                {
                                    "type": "icon",
                                    "url": "https://png.pngtree.com/element_our/20190530/ourmid/pngtree-cartoon-navigation-icon-download-image_1251411.jpg",
                                    "position": "relative"
                                },
                                {
                                    "type": "text",
                                    "text": exhi_loca,
                                    "wrap": True,
                                    "color": "#666666",
                                    "size": "sm",
                                    "flex": 5
                                }
                            ]
                        },
                        {
                            "type": "box",
                            "layout": "baseline",
                            "spacing": "sm",
                            "contents": [
                                {
                                    "type": "icon",
                                    "url": "https://png.pngtree.com/element_our/20190531/ourmid/pngtree-clock-time-icon-image_1287819.jpg",
                                    "size": "lg"
                                },
                                {
                                    "type": "text",
                                    "text": exhi_date,
                                    "wrap": True,
                                    "color": "#666666",
                                    "size": "sm",
                                    "flex": 5
                                }
                            ]
                        },
                        {
                            "type": "text",
                            "text": highlight,
                            "wrap": True,
                            "size": "md",
                            "margin": "md",
                            # "color": "#",
                            "flex": 0
                        }
                    ]
                }
            ]
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "contents": [
                {
                    "type": "button",
                    "style": "secondary", # 顯示為次要按鈕
                    # "style": "link",
                    "height": "sm",
                    "action": {
                        "type": "uri",
                        "label": "官方網站",
                        "uri": exhi_url.replace(' ', '')
                    }
                },
                {
                    "type": "button",
                    "style": "secondary", # 顯示為次要按鈕
                    # "style": "link",
                    "height": "sm",
                    "action": {
                        "type": "uri",
                        "label": "展覽位置",
                        "uri": f"https://www.google.com.tw/maps/search/?api=1&query={encoded_exhi_loca}"
                    }
                },
                {
                    "type": "button",
                    "style": "secondary", # 顯示為次要按鈕
                    "height": "sm",
                    "action": {
                        "type": "postback",
                        "label": "店家推薦",
                        "data": f"show_recom_shop@@@@@{exhi_id}@@@@@{exhi_name}"
                    }
                },
                {
                    "type": "button",
                    "style": "primary", # 顯示為主要按鈕
                    "action": {
                        "type": "postback",
                        "label": "加到我的最愛",
                        "data": f"add_love_exhib@@@@@{exhi_id}",
                        "displayText": f"我喜歡「{exhi_name}」！"
                    }
                }
            ],
            "flex": 0
        }
    }
    # return json.dumps(bubble) # 轉換成 json 格式
    # label: 按鈕上的文字
    # "displayText": 發送到聊天視窗上的文字
    return bubble


def show_recom_exhi_mgdb(latest_love_exhib_id: str) -> dict:
    '''
    產生一個 carousel ，格式為 dict，裡面包含展覽的資訊
    params:
        latest_love_exhib_id: str, 使用者最新收藏的展覽 id
    return:
        dict, carousel
    '''
    
    langchain_result_dict = user_interaction(db, latest_love_exhib_id)
    # langchain_result_dict = """{'2025 臺灣國際食品暨設備展': {'_id': '677ef9ec3e99f7ca1629826f', 'highlight': '網羅各類型食品、食材及網路人氣美食，提供豐富的國際食品交流平台。'}, 
    #                         '2025台灣國際飯店暨餐飲設備用品展': {'_id': '677c9e86cd9d6e9adaf09578', 'highlight': '以「精準食代・綠色智慧」為主題，展示最新的餐飲及飯店設備技術。'},
    #                         '2025 臺中國際茶、咖啡暨烘焙展': {'_id': '677ca545cd9d6e9adaf09706', 'highlight': '涵蓋品茗好茶、飲盡黑金及玩美烘焙，為中部地區最大的餐飲採購平台。'}}"""
    
    love_exhi = []
    highlight = []
    for _id, exhib in langchain_result_dict.items():
        if exhib.get('highlight', None) is None:
            break
        love_exhi.append(_id)
        highlight.append(exhib['highlight'])
    
    exhi_info_dict = {}
    if love_exhi == []:
        # 如果 love_exhi 是空的，則從 exhi_db 中隨機挑選 5 個展覽
        collections_list = random.sample(exhi_db.list_collection_names(), 5)
        for collection in collections_list:
            count = exhi_db[collection].count_documents({})
            if count != 0:
                random_index = random.randint(0, count - 1)
                exhib = exhi_db[collection].find().skip(random_index).limit(1).next()
                exhi_info_dict[str(exhib['_id'])] = exhib
    else:
        # 如果 love_exhi 不是空的，則從 exhi_db 中挑選出使用者收藏的展覽
        for exhi_id in love_exhi:
            for collection in exhi_db.list_collection_names():
                print(f"這裡 exhi_id: {exhi_id}, collection: {collection}")
                exhib = exhi_db[collection].find_one({"_id": ObjectId(exhi_id)})
                if exhib:
                    exhi_info_dict[exhi_id] = exhib
                    exhi_info_dict[exhi_id]['highlight'] = highlight[love_exhi.index(exhi_id)]
                    break
    
    # contents 為 bubble 的 list，每個 bubble 代表一個展覽
    contents = []
    for _id, exhi in exhi_info_dict.items():

        print(f"這裡是 logo: {exhi['logo']}")
        print(f"這裡是 url: {exhi['url']}")
        print(f"這裡是 title: {exhi['title']}")
        print(f"這裡是 location: {exhi['location']}")
        print(f"這裡是 date: {exhi['date']}")
        print(f"這裡是 highlight: {exhi['highlight']}")
        print(f"這裡是 _id: {_id}")

        # 檢查 name 是否是 None
        if exhi.get('title', None) is None:
            exhi['title'] = '展覽名稱待更新'
        
        # 檢查 date 是否是 None
        if exhi.get('date', None) is None:
            exhi['date'] = '展覽時間待更新'

        # 檢查 location 是否是 None
        if exhi.get('location', None) is None:
            exhi['location'] = '展覽地點待更新'

        # 檢查 logo 是否是 None
        if (logo := exhi.get('logo', None)):
            if logo[-4:] not in ['.jpg', '.png', '.gif', 'jpeg', 'webp', 'tiff', '.bmp', '.svg'] or logo[:5] != 'https':
                exhi['logo'] = 'https://developers-resource.landpress.line.me/fx/img/01_1_cafe.png'
        elif not logo.startswith('https:'):
            exhi['logo'] = 'https://developers-resource.landpress.line.me/fx/img/01_1_cafe.png'
        else:
            exhi['logo'] = 'https://developers-resource.landpress.line.me/fx/img/01_1_cafe.png'
        
        # 檢查 url 是否是 None
        if (url := exhi.get('url', None)) is None or url[:5] == 'http:':
            encoded_title = urllib.parse.quote(exhi['title'])
            # exhi['url'] = 'https://www.google.com.tw/search?q=' + exhi['title'].replace(' ', '').replace('、', '')
            exhi['url'] = f"https://www.google.com.tw/search?q={encoded_title}"
        
        # 檢查 highlight 是否是 None
        if (highlight := exhi.get('highlight', None)) is None:
            exhi['highlight'] = '展覽亮點待更新'

        # todo: 檢查 check 是否是 True

        print("------"*6)
        print(f"這裡是 logo: {exhi['logo']}")
        print(f"這裡是 url: {exhi['url']}")
        print(f"這裡是 title: {exhi['title']}")
        print(f"這裡是 location: {exhi['location']}")
        print(f"這裡是 date: {exhi['date']}")
        print(f"這裡是 highlight: {exhi['highlight']}")
        print(f"這裡是 _id: {_id}")
        print("====="*6)

        contents.append(bubb_temp(exhi['logo'], exhi['url'], exhi['title'], 
                                    exhi['location'], exhi['date'], _id, exhi['highlight']))

    # 回傳 carousel 的 dict
    return {"type": "carousel", "contents":contents}




def bubb_temp_firm(img_url, exhi_url, exhi_name, exhi_loca, exhi_date, exhi_id, highlight):
    # 對中文 exhi_loca 進行 URL 編碼
    encoded_exhi_loca = urllib.parse.quote(exhi_loca.replace(' ', '').replace('、', ''))

    bubble = {
        "type": "bubble",
        "hero": {
            "type": "image",
            "url": img_url,
            "size": "full",
            "aspectRatio": "20:13",
            "aspectMode": "cover",
            "action": {
                "type": "uri",
                "uri": exhi_url.replace(' ', '')
            }
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "text",
                    "text": exhi_name,
                    "weight": "bold",
                    "size": "xl"
                },
                {
                    "type": "box",
                    "layout": "vertical",
                    "margin": "lg",
                    "spacing": "sm",
                    "contents": [
                        {
                            "type": "box",
                            "layout": "baseline",
                            "spacing": "sm",
                            "contents": [
                                {
                                    "type": "icon",
                                    "url": "https://png.pngtree.com/element_our/20190530/ourmid/pngtree-cartoon-navigation-icon-download-image_1251411.jpg",
                                    "position": "relative"
                                },
                                {
                                    "type": "text",
                                    "text": exhi_loca,
                                    "wrap": True,
                                    "color": "#666666",
                                    "size": "sm",
                                    "flex": 5,
                                    "action": {
                                        "type": "uri",
                                        "label": "展覽位置",
                                        "uri": f"https://www.google.com.tw/maps/search/?api=1&query={encoded_exhi_loca}"
                                    }
                                }
                            ]
                        },
                        {
                            "type": "box",
                            "layout": "baseline",
                            "spacing": "sm",
                            "contents": [
                                {
                                    "type": "icon",
                                    "url": "https://png.pngtree.com/element_our/20190531/ourmid/pngtree-clock-time-icon-image_1287819.jpg",
                                    "size": "lg"
                                },
                                {
                                    "type": "text",
                                    "text": exhi_date,
                                    "wrap": True,
                                    "color": "#666666",
                                    "size": "sm",
                                    "flex": 5
                                }
                            ]
                        },
                        {
                            "type": "text",
                            "text": highlight,
                            "wrap": True,
                            "size": "md",
                            "margin": "md",
                            # "color": "#",
                            "flex": 0
                        }
                    ]
                }
            ]
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "contents": [
                {
                    "type": "button",
                    "style": "secondary", # 顯示為次要按鈕
                    # "style": "link",
                    "height": "sm",
                    "action": {
                        "type": "uri",
                        "label": "官方網站",
                        "uri": exhi_url.replace(' ', '')
                    }
                },
                {
                    "type": "button",
                    "style": "secondary", # 顯示為次要按鈕
                    # "style": "link",
                    "height": "sm",
                    "action": {
                        "type": "uri",
                        "label": "展覽位置",
                        "uri": f"https://www.google.com.tw/maps/search/?api=1&query={encoded_exhi_loca}"
                    }
                },
                {
                    "type": "button",
                    "style": "primary", # 顯示為主要按鈕
                    "action": {
                        "type": "postback",
                        "label": "加到我的最愛",
                        "data": f"add_love_firm@@@@@{exhi_id}",
                        "displayText": f"我喜歡「{exhi_name}」！"
                    }
                }
            ],
            "flex": 0
        }
    }
    # return json.dumps(bubble) # 轉換成 json 格式
    # label: 按鈕上的文字
    # "displayText": 發送到聊天視窗上的文字
    return bubble


def show_recom_firm_mgdb(exhi_id, user_input) -> dict:
    '''
    產生一個 carousel ，格式為 dict，裡面包含展覽的資訊
    params:
        latest_love_exhib_id: str, 使用者最新收藏的展覽 id
    return:
        dict, carousel
    '''
    vector_stores, processed_documents_dict = initialize_environment_store(exhi_id)
    langchain_result_dict = user_interaction_store(exhi_id, user_input, 
                                                    vector_stores = vector_stores,
                                                    processed_documents_dict = processed_documents_dict)
    '''{ObjectId('677c94b6cd8df06c18b4f096'): {'name': '聖比德蛋糕有限公司', 'highlight': '提供多種口味的精緻蛋糕，位於名店街的知名甜點店。'},
        ObjectId('677c94b6cd8df06c18b4f0a1'): {'name': '易德食食品有限公司', 'highlight': '以多樣化的糕點選擇聞名，名店街中深受顧客喜愛。'}, 
        ObjectId('677c94b6cd8df06c18b4f06c'): {'name': '一之軒食品有限公司', 'highlight': '以高品質的糕點產品著稱，名店街的糕點愛好者必訪之地。'}, 
        ObjectId('677c94b6cd8df06c18b4f09e'): {'name': '超品企業股份有限公司', 'highlight': '提供創新口味的糕點選擇，名店街上的甜點創新者。'}}
    '''

    if "result" in langchain_result_dict:
        return {"type": "text", "text": langchain_result_dict["result"]}
    
    love_firm_objid = []
    highlight = []
    for obj_id, exhib in langchain_result_dict.items():
        if exhib.get('highlight', None) is None:
            break
        love_firm_objid.append(obj_id)
        highlight.append(exhib['highlight'])
    
    firm_info_dict = {}
    if love_firm_objid == []:
        return {"type": "text", "text": langchain_result_dict["result"]}
    else:
        # 如果 love_exhi 不是空的，則從 exhi_db 中挑選出使用者收藏的展覽
        for obj_id in love_firm_objid:
            # for collection in firm_db.list_collection_names():
                firm = firm_db[exhi_id].find_one({"_id": obj_id})
                if firm:
                    firm_info_dict[obj_id] = firm
                    firm_info_dict[obj_id]['highlight'] = highlight[love_firm_objid.index(obj_id)]
                    

    # 搜尋展覽的資訊
    for collection in exhi_db.list_collection_names():
        exhib = exhi_db[collection].find_one({"_id": ObjectId(exhi_id)})
        if exhib:
            exhi_docu = exhib
            break
    
    # contents 為 bubble 的 list，每個 bubble 代表一個展覽
    contents = []
    for obj_id, firm in firm_info_dict.items():

        print(f"這裡是 logo: {firm['logo']}")
        print(f"這裡是 url: {firm['url']}")
        print(f"這裡是 name: {firm['name']}")
        print(f"這裡是 location: {exhi_docu['location']}")
        print(f"這裡是 date: {exhi_docu['date']}")
        print(f"這裡是 highlight: {firm['highlight']}")
        print(f"這裡是 obj_id: {obj_id}")

        # 檢查 name 是否是 None
        if firm.get('name', None) is None:
            firm['name'] = '店家名稱待更新'
        
        # 檢查 date 是否是 None
        if exhi_docu.get('date', None) is None:
            exhi_docu['date'] = '展覽時間待更新'

        # 檢查 location 是否是 None
        if exhi_docu.get('location', None) is None:
            exhi_docu['location'] = '展覽地點待更新'

        # 檢查 logo 是否是 None
        if (logo := firm.get('logo', None)):
            if logo[-4:] not in ['.jpg', '.png', '.gif', 'jpeg', 'webp', 'tiff', '.bmp', '.svg'] or logo[:5] != 'https':
                firm['logo'] = 'https://developers-resource.landpress.line.me/fx/img/01_1_cafe.png'
        elif not logo.startswith('https:'):
            firm['logo'] = 'https://developers-resource.landpress.line.me/fx/img/01_1_cafe.png'
        else:
            firm['logo'] = 'https://developers-resource.landpress.line.me/fx/img/01_1_cafe.png'
        
        # 檢查 url 是否是 None
        if (url := firm.get('url', None)) is None or url[:5] == 'http:' or url == "":
            encoded_title = urllib.parse.quote(firm['name'])
            firm['url'] = f"https://www.google.com.tw/search?q={encoded_title}"
        
        # 檢查 highlight 是否是 None
        if (highlight := firm.get('highlight', None)) is None:
            firm['highlight'] = '展覽亮點待更新'

        # todo: 檢查 check 是否是 True

        print("------"*6)
        print(f"這裡是 logo: {firm['logo']}")
        print(f"這裡是 url: {firm['url']}")
        print(f"這裡是 name: {firm['name']}")
        print(f"這裡是 location: {exhi_docu['location']}")
        print(f"這裡是 date: {exhi_docu['date']}")
        print(f"這裡是 highlight: {firm['highlight']}")
        print(f"這裡是 obj_id: {obj_id}")
        print("====="*6)

        contents.append(bubb_temp_firm(firm['logo'], firm['url'], firm['name'], 
                                    exhi_docu['location'], exhi_docu['date'], str(obj_id), firm['highlight']))

    # 回傳 carousel 的 dict
    return {"type": "carousel", "contents":contents}