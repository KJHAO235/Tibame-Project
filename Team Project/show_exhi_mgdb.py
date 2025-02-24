import json
import random
import urllib.parse
from pymongo import MongoClient
from bson import ObjectId

# 設定 MongoDB 連接
client = MongoClient("mongodb+srv://everybody:gac244@cluster0.mh9yw.mongodb.net/")
exhi_db = client['exhibitionDB'] # db 只有在有資料寫入時才會建立

# bubble 模板
def bubb_temp(img_url, exhi_url, exhi_name, exhi_loca, exhi_date, exhi_id):
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
                    "size": "xl",
                    "wrap": True,
                    "action": {
                        "type": "uri",
                        "label": "action",
                        "uri": exhi_url.replace(' ', '')
                    }
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


def show_exhi_mgdb(love_exhi = list(), type = str()) -> dict:
    '''
    產生一個 carousel ，格式為 dict，裡面包含最多 12 個展覽的資訊
    params:
        love_exhi: list, 包含使用者收藏的展覽的 ObjectId
    return:
        dict, carousel
    '''
    
    exhi_info_dict = {}
    if type != "":
        # 如果 type 不是空的，則從 exhi_db 中挑選出 type 類型的展覽
        count = exhi_db[type].count_documents({})
        if count != 0:
            random_exhibs = exhi_db[type].aggregate([{"$sample": {"size": min(5, count)}}])
            for exhib in random_exhibs:
                exhi_info_dict[str(exhib['_id'])] = exhib

    elif love_exhi == []:
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
            if isinstance(exhi_id, str):
                for collection in exhi_db.list_collection_names():
                    exhib = exhi_db[collection].find_one({"_id": ObjectId(exhi_id)})
                    if exhib:
                        exhi_info_dict[exhi_id] = exhib
                        break
            else:
                for collection in exhi_db.list_collection_names():
                    exhib = exhi_db[collection].find_one({"_id": exhi_id})
                    if exhib:
                        exhi_info_dict[exhi_id] = exhib
                        break
    
    # contents 為 bubble 的 list，每個 bubble 代表一個展覽
    contents = []
    for _id, exhi in exhi_info_dict.items():
        print("logo:", exhi['logo']) # logo
        print("網址:", exhi['url'])
        print("廠商:", exhi['title']) # 展覽名稱
        print("展覽:", exhi['location']) # 展覽地點
        print("exhi:", _id)

        print("-----------------")
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
        
        # todo: 檢查 check 是否是 True

        print("logo:", exhi['logo']) # logo
        print("網址:", exhi['url'])
        print("廠商:", exhi['title']) # 展覽名稱
        print("展覽:", exhi['location']) # 展覽地點
        print("exhi:", _id)
        print("=========")

        contents.append(bubb_temp(exhi['logo'], exhi['url'], exhi['title'], 
                                    exhi['location'], exhi['date'], _id))

    # 回傳 carousel 的 dict
    return {"type": "carousel", "contents":contents}


def list_bubble(exhi_list: list) -> dict:
    '''
    產生一個 list_bubble，格式為 dict，會是一個顯示我的最愛展覽清單的 bubble
    params:
        exhi_list: list, 包含展覽的資訊，[{}, {}, ...]，每個 dict 包含 title 與 id_add
    return:
        dict, list_bubble
    '''
    contents = list()
    # colors = ['#FF0000', '#FFA500', '#FFFF00', '#008000', '#0000FF', '#4B0082', '#EE82EE']
    for exhi in exhi_list:
        temp_dict = {
                      "type": "text",
                      "text": exhi['title'],
                      # "align": "center",
                      "margin": "lg",
                      "size": "lg",
                      "color": "#0066cc",
                      # "color": colors[exhi_list.index(exhi) % len(colors)],
                      "action": {
                        "type": "postback",
                        "label": "action",
                        "data": "show_exhib@@@@@" + exhi['id_add']
                      }
                    }
        contents.append(temp_dict)
    bubble = {
      "type": "bubble",
      "body": {
        "type": "box",
        "layout": "vertical",
        "contents": [
          {
            "type": "text",
            "text": "LOVE EXHIB",
            "weight": "bold",
            "color": "#1DB446",
            "size": "md"
          },
          {
            "type": "text",
            "text": "我的最愛展覽清單",
            "weight": "bold",
            "size": "xxl",
            "margin": "xs"
          },
          {
            "type": "text",
            "text": "可點擊展覽名稱查看詳細資訊",
            "size": "sm",
            "color": "#aaaaaa",
            "wrap": True
          },
          {
            "type": "separator",
            "margin": "lg"
          },
          {
            "type": "box",
            "layout": "vertical",
            # "margin": "xxl",
            "spacing": "sm",
            "contents": contents
          },
          {
            "type": "separator",
            "margin": "xxl"
          },
        ]
      },
      "styles": {
        "footer": {
          "separator": True
        }
      }
    }
    return bubble


def show_list_exhi_mgdb(love_exhi: list) -> dict:
    '''
    為了 show 出 list_bubble，為了讓使用者查看自己收藏的展覽
    拿到 展覽 id ，並從 kuei_dict 中找出對應的展覽名稱
    再將展覽名稱與 展覽 id 放入 list_bubble 中，
    並回傳 dict 格式的 bubble
    params:
        love_exhi: list, 使用者收藏的展覽 id
    return:
        dict, list_bubble
    '''
    exhi_info_dict = {}
    # 如果 love_exhi 不是空的，則從 exhi_db 中挑選出使用者收藏的展覽
    for exhi_id in love_exhi:
        for collection in exhi_db.list_collection_names():
            exhib = exhi_db[collection].find_one({"_id": ObjectId(exhi_id)})
            if exhib:
                exhi_info_dict[exhi_id] = exhib
                break
    
    
    for _id, exhi_info in exhi_info_dict.items():
        # 檢查 name 是否是 None
        if exhi_info.get('title', None) is None:
            exhi_info['title'] = '展覽名稱待更新'
        
        exhi_info_dict[_id] = {'title': exhi_info['title'], 'id_add': _id}
    return list_bubble(list(exhi_info_dict.values()))


def list_exhi_del_bubble(exhi_list: list) -> dict:
    '''
    產生一個 list_bubble，格式為 dict，會是一個顯示我的最愛展覽清單的 bubble
    params:
        exhi_list: list, 包含展覽的資訊，[{}, {}, ...]，每個 dict 包含 title 與 id_add
    return:
        dict, list_bubble
    '''
    contents = list()
    # colors = ['#FF0000', '#FFA500', '#FFFF00', '#008000', '#0000FF', '#4B0082', '#EE82EE']
    for exhi in exhi_list:
        temp_dict = {
            "type": "box",
            "layout": "baseline",
            "margin": "lg",
            "contents": [
            {
                "type": "text",
                "text": exhi['title'],
                "size": "lg",   
                "color": "#0066cc",
                # "color": colors[firm_list.index(firm) % len(colors)],
                "action": {
                    "type": "postback",
                    "label": "action",
                    "data": "show_exhib@@@@@" + exhi['id_add']
                }
            },
            {
                "type": "text",
                "text": "刪",
                "size": "lg",
                "align": "end",
                "flex": 0,
                "color": "#ad3128",
                "action": {
                    "type": "postback",
                    "label": "action",
                    "data": "del_exhib@@@@@" + exhi['id_add'] + "@@@@@" + exhi['title']
            }
            }
            ]
        }
        contents.append(temp_dict)
    bubble = {
      "type": "bubble",
      "body": {
        "type": "box",
        "layout": "vertical",
        "contents": [
          {
            "type": "text",
            "text": "LOVE EXHIB",
            "weight": "bold",
            "color": "#1DB446",
            "size": "md"
          },
          {
            "type": "text",
            "text": "我的最愛展覽清單",
            "weight": "bold",
            "size": "xxl",
            "margin": "xs"
          },
          {
            "type": "text",
            "text": "可點擊展覽名稱查看詳細資訊",
            "size": "sm",
            "color": "#aaaaaa",
            "wrap": True
          },
          {
            "type": "separator",
            "margin": "lg"
          },
          {
            "type": "box",
            "layout": "vertical",
            # "margin": "xxl",
            "spacing": "sm",
            "contents": contents
          },
          {
            "type": "separator",
            "margin": "xxl"
          },
        ]
      },
      "styles": {
        "footer": {
          "separator": True
        }
      }
    }
    return bubble


def show_list_exhi_del_mgdb(love_exhi: list) -> dict:
    '''
    為了 show 出 list_bubble，為了讓使用者查看自己收藏的展覽
    拿到 展覽 id ，並從 kuei_dict 中找出對應的展覽名稱
    再將展覽名稱與 展覽 id 放入 list_bubble 中，
    並回傳 dict 格式的 bubble
    params:
        love_exhi: list, 使用者收藏的展覽 id
    return:
        dict, list_bubble
    '''
    exhi_info_dict = {}
    # 如果 love_exhi 不是空的，則從 exhi_db 中挑選出使用者收藏的展覽
    for exhi_id in love_exhi:
        for collection in exhi_db.list_collection_names():
            exhib = exhi_db[collection].find_one({"_id": ObjectId(exhi_id)})
            if exhib:
                exhi_info_dict[exhi_id] = exhib
                break
    
    
    for _id, exhi_info in exhi_info_dict.items():
        # 檢查 name 是否是 None
        if exhi_info.get('title', None) is None:
            exhi_info['title'] = '展覽名稱待更新'
        
        exhi_info_dict[_id] = {'title': exhi_info['title'], 'id_add': _id}
    return list_exhi_del_bubble(list(exhi_info_dict.values()))



def bubb_temp_chikawa(img_url, exhi_url, exhi_name, exhi_loca, exhi_date, exhi_id):
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
                        "type": "postback",
                        "label": "帶我趣",
                        "data": f"chikawa@@@@@{exhi_id}"
                    }
                },
                {
                    "type": "button",
                    "style": "secondary", # 顯示為次要按鈕
                    "height": "sm",
                    "action": {
                        "type": "postback",
                        "label": "店家推薦",
                        "data": f"show_shop@@@@@{exhi_id}"
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


def show_exhi_mgdb_chikawa(love_exhi: list = list(), type: str = "") -> dict:
    '''
    產生一個 carousel ，格式為 dict，裡面包含最多 12 個展覽的資訊
    params:
        love_exhi: list, 包含使用者收藏的展覽的 ObjectId
    return:
        dict, carousel
    '''
    
    exhi_info_dict = {}
    if type != "":
        # 如果 type 不是空的，則從 exhi_db 中挑選出 type 類型的展覽
        count = exhi_db[type].count_documents({})
        if count != 0:
            random_exhibs = exhi_db[type].aggregate([{"$sample": {"size": min(5, count)}}])
            for exhib in random_exhibs:
                exhi_info_dict[str(exhib['_id'])] = exhib

    elif love_exhi == []:
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
                    break
    
    # contents 為 bubble 的 list，每個 bubble 代表一個展覽
    contents = []
    for _id, exhi in exhi_info_dict.items():

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
            if logo[-4:] not in ['.jpg', '.png', '.gif', 'jpeg', 'webp', 'tiff', '.bmp', '.svg'] or logo[:5] == 'http:':
                exhi['logo'] = 'https://developers-resource.landpress.line.me/fx/img/01_1_cafe.png'
        elif not logo.startswith('https:'):
            exhi['logo'] = 'https://developers-resource.landpress.line.me/fx/img/01_1_cafe.png'
        # else:
        #     exhi['logo'] = 'https://developers-resource.landpress.line.me/fx/img/01_1_cafe.png'
        
        # 檢查 url 是否是 None
        if (url := exhi.get('url', None)) is None or url[:5] == 'http:':
            encoded_title = urllib.parse.quote(exhi['title'])
            # exhi['url'] = 'https://www.google.com.tw/search?q=' + exhi['title'].replace(' ', '').replace('、', '')
            exhi['url'] = f"https://www.google.com.tw/search?q={encoded_title}"
        
        # todo: 檢查 check 是否是 True

        print("=========")
        print("logo:", exhi['logo']) # logo
        print("網址:", exhi['url'])
        print("廠商:", exhi['title']) # 展覽名稱
        print("展覽:", exhi['location']) # 展覽地點
        print("exhi:", _id)

        contents.append(bubb_temp_chikawa(exhi['logo'], exhi['url'], exhi['title'], 
                                    exhi['location'], exhi['date'], _id))

    # 回傳 carousel 的 dict
    return {"type": "carousel", "contents":contents}


def chikawa(exhib_id):
    '''
    顯示吉伊卡哇的展覽資訊
    '''
    exhi_db[exhib_id]
    exhib = exhi_db['Other'].find_one({"_id": ObjectId(exhib_id)})

    # 檢查 location 是否是 None
    if exhib.get('location', None) is None:
        exhib['location'] = '展覽地點待更新'

    return {"location": exhib['location'], "map": exhib['map'][1]}





if __name__ == '__main__':
    import requests
    import time

    # print(show_love_exhi())
    # print(type(show_love_exhi()))

    # 讀取展覽資料
    try:
        with open('static/checked/kuei_20241229_0045.json', 'r', encoding='utf-8') as f:
            kuei_dict = json.load(f)
    except Exception as e:
        kuei_dict = {"message": "Error reading kuei.json"}
        print(f"Error reading kuei.json: {e}")