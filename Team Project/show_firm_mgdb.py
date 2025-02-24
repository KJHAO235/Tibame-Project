import os
import json
import random
import urllib.parse
from pymongo import MongoClient
from bson import ObjectId
from show_exhi_mgdb import show_list_exhi_mgdb, show_list_exhi_del_mgdb

# 設定 MongoDB 連接
client = MongoClient("mongodb+srv://everybody:gac244@cluster0.mh9yw.mongodb.net/")
firm_db = client['companyDB']


def bubb_temp(img_url, firm_url, firm_name, firm_loca, firm_id) -> dict:

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
                "uri": firm_url
            }
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "text",
                    "text": firm_name,
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
                                    "text": firm_loca,
                                    "wrap": True,
                                    "color": "#666666",
                                    "size": "sm",
                                    "flex": 5
                                }
                            ]
                        },
                        # {
                        #     "type": "box",
                        #     "layout": "baseline",
                        #     "spacing": "sm",
                        #     "contents": [
                        #         {
                        #             "type": "icon",
                        #             "url": "https://png.pngtree.com/element_our/20190531/ourmid/pngtree-clock-time-icon-image_1287819.jpg",
                        #             "size": "lg"
                        #         },
                        #         {
                        #             "type": "text",
                        #             "text": firm_date,
                        #             "wrap": True,
                        #             "color": "#666666",
                        #             "size": "sm",
                        #             "flex": 5
                        #         }
                        #     ]
                        # }
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
                        "uri": firm_url
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
                        "data": f"bring_me_to@@@@@{firm_id}"
                    }
                },
                {
                    "type": "button",
                    "style": "primary", # 顯示為主要按鈕
                    "action": {
                        "type": "postback",
                        "label": "加到我的最愛",
                        "data": f"add_love_firm@@@@@{firm_id}",
                        "displayText": f"我喜歡「{firm_name}」！"
                    }
                }
            ],
            "flex": 0
        }
    }
    # label: 按鈕上的文字
    # "displayText": 發送到聊天視窗上的文字
    # return json.dumps(bubble) # 轉換成 json 格式
    return bubble


def show_firm_mgdb(love_firm: list = list()) -> dict:
    '''
    產生一個 carousel ，格式為 dict，裡面包含最多 12 個廠商的資訊
    params:
        love_firm: list, 使用者收藏的展覽 id (firm_info)
    return:
        dict, carousel
    '''
    
    firm_info_dict = dict()
    if love_firm == []:
        # 若未有喜歡廠商，隨機挑選 5 個廠商
        # 先從所有展覽中隨機挑選 5 個展覽 collection
        collections_list = random.sample(firm_db.list_collection_names(), 5)
        for exhib_collection in collections_list:
            count = firm_db[exhib_collection].count_documents({})
            if count != 0:
                random_index = random.randint(0, count - 1)
                firm_info = firm_db[exhib_collection].find().skip(random_index).limit(1).next()
                firm_info["exhib_id"] = exhib_collection # 加入展覽 id
                firm_info_dict[str(firm_info['_id'])] = firm_info
        
    # elif isinstance(love_firm, str): # 若是 str，表示只有一個廠商（的id）
    #     firm_id = love_firm
    #     # 從所有展覽中找出該廠商
    #     for exhib_collection in firm_db.list_collection_names():
    #         firm_info = firm_db[exhib_collection].find_one({"_id": ObjectId(firm_id)})
    #         # 若找到該廠商，則將其加入 firm_info_dict
    #         if firm_info:
    #             firm_info["exhib_id"] = exhib_collection # 加入展覽 id
    #             firm_info_dict[str(firm_info['_id'])] = firm_info
    #             break
    else:
        # 若有喜歡廠商，則從喜歡廠商中挑選
        for firm_id in love_firm:
            # 從所有展覽中找出該廠商
            for exhib_collection in firm_db.list_collection_names():
                firm_info = firm_db[exhib_collection].find_one({"_id": ObjectId(firm_id)})
                # 若找到該廠商，則將其加入 firm_info_dict
                if firm_info:
                    firm_info["exhib_id"] = exhib_collection # 加入展覽 id
                    firm_info_dict[str(firm_info['_id'])] = firm_info
                    break

    # 產生 carousel 的 dict
    # contents 為 bubble 的 list，每個 bubble 代表一個展覽
    contents = []
    for firm_id, firm_info in firm_info_dict.items():
        print("logo:", firm_info['logo']) # logo
        print("網址:", firm_info['url'])
        print("廠商:", firm_info['name'])
        print("展覽:", firm_info['name']) # 展覽名稱
        print("firm:", firm_id)
        print("-------------------")

        # 檢查 name 是否是 None
        if firm_info.get('name', None) is None:
            firm_info['name'] = '廠商名稱待更新'


        # 到 exhibitionDB 中找出該展覽的名稱
        exhi_db = client['exhibitionDB'] # 連接到展覽資料庫
        for exhi_collection in exhi_db.list_collection_names():
            # 到個個 collection 中找出該展覽，若找到該展覽，將其名稱加入 firm_info
            exhi_info = exhi_db[exhi_collection].find_one({"_id": ObjectId(firm_info['exhib_id'])})
            if exhi_info:
                firm_info['title'] = exhi_info['title']
                break
        if firm_info.get('title', None) is None:
            firms_data = ['展覽名稱待更新']


        # 檢查 logo 是否是 None
        if firm_info.get('logo', None) is None:
            firm_info['logo'] = 'https://developers-resource.landpress.line.me/fx/img/01_1_cafe.png'
        elif firm_info['logo'].startswith('www'):
            firm_info['logo'] = 'https://' + firm_info['logo']
        elif not firm_info['logo'].startswith('http'):
            firm_info['logo'] = 'https://developers-resource.landpress.line.me/fx/img/01_1_cafe.png'
        
        # 檢查 url 是否是 None
        if firm_info.get('url', None) is None: # 如果沒有該欄位
            firm_name_urllib = urllib.parse.quote(firm_info['name'].replace(' ', '').replace('、', ''))
            firm_info['url'] = 'https://www.google.com.tw/search?q=' + firm_name_urllib
        elif firm_info['url'].startswith('www'): # 如果是以 www 開頭
            firm_info['url'] = 'https://' + firm_info['url']
        elif firm_info['url'] == "" or firm_info['url'].startswith('http:'): 
            # 若 firm_info['url'] 是空的字串，或是以 http: 開頭
            firm_name_urllib = urllib.parse.quote(firm_info['name'].replace(' ', '').replace('、', ''))
            firm_info['url'] = 'https://www.google.com.tw/search?q=' + firm_name_urllib

        
        # todo: 檢查 check 是否是 True

        # 產生 bubble 的 str
        # firm_str = bubb_temp(firm['logo'], firm['url'], firm['name'], firms_data[0], flle_id)
        # 產生 bubble 的 dict
        print("logo:", firm_info['logo']) # logo
        print("網址:", firm_info['url'])
        print("廠商:", firm_info['name'])
        print("展覽:", firm_info['title']) # 展覽名稱
        print("firm:", firm_id)
        print("====================")
        contents.append(bubb_temp(firm_info['logo'], firm_info['url'], firm_info['name'], firm_info['title'], firm_id))

    # 產生 carousel 的 dict
    # car_dict = {"type": "carousel", "contents":contents}
    return {"type": "carousel", "contents":contents}


def list_bubble(firm_list: list) -> dict:
    '''
    產生一個 list_bubble，格式為 dict
    params:
        firm_list: list, 元素為dict，使用者收藏的展覽資訊：name、id_add
    return:
        dict, list_bubble
    '''
    contents = list()
    # colors = ['#FF0000', '#FFA500', '#FFFF00', '#008000', '#0000FF', '#4B0082', '#EE82EE']
    for firm in firm_list:
        temp_dict = {
                      "type": "text",
                      "text": firm['name'],
                      # "align": "center",
                      "margin": "lg",
                      "size": "lg",
                      "color": "#0066cc",
                      # "color": colors[firm_list.index(firm) % len(colors)],
                      "action": {
                        "type": "postback",
                        "label": "action",
                        "data": "show_firm@@@@@" + firm['id_add']
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
            "text": "LOVE FIRM",
            "weight": "bold",
            "color": "#1DB446",
            "size": "md"
          },
          {
            "type": "text",
            "text": "我的最愛廠商清單",
            "weight": "bold",
            "size": "xxl",
            "margin": "xs"
          },
          {
            "type": "text",
            "text": "可點擊廠商名稱查看詳細資訊",
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


def show_list_firm_mgdb(love_firm: list) -> dict:
    '''
    為了 show 出 list_bubble，為了讓使用者查看自己收藏的展覽
    拿到 廠商 id ，並從 資料夾 中找出對應的檔案和廠商資訊，
    再將展覽名稱與 展覽 id 放入 list_bubble 中，
    並回傳 dict 格式的 bubble
    params:
        love_firm: list, 使用者收藏的展覽 id
    return:
        dict, list_bubble
    '''

    firm_info_dict = {}
    # 讀取廠商資料
    for firm_id in love_firm:
        # 從所有展覽中找出該廠商
        for exhib_collection in firm_db.list_collection_names():
            firm_info = firm_db[exhib_collection].find_one({"_id": ObjectId(firm_id)})
            # 若找到該廠商，則將其資訊與所屬展覽 id 加入 firm_info_dict
            if firm_info:
                firm_info["exhib_id"] = exhib_collection # 加入展覽 id
                firm_info_dict[str(firm_info['_id'])] = firm_info # 將整個 firm_info 加入 firm_info_dict
                break

    for firm_id, firm in firm_info_dict.items():
        # 檢查 name 是否是 None
        if firm.get('name', None) is None:
            firm['name'] = '展覽名稱待更新'
        
        firm_info_dict[firm_id] = {'name': firm['name'], 'id_add': firm_id}

    return list_bubble(list(firm_info_dict.values()))


def show_both_mgdb(love_exhi, love_firm) -> dict:
    '''
    顯示兩個 bubble
    params:
        bubble1: dict, bubble
        bubble2: dict, bubble
    return:
        dict, carousel
    '''
    bubble1 = show_list_exhi_mgdb(love_exhi)
    bubble2 = show_list_firm_mgdb(love_firm)
    return {"type": "carousel", "contents": [bubble1, bubble2]}



def list_firm_del_bubble(firm_list: list) -> dict:
    '''
    產生一個 list_bubble，格式為 dict
    params:
        firm_list: list, 元素為dict，使用者收藏的展覽資訊：name、id_add
    return:
        dict, list_bubble
    '''
    contents = list()
    # colors = ['#FF0000', '#FFA500', '#FFFF00', '#008000', '#0000FF', '#4B0082', '#EE82EE']
    for firm in firm_list:
        temp_dict = {
            "type": "box",
            "layout": "baseline",
            "margin": "lg",
            "contents": [
            {
                "type": "text",
                "text": firm['name'],
                # "spacing": "sm",
                "size": "lg",   
                "color": "#0066cc",
                # "color": colors[firm_list.index(firm) % len(colors)],
                "action": {
                    "type": "postback",
                    "label": "action",
                    "data": "show_firm@@@@@" + firm['id_add']
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
                    "data": "del_firm@@@@@" + firm['id_add'] + "@@@@@" + firm['name']
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
            "text": "LOVE FIRM",
            "weight": "bold",
            "color": "#1DB446",
            "size": "md"
          },
          {
            "type": "text",
            "text": "我的最愛廠商清單",
            "weight": "bold",
            "size": "xxl",
            "margin": "xs"
          },
          {
            "type": "text",
            "text": "可點擊廠商名稱查看詳細資訊",
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
            "flex": 0,
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



def show_list_firm_del_mgdb(love_firm: list) -> dict:
    '''
    為了 show 出 list_bubble，為了讓使用者查看自己收藏的展覽
    拿到 廠商 id ，並從 資料夾 中找出對應的檔案和廠商資訊，
    再將展覽名稱與 展覽 id 放入 list_bubble 中，
    並回傳 dict 格式的 bubble
    params:
        love_firm: list, 使用者收藏的展覽 id
    return:
        dict, list_bubble
    '''

    firm_info_dict = {}
    # 讀取廠商資料
    for firm_id in love_firm:
        # 從所有展覽中找出該廠商
        for exhib_collection in firm_db.list_collection_names():
            firm_info = firm_db[exhib_collection].find_one({"_id": ObjectId(firm_id)})
            # 若找到該廠商，則將其資訊與所屬展覽 id 加入 firm_info_dict
            if firm_info:
                firm_info["exhib_id"] = exhib_collection # 加入展覽 id
                firm_info_dict[str(firm_info['_id'])] = firm_info # 將整個 firm_info 加入 firm_info_dict
                break

    for firm_id, firm in firm_info_dict.items():
        # 檢查 name 是否是 None
        if firm.get('name', None) is None:
            firm['name'] = '展覽名稱待更新'
        
        firm_info_dict[firm_id] = {'name': firm['name'], 'id_add': firm_id}

    return list_firm_del_bubble(list(firm_info_dict.values()))


def show_both_del_mgdb(love_exhi, love_firm) -> dict:
    '''
    顯示兩個 bubble
    params:
        bubble1: dict, bubble
        bubble2: dict, bubble
    return:
        dict, carousel
    '''
    bubble1 = show_list_exhi_del_mgdb(love_exhi)
    bubble2 = show_list_firm_del_mgdb(love_firm)
    return {"type": "carousel", "contents": [bubble1, bubble2]}





if __name__ == '__main__':

    pass
'''
              {
                "type": "button",
                "action": {
                  "type": "postback",
                  "data": "hello",
                  "label": "刪"
                },
                "style": "secondary",
                "color": "#f78888",
                "position": "relative",
                "gravity": "center",
                "flex": 0
              }
'''