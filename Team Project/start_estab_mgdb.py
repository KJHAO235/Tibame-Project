from pymongo import MongoClient

# 設定 MongoDB 連接
client = MongoClient("mongodb+srv://everybody:gac244@cluster0.mh9yw.mongodb.net/") # 鎮均
# client = MongoClient("mongodb+srv://jenhao:abcd1234@cluster0.0vluw.mongodb.net/") # 人豪
user_db = client['userDB'] # db 只有在有資料寫入時才會建立


def admin_collection():
    """
    建立 admin collection
    """
    if "admin" not in user_db.list_collection_names():
        #  user_db['admin'] 為 collection ，只有在有資料寫入時才會建立
        if not user_db['admin'].find_one({"_id": "super"}):
            user_db['admin'].insert_one({"_id": "super", "user_id": ["U8d9e233fe0444d901d8904b200c3f966"],})
        if not user_db['admin'].find_one({"_id": "admin"}):
            user_db['admin'].insert_one({"_id": "admin", "user_id": [],})


def read_user_info_mgdb(user_id) -> dict:
    """
    讀取該 user_id 的 collection，若不存在則建立一個新的
    param user_id: str
    return user_history: dict
    """
    user_collection = user_db[user_id] # collection 只有在有資料寫入時才會建立
    user_info = user_collection.find_one({"_id": user_id})
    if not user_info:
        # 如果 user_info 不存在，則建立一個新的
        user_info = {"_id": user_id, 
                    "status": None, 
                    "favorite": [],
                    "love_exhi": [],
                    "love_firm": [],
                    }
        user_collection.insert_one(user_info)
    return user_info


def write_user_info_mgdb(user_info):
    """
    寫入 user_info 至 collection
    param user_info: dict
    """
    user_id = user_info["_id"]
    user_collection = user_db[user_id]
    user_collection.update_one({"_id": user_id}, {"$set": user_info}, upsert=True)


def write_user_event_mgdb(event, img_url=None):
    """
    寫入使用者事件至 collection
    param event: LineEvent
    """
    user_id = event.source.user_id
    if event.type == "postback":
        event_info = {"timestamp": event.timestamp, 
                        "message": {"data": event.postback.data,
                                    "params": event.postback.params,
                                    }}
    elif event.message.type == "text":
        event_info = {"timestamp": event.timestamp, 
                        "message": {"id": event.message.id,
                                    "text": event.message.text,
                                    "type": event.message.type,
                                    "quote_token": event.message.quote_token,
                                    }}
    elif event.message.type == "image":
        event_info = {"timestamp": event.timestamp, 
                        "message": {"id": event.message.id,
                                    "img_url": img_url,
                                    "content_provider": event.message.content_provider.to_dict(),
                                    "quote_token": event.message.quote_token,
                                    }}

    # 插入 event_info 至 collection
    user_collection = user_db[user_id]
    user_collection.insert_one(event_info)


def add_admin(user_id, role="admin"):
    """
    將使用者加入 admin collection
    param user_id: str
    param role: str
    """
    admin_collection()
    admin = user_db['admin'].find_one({"_id": role})
    if user_id not in admin["user_id"]:
        admin["user_id"].append(user_id)
        user_db['admin'].update_one({"_id": role}, {"$set": admin}, upsert=True)


def remove_admin(user_id, role="admin"):
    """
    將使用者移出 admin collection
    param user_id: str
    param role: str
    """
    admin_collection()
    admin = user_db['admin'].find_one({"_id": role})
    if user_id in admin["user_id"]:
        admin["user_id"].remove(user_id)
        user_db['admin'].update_one({"_id": role}, {"$set": admin}, upsert=True)


def is_admin(user_id, role="admin") -> bool:
    """
    檢查使用者是否為 admin
    param user_id: str
    param role: str
    return: bool
    """
    admin_collection()
    admin = user_db['admin'].find_one({"_id": role})
    return user_id in admin["user_id"]