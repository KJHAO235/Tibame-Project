import os
from google.cloud import firestore


# 初始化 Firestore 客戶端
def initialize_firestore_with_key(json_key_path):
    try:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_key_path
        project_id = "order-assistant-20250214"
        dbid = 'user-state-20250214'
        client = firestore.Client(project=project_id, database=dbid)
        print("Firestore 客戶端初始化成功")
        return client
    except Exception as e:
        print(f"初始化 Firestore 客戶端時發生錯誤: {e}")
        return None

# 初始化 Firestore 客戶端（全域初始化，只執行一次）
db = initialize_firestore_with_key("firestore.json")

# 確保用戶狀態在 Firestore 中存在
def ensure_user_state_in_firestore(user_id):
    """ 確保用戶在 Firestore 中有資料，若沒有則創建預設資料 """
    if db:
        try:
            doc_ref = db.collection("user_states").document(user_id)
            doc = doc_ref.get()

            if not doc.exists:
                # 如果用戶資料不存在，建立初始資料
                default_data = {
                    "state": "",
                    "audio_url": None,
                    "image_url": None,
                    "speech_info": None,
                    "image_info": '繁體中文',
                    "food_list": [],
                    "menu_language": None,
                }
                doc_ref.set(default_data)
                print(f"新用戶 {user_id} 資料已創建")
                return default_data
            else:
                print(f"用戶 {user_id} 已存在，讀取現有資料")
                return doc.to_dict()
        except Exception as e:
            print(f"Firestore 錯誤: {e}")
            return None
    else:
        print("Firestore 客戶端尚未初始化")

# 更新用戶狀態的部分欄位（避免覆蓋其他數據）
def update_user_state_in_firestore(user_id, update_data):
    if db:
        try:
            db.collection("user_states").document(user_id).update(update_data)
            print(f"成功更新用戶 {user_id} 的狀態")
        except Exception as e:
            print(f"更新失敗: {e}")
    else:
        print("Firestore 客戶端尚未初始化")

# 刪除用戶狀態
def delete_user_state_from_firestore(user_id):
    if db:
        db.collection("user_states").document(user_id).delete()
        print(f"成功刪除用戶狀態：{user_id}")
    else:
        print("Firestore 客戶端尚未初始化")


if __name__ == "__main__":
    # 測試用戶狀態的讀取、寫入和刪除
    user_id = "test_user1"
    state_data = {"state": "order", "food": "pizza"}

    # 確保用戶狀態存在
    user_info = ensure_user_state_in_firestore(user_id)

    # 更新用戶狀態
    update_user_state_in_firestore(user_id, {"new1": [{'food_original_language_name': 'XiaoLongBao', 'food_translation_name': '小籠包', 'ingredients': ['pork', 'crab roe', 'chicken', 'green squash', 'truffle'], 'food_id': '2e80e7ae'}, {'food_original_language_name': 'Steamed Vegetable and Ground Pork Dumplings', 'food_translation_name': '蒸蔬菜和豬肉餡餃子', 'ingredients': ['vegetable', 'pork'], 'food_id': '3a40cb60'}]})
    
    # 刪除用戶狀態
    delete_user_state_from_firestore(user_id)
    
    # 確保用戶狀態不存在
    # ensure_user_state_in_firestore(user_id)
