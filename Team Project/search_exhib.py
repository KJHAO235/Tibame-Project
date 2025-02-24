from pymongo import MongoClient
import random
from bson.objectid import ObjectId

# 設定 MongoDB 連接
client = MongoClient("mongodb+srv://everybody:gac244@cluster0.mh9yw.mongodb.net/")
exhi_db = client['exhibitionDB'] # db 只有在有資料寫入時才會建立

def search_exhib(input_word, number = 5):
    selected_exhib_id = set()
    
    for exhib_type in exhi_db.list_collection_names():
        for exhi in exhi_db[exhib_type].find():
            if input_word in exhi['title']:
                selected_exhib_id.add(exhi['_id'])
                # print(exhi['title'])

    if len(selected_exhib_id) >= number:
        selected_exhib_id_list = list(selected_exhib_id)
        random.shuffle(selected_exhib_id_list)
        return selected_exhib_id_list[:number]


    for exhib_type in exhi_db.list_collection_names():
        for exhi in exhi_db[exhib_type].find():
            if input_word in exhi['info']:
                selected_exhib_id.add(exhi['_id'])
                # print(exhi['title'])
    
    if len(selected_exhib_id) >= number:
        selected_exhib_id_list = list(selected_exhib_id)
        random.shuffle(selected_exhib_id_list)
        return selected_exhib_id_list[:number]


    for exhib_type in exhi_db.list_collection_names():
        for exhi in exhi_db[exhib_type].find():
            if input_word in exhi['keywords']:
                selected_exhib_id.add(exhi['_id'])
                # print(exhi['title'])
    
    if len(selected_exhib_id) >= number:
        selected_exhib_id_list = list(selected_exhib_id)
        random.shuffle(selected_exhib_id_list)
        return selected_exhib_id_list[:number]


    for exhib_type in exhi_db.list_collection_names():
        for exhi in exhi_db[exhib_type].find():
            for keyword in exhi['keywords']:
                if input_word in keyword:
                    selected_exhib_id.add(exhi['_id'])
                    # print(exhi['title'])

    if len(selected_exhib_id) > 0:
        selected_exhib_id_list = list(selected_exhib_id)
        random.shuffle(selected_exhib_id_list)
        return selected_exhib_id_list[:number]
    else:
        print("No exhibition found in", input_word)
        return list(selected_exhib_id)


def search_exhib_2(input_word, number = 5):
    selected_exhib_id = dict()
    
    for exhib_type in exhi_db.list_collection_names():
        for exhi in exhi_db[exhib_type].find():
            if input_word in exhi['title']:
                selected_exhib_id[exhi['_id']] = 100
                

            if input_word in exhi['info']:
                if exhi['_id'] not in selected_exhib_id:
                    selected_exhib_id[exhi['_id']] = 0
                selected_exhib_id[exhi['_id']] += exhi['info'].count(input_word)*10
    

            if input_word in exhi['keywords']:
                if exhi['_id'] not in selected_exhib_id:
                    selected_exhib_id[exhi['_id']] = 0
                selected_exhib_id[exhi['_id']] += 30
    

            for keyword in exhi['keywords']:
                if input_word in keyword:
                    if exhi['_id'] not in selected_exhib_id:
                        selected_exhib_id[exhi['_id']] = 0
                    selected_exhib_id[exhi['_id']] += 3

    
    if len(selected_exhib_id) > 0:
        # selected_exhib_id_list = list(selected_exhib_id.keys())
        # selected_exhib_id_list.sort(key=lambda x: selected_exhib_id[x], reverse=True)

        selected_exhib_id_list = sorted(selected_exhib_id.keys(), key=lambda x: selected_exhib_id[x], reverse=True)

        print("Found", len(selected_exhib_id_list), "exhibitions in", input_word)
        for obj_id in selected_exhib_id_list:
            print(obj_id, selected_exhib_id[obj_id])
        return selected_exhib_id_list[:number]
    else:
        print("No exhibition found in", input_word)
        return list(selected_exhib_id)
    

if __name__ == "__main__":
    input_word = input('請輸入要搜尋的展覽關鍵字：')
    # input_word = "寵物"
    # exhib_id_list0 = search_exhib('AI', 200)
    exhib_id_list = search_exhib_2(input_word, 200)
    print(exhib_id_list)
    for obj_id in exhib_id_list:
        for exhib_type in exhi_db.list_collection_names():
            for exhi in exhi_db[exhib_type].find():
                if exhi['_id'] == obj_id:
                    print(exhi['title'])
                    print(exhi['info'])
                    print("===")