import os
import re
import sys
import json
import timeit
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from configparser import ConfigParser
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from pymongo import MongoClient
from bson import ObjectId  # 確保正確處理 MongoDB 的 ObjectId

# 禁用 TensorFlow 的 oneDNN 優化
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# 禁用 Transformers 的 TensorFlow 加載
os.environ["TRANSFORMERS_NO_TF"] = "1"

try:
    config = ConfigParser()
    config.read("config.ini", encoding="utf-8")
    DEPLOYMENT_NAME_EMBEDDING_LARGE = config.get("AzureOpenAI", "DEPLOYMENT_NAME_EMBEDDING_LARGE")
    VERSION = config.get("AzureOpenAI", "VERSION")
    KEY = config.get("AzureOpenAI", "KEY")
    ENDPOINT = config.get("AzureOpenAI", "ENDPOINT")
    GPT4o_DEPLOYMENT_NAME = config.get("AzureOpenAI", "GPT4o_DEPLOYMENT_NAME")
    if not DEPLOYMENT_NAME_EMBEDDING_LARGE or not VERSION or not KEY or not ENDPOINT or not GPT4o_DEPLOYMENT_NAME:
        raise ValueError("Config file variables not set properly")
except Exception as e:
    print(f"Failed to load config file: {e}")
    try:
        DEPLOYMENT_NAME_EMBEDDING_LARGE = os.getenv("DEPLOYMENT_NAME_EMBEDDING_LARGE")
        VERSION = os.getenv("VERSION")
        KEY = os.getenv("KEY")
        ENDPOINT = os.getenv("ENDPOINT")
        GPT4o_DEPLOYMENT_NAME = os.getenv("GPT4o_DEPLOYMENT_NAME")
    except Exception as e:
        print(f"Failed to load config file: {e}")
        print("Please set the environment variables or provide a valid config.ini file.")
        sys.exit(1)


# 1. 加載數據集
def load_dataset_from_mongodb_collection(collection_name):
    """從 MongoDB 的特定集合中加載廠商資料。

    Args:
        collection_name (str): 要爬取文檔的集合名稱。

    Returns:
        list: 包含文檔的 Document 對象列表。
    """
    client = MongoClient("mongodb+srv://jenhao:abcd1234@cluster0.0vluw.mongodb.net/")
    # 切換到特定資料庫
    db_cp = client["companyDB"]

    # 確認集合名稱是否存在
    collections = db_cp.list_collection_names()  # 獲取資料庫中的所有集合名稱
    # print(f"發現集合: {collections}")  # 調試打印集合名稱

    if collection_name not in collections:
        print(f"集合 {collection_name} 不存在於資料庫中！")
        return []

    # 選擇特定集合
    collection = db_cp[collection_name]

    documents = []
    for record in collection.find():
        # 提取所需字段
        info = (record.get("info") or "").strip()
        doc_type = (record.get("type") or "").strip()
        name = (record.get("name") or "").strip()
        # 檢查 info 是否包含 "廠商分類：" + {type}
        match_phrase = f"廠商分類：{doc_type}"
        if not info or match_phrase in info:
            page_content = f"{name}\n{doc_type}".strip()  # 若匹配則改用廠商名稱
        else:
            page_content = f"{info}\n{doc_type}".strip()  # 否則保留原 info

        # 添加到文檔列表
        documents.append(Document(
            page_content=page_content,
            metadata={
                "_id": record.get("_id", ""),
                "name": name,
                "id": record.get("id", ""),
                "type": doc_type,
                "collection": collection_name,  # 添加集合名稱作為來源元數據
            }
        ))
    # print(f"加載文檔:{documents}")
    print(f"從集合 {collection_name} 共加載了 {len(documents)} 條資料。")
    return documents

# 4. 輸入分類與匹配
def classify_and_complete_input(user_input, processed_documents):
    """分類用戶輸入，並返回匹配的結果。

    Args:
        user_input (str): 用戶輸入的文字。
        processed_documents (list): 處理後的 Document 列表。

    Returns:
        tuple: 包含輸入類型和匹配信息的元組。
    """
    # 優先匹配攤位編號
    for doc in processed_documents:
        if "id" in doc.metadata:
            id_value = doc.metadata.get("id", "").strip().upper()
            if id_value == user_input.strip().upper():
                return "store", doc

    # 匹配部分公司名稱
    matched_stores = [
        doc for doc in processed_documents
        if user_input in doc.metadata.get("name", "")
    ]

    # 去重，根據 id 來確保唯一性
    unique_stores = {doc.metadata.get("id", ""): doc for doc in matched_stores}

    # 判定過於廣泛
    if len(unique_stores) > 5:
        return "too_many_matches", list(unique_stores.values())

    # 精準匹配單個公司
    if len(unique_stores) == 1:
        return "store", list(unique_stores.values())[0]

    # 默認為描述類型
    return "object", user_input

# 5. 根據輸入類型進行推薦
def recommend_based_on_input(input_type, matching_info, user_input, db, k=4):
    """根據分類結果進行推薦。
    Args:
        input_type (str): 輸入類型，可為 "store", "too_many_matches", "object"。
        matching_info: 匹配的相關資料，可能是單個店家或多個店家列表。
        user_input (str): 用戶輸入。
        db (FAISS): 向量索引對象。
        k (int): 返回的推薦結果數量。
    """
    if input_type == "too_many_matches":
        print("輸入過於廣泛，請重新輸入更精確的內容。")
        return
    if input_type == "store":
        query = f"I like {matching_info.metadata.get('name', '未知店家名稱')}，Which stores could you suggest to me?"
        results = db.similarity_search_with_score(query, k=k * 2)
        grouped_results = {}
        for doc, score in results:
            store_name = doc.metadata.get("name", "未知店家")
            if store_name not in grouped_results or grouped_results[store_name]["score"] > score:
                grouped_results[store_name] = {"doc": doc, "score": score}
        return sorted(grouped_results.values(), key=lambda x: x["score"])[:k]
    elif input_type == "object":
        query = f"I like {user_input} ，Which stores could you suggest to me?"
    results = db.similarity_search_with_score(query, k=k * 2)
    grouped_results = {}
    for doc, score in results:
        store_name = doc.metadata.get("name", "未知店家")
        if store_name not in grouped_results or grouped_results[store_name]["score"] > score:
            grouped_results[store_name] = {"doc": doc, "score": score}
    return sorted(grouped_results.values(), key=lambda x: x["score"])[:k]

# 6. 加載 LLM 模型和 prompt
def initialize_llm():
    """初始化 LLM 模型和提示模板。

    Returns:
        StuffDocumentsChain: 文檔處理鏈。
    """
    llm = AzureChatOpenAI(
        azure_deployment=GPT4o_DEPLOYMENT_NAME,
        openai_api_version=VERSION,
        api_key=KEY,
        azure_endpoint=ENDPOINT,
    )
    prompt = ChatPromptTemplate.from_template(
        """ You are a helpful guide to recommend exhibitions.
        Answer the following question based only on the provided exhibitions context:
        <context>
        {context}
        </context>
        Question: {input}
        Based on the user's interest in "{input}", recommend 3 stores from the provided context.
        Your recommendations should:
        1.The recommendations have similar points.
        2.Include the store name and unique highlights in **a single sentence with no more than 50 words**.
        3.Translate all responses into Traditional Chinese.
        Please use the following structure for your answer:
        1. 店家名稱: [Name]
           獨特亮點: [Highlights]
        2. 店家名稱: [Name]
           獨特亮點: [Highlights]
        3. 店家名稱: [Name]
           獨特亮點: [Highlights]
        4. 店家名稱: [Name]
           獨特亮點: [Highlights]
        若無相關雷同點時，描述為：可能會有[與該店家的關聯性說明]，然後介紹該店家的亮點。 
        """
    )
    #in one sentences
    print("LLM 模型已初始化完成。")
    return create_stuff_documents_chain(llm, prompt)

# 7. 生成 LLM 推薦結果
def generate_llm_recommendations(chain, query, context_docs):
    """生成基於 LLM 的推薦結果。

    Args:
        chain (StuffDocumentsChain): 文檔處理鏈。
        query (str): 查詢字符串。
        context_docs (list): 上下文文檔列表。

    Returns:
        str: LLM 推薦的答案。
    """
    context_docs = [
        Document(
            page_content=f"店家名稱: {doc.metadata['name']}\n"
                         f"內容: {doc.page_content}",
            metadata=doc.metadata
        )
        for doc in context_docs
    ]
    print(f"生成推薦中context_docs：{context_docs}")
    # 生成推薦中context_docs：[Document(metadata={'_id': ObjectId('677c94b6cd8df06c18b4f096'), 'name': '聖比德蛋糕有限公司', 'id': '尚未更新', 'type': '名店街', 'collection': '677c94b5cd8df06c18b4f021', 'chunk_index': 0}, page_content='店家名稱: 聖比德蛋糕有限公司\n內容: 聖比德蛋糕有限公司\n名店街')]
    print(f"生成推薦中query：{query}")
    # 調用 chain.invoke 方法，確保傳遞了 input 和 context 屬性
    return chain.invoke({"input": query, "context": context_docs})

def parse_llm_response(llm_result: str) -> dict:
    """
    從 LLM 回答中，解析出「展覽名稱 -> 亮點」的字典映射。
    
    Args:
        llm_result (str): LLM 的文字回答
    
    Returns:
        highlight_map (dict): { '2025 台北國際自動化工業大展': '聚焦工業4.0...', 
                                '2025 智慧顯示展覽會': '展示智慧顯示...', 
                                '2025 台灣機器人與智慧自動化展': '匯集全球...' }
    """
    # 你也可以依實際 LLM 回答格式調整 regex
    # 假設格式大致為：
    # 1. 展覽名稱: XXXXX
    #    獨特亮點: YYYYY
    pattern = re.compile(
        r"\d+\.\s*店家名稱:\s*(.*?)\s*獨特亮點:\s*(.*)"
    )
    matches = pattern.findall(llm_result)

    highlight_map = {}
    for name, highlight in matches:
        # 去掉前後空白
        exhibition_name = name.strip()
        exhibition_highlight = highlight.strip()
        highlight_map[exhibition_name] = exhibition_highlight

    return highlight_map

def save_llm_results(llm_result, context_docs):
    highlight_map = parse_llm_response(llm_result)

    doc_id_insert = {}
    # 🔴 原本是 for doc in context_docs[1:]:
    for doc in context_docs:  
        if "_id" in doc.metadata:
            _id = doc.metadata["_id"]
            name = doc.metadata.get("name", "未知展覽")
            matched_highlight = highlight_map.get(name, "")
            # 如果 matched_highlight 為空，代表 LLM 沒有提供它的亮點，就直接跳過
            if not matched_highlight:
                continue
            doc_id_insert[_id] = {
                "name": name,
                "highlight": matched_highlight
            }

    print("解析結果:doc_id_insert ", doc_id_insert)
    print(f"解析結果：type", type(doc_id_insert))
    return doc_id_insert

# def save_llm_results(llm_result, context_docs):
#     """解析 LLM 推薦結果並保存為字典。

#     Args:
#         llm_result (str): LLM 的答案字符串。
#         context_docs (list): 與推薦相關的文檔上下文。

#     Returns:
#         dict: 包含推薦結果的字典。
#     """
#     parsed_results = {}
#     doc_id_insert = {
#         doc.metadata["_id"]: {"name": doc.metadata.get("name", "未知店家")}
#         for doc in context_docs[1:] if "_id" in doc.metadata
#     }
#     print(f"建立doc_id_insert: {doc_id_insert}")
#     highlight = [hl.split('獨特亮點: ')[-1].strip() for hl in llm_result.split('\n\n')]
#     print(f"highlight: {highlight}")

#     # # 移除說明性文字，確保高亮與店家對應################################################################
#     # if len(highlight) > len(doc_id_insert):
#     #     highlight = highlight[1:]  # 略過第一項

#     for i, (doc_id, doc_info) in enumerate(doc_id_insert.items()):
#         doc_info["highlight"] = highlight[i]
    
#     print("解析結果:doc_id_insert ", doc_id_insert)
#     print(f"解析結果：type", type(doc_id_insert))
#     return doc_id_insert

# 8. 保存推薦結果
# def save_llm_results(llm_result, context_docs):
#     """解析 LLM 推薦結果並保存為字典。

#     Args:
#         llm_result (str): LLM 的答案字符串。
#         context_docs (list): 與推薦相關的文檔上下文。

#     Returns:
#         dict: 包含推薦結果的字典。
#     """
#     # 構建 doc_id_insert 字典
#     doc_id_insert = {
#         doc.metadata["_id"]: {"name": doc.metadata.get("name", "未知店家")}
#         for doc in context_docs if "_id" in doc.metadata
#     }
#     print(f"建立 doc_id_insert: {doc_id_insert}")

#     # 用正則表達式提取所有「獨特亮點」
#     # 匹配形如 "獨特亮點: ..." 的內容
#     highlight = re.findall(r'獨特亮點: (.+)', llm_result)
#     print(f"提取的 highlight: {highlight}")

#     # 確保 highlight 與 doc_id_insert 的數量一致
#     if len(highlight) != len(doc_id_insert):
#         raise ValueError(f"highlight 的數量 ({len(highlight)}) 與 doc_id_insert ({len(doc_id_insert)}) 不匹配。請檢查輸入數據。")

#     # 分配 highlight 到 doc_id_insert
#     for i, (doc_id, doc_info) in enumerate(doc_id_insert.items()):
#         doc_info["highlight"] = highlight[i]
    
#     print("解析結果: doc_id_insert ", doc_id_insert)
#     return doc_id_insert

# 9. 初始化環境
def initialize_environment_store(current_collection, config_path="config.ini"):
    """初始化環境，包括嵌入向量、FAISS 資料庫和文檔加載。

    Args:
        current_collection (str): 展覽集合名稱。
        config_path (str): 配置檔案路徑。

    Returns:
        tuple: (vector_stores, processed_documents_dict)
    """
    # 加載配置
    config = ConfigParser()
    config.read(config_path)

    # 初始化嵌入向量模型
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=config["AzureOpenAI"]["DEPLOYMENT_NAME_EMBEDDING_LARGE"],
        openai_api_version=config["AzureOpenAI"]["VERSION"],
        api_key=config["AzureOpenAI"]["KEY"],
        azure_endpoint=config["AzureOpenAI"]["ENDPOINT"],
    )

    # 構建 FAISS 路徑
    faiss_path = os.path.join("company_db_faiss", current_collection)

    # 加載 FAISS 向量資料庫
    try:
        prebuilt_faiss = FAISS.load_local(
            faiss_path,
            embeddings,
            "index",
            allow_dangerous_deserialization=True
        )
        print(f"成功加載 FAISS 資料庫: {faiss_path}")
    except Exception as e:
        print(f"加載 FAISS 資料庫失敗: {e}")
        sys.exit(1)

    # 加載文檔
    documents = load_dataset_from_mongodb_collection(current_collection)
    if not documents:
        print("未能加載任何文檔，請檢查集合名稱或資料庫連接。")
        sys.exit(1)

    # 構建 vector_stores 和 processed_documents_dict
    vector_stores = {current_collection: prebuilt_faiss}
    processed_documents_dict = {current_collection: documents}
    
    return vector_stores, processed_documents_dict

# 10. 使用者互動處理
def user_interaction_store(current_collection, user_input, vector_stores, processed_documents_dict):
    """處理使用者輸入，基於已知的當前展覽集合進行檢索和推薦。

    Args:
        current_collection (str): 當前展覽的集合名稱。
        vector_stores (dict): 向量索引字典，包含每個集合的向量索引。
        processed_documents_dict (dict): 文檔列表字典，包含每個集合的處理後文檔。
    """
    # 驗證集合名稱是否有效
    if current_collection not in vector_stores:
        print(f"無效的集合名稱: {current_collection}")
        return {"result": f"無效的集合名稱: {current_collection}"}

    # 獲取對應集合的數據
    db = vector_stores[current_collection]
    processed_documents = processed_documents_dict[current_collection]

    # 提示用戶輸入查詢，例如：
    # user_input = "香帥蛋糕"

    # 分類用戶輸入
    input_type, matching_info = classify_and_complete_input(user_input, processed_documents)

    # 根據分類進行推薦
    if input_type == "too_many_matches":
        print(f"匹配到的店家超過 5 個 ({len(matching_info)} 個)，請重新輸入更精確的內容。")
        return {"result": "too_many_matches"}
    # 根據分類結果執行檢索或推薦
    if input_type in ["store", "object"]:
        recommend_based_on_input(input_type, matching_info, user_input, db)
        print(f"根據用戶輸入類型 {input_type} 進行推薦。")

        # 初始化 LLM 並生成推薦結果
        document_chain = initialize_llm()
        
        # 執行分類與檢索
        top_results = recommend_based_on_input(
            input_type,
            matching_info=matching_info,
            user_input=user_input,
            db=db,
            k=3
        )

        # 如果沒有檢索結果，退出
        if not top_results:
            return {"result": "not top_results，沒有檢索結果"}
        
        # 打印檢索結果
        print("檢索結果：")
        for result in top_results:
            doc = result["doc"]
            score = result["score"]
            print(f"店家名稱: {doc.metadata['name']}")
            print(f"段落內容: {doc.page_content}")
            print(f"相似度分數: {score}")
            print("-" * 50)
        # 初始化 LLM 並生成推薦結果
        document_chain = initialize_llm()
        context_docs = [result["doc"] for result in top_results]
        print(f"這裡是context_docs：{context_docs}")
        llm_result = generate_llm_recommendations(document_chain, user_input, context_docs)


        # 打印並保存 LLM 推薦結果
        print("LLM Answer: ", llm_result)
        print('context_docs:', context_docs)
        return save_llm_results(llm_result, context_docs)

# 11. 計算記憶體大小
def calculate_faiss_memory(vector_store):
    """計算 FAISS 向量索引的記憶體佔用大小。

    Args:
        vector_store (FAISS): FAISS 向量索引對象。

    Returns:
        int: 記憶體佔用大小（以 bytes 為單位）。
    """
    index = vector_store.index  # 獲取 FAISS 索引
    dimension = index.d  # 向量的維度
    num_vectors = index.ntotal  # 向量的數量
    vector_size = dimension * 4  # 假設每個數值是 float32（4 bytes）

    total_size = num_vectors * vector_size
    return total_size


if __name__ == "__main__":
    # 計算初始化執行時間
    # start_time = timeit.default_timer()
    # vector_stores, processed_documents_dict = backend_initialize_store()
    # end_time = timeit.default_timer()
    # execution_time = end_time - start_time
    # print(f"backend_initialize_store() 執行時間: {execution_time:.4f} 秒")

    # 計算 FAISS 向量索引的記憶體大小
    # total_memory = 0
    # for collection_name, vector_store in vector_stores.items():
    #     memory_size = calculate_faiss_memory(vector_store)
    #     total_memory += memory_size
    #     print(f"集合 {collection_name} 的向量索引記憶體大小: {memory_size / (1024 ** 2):.2f} MB")

    # print(f"所有向量索引的總記憶體大小: {total_memory / (1024 ** 2):.2f} MB")

    # 指定當前展覽的集合名稱，蘇哥請改這裡
    current_collection = "677c94b5cd8df06c18b4f021"
    user_input = "糕"

    # 初始化環境
    vector_stores, processed_documents_dict = initialize_environment_store(current_collection)


    # 使用者互動，根據當前展覽的集合進行操作
    a = user_interaction_store(current_collection, user_input, vector_stores, processed_documents_dict)
    print('++++++++++++++++++++++++++')
    print(a)
    # 輸出範例：
    # {ObjectId('677c94b6cd8df06c18b4f096'): {'name': '聖比德蛋糕有限公司',
    # 'highlight': '提供多種口味的精緻蛋糕，位於名店街的知名甜點店。'},
    # ObjectId('677c94b6cd8df06c18b4f0a1'): {'name': '易德食食品有限公司',
    # 'highlight': '以多樣化的糕點選擇聞名，名店街中深受顧客喜愛。'},
    # ObjectId('677c94b6cd8df06c18b4f06c'): {'name': '一之軒食品有限公司',
    # 'highlight': '以高品質的糕點產品著稱，名店街的糕點愛好者必訪之地。'},
    # ObjectId('677c94b6cd8df06c18b4f09e'): {'name': '超品企業股份有限公司',
    # 'highlight': '提供創新口味的糕點選擇，名店街上的甜點創新者。'}}

