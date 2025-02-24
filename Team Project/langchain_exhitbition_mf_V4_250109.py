import os
import json
import re
# from pprint import pprint
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from pymongo import MongoClient
import sys
from configparser import ConfigParser
from bson import ObjectId  # 確保正確處理 MongoDB 的 ObjectId

# 初始化配置解析器
# config = ConfigParser()
# config.read("config.ini", encoding="utf-8")

try:
    DEPLOYMENT_NAME_EMBEDDING_LARGE = os.getenv("DEPLOYMENT_NAME_EMBEDDING_LARGE")
    VERSION = os.getenv("VERSION")
    KEY = os.getenv("KEY")
    ENDPOINT = os.getenv("ENDPOINT")
    GPT4o_DEPLOYMENT_NAME = os.getenv("GPT4o_DEPLOYMENT_NAME")

    if not all([DEPLOYMENT_NAME_EMBEDDING_LARGE, VERSION, KEY, ENDPOINT, GPT4o_DEPLOYMENT_NAME]):
        raise ValueError("One or more environment variables are missing.")
except Exception as e:
    print(f"Error loading environment variables: {e}")
    try:
        config = ConfigParser()
        config.read("config.ini", encoding="utf-8")

        DEPLOYMENT_NAME_EMBEDDING_LARGE = config.get("AzureOpenAI", "DEPLOYMENT_NAME_EMBEDDING_LARGE")
        VERSION = config.get("AzureOpenAI", "VERSION")
        KEY = config.get("AzureOpenAI", "KEY")
        ENDPOINT = config.get("AzureOpenAI", "ENDPOINT")
        GPT4o_DEPLOYMENT_NAME = config.get("AzureOpenAI", "GPT4o_DEPLOYMENT_NAME")
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1)
        




# 禁用 TensorFlow 的 oneDNN 優化
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# 禁用 Transformers 的 TensorFlow 加載
os.environ["TRANSFORMERS_NO_TF"] = "1"


def load_dataset_from_mongodb() -> list:
    """從 MongoDB 的 exhibitionDB 的所有 collection 加載展覽資料。

    Returns:
        list: 包含所有展覽數據的 Document 形成的列表。
    """
    client = MongoClient("mongodb+srv://jenhao:abcd1234@cluster0.0vluw.mongodb.net/")
    db_ex = client["exhibitionDB"] 
    
    documents = []
    collections = db_ex.list_collection_names()  # 獲取資料庫中的所有集合名稱
    print(f"發現集合: {collections}")  # 調試打印集合名稱

    for collection_name in collections:
        # 對所有 collection 進行迭代，每個 collection 都是一個（展覽的）種類，例如「旅遊展」
        collection = db_ex[collection_name]

        # 對每個 collection 中的所有文檔（這裡是record，即 MongoDB 中的一條記錄）進行迭代
        for record in collection.find(): 
            info = record.get("info", "").strip() # 獲取展覽資訊
            # print(f"檢查集合 {collection_name} 的 info: {info}")  # 打印 info

            documents.append(Document(
                    page_content=info,
                    metadata={
                        "_id": record.get("_id", ""),
                        "title": record.get("title", ""),
                        "date": record.get("date", ""),
                        "location": record.get("location", ""),
                        "collection": collection_name,  # 添加集合名稱作為來源元數據
                        "info": info
                    }
                ))
    print(f"從 MongoDB 加載了 {len(documents)} 條展覽資料。")
    return documents


# def process_documents(documents, chunk_size=75, chunk_overlap=10):
#     """將文檔進行文本切割並處理。
#     Args:
#         documents (list): 原始文檔列表。
#         chunk_size (int, optional): 每段文字的大小。設定為 50。
#         chunk_overlap (int, optional): 每段文字的重疊大小。設定為 10。
#     Returns:
#         list: 處理後的文檔列表。
#     """
#     # 初始化文本切割器
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         length_function=len
#     )

#     processed_docs = []
#     for doc in documents: # documents 是 Documnet 的list， Documnet 是 langchain.schema 中的一個類
#         try:
#             # 獲取文檔內容
#             info = doc.page_content.strip()
#             name = doc.metadata.get("title", "")
#             if not info:  # 如果內容為空，用展覽名稱替代
#                 info = name
            
#             # 切割文本
#             split_texts = splitter.split_text(info)
#             for idx, split_text in enumerate(split_texts):
#                 processed_docs.append(Document(
#                     page_content=split_text,
#                     metadata={
#                         "_id": str(doc.metadata.get('_id', '未知 _id')),
#                         "title": name,
#                         "date": doc.metadata.get("date", ""),
#                         "chunk_index": idx
#                     }
#                 ))
#         except Exception as e:
#             print(f"處理文檔時出錯: {doc.page_content}, 錯誤信息: {e}")
#             continue
#     return processed_docs

# # 3. 初始化向量索引
# def initialize_faiss(processed_documents):
#     """初始化 FAISS 向量索引。
#     Args:
#         processed_documents (list): 處理後的文檔列表。
#     Returns:
#         FAISS: 初始化的向量索引。
#     """
#     embeddings = AzureOpenAIEmbeddings(
#         azure_deployment=DEPLOYMENT_NAME_EMBEDDING_LARGE,
#         openai_api_version=VERSION,
#         api_key=KEY,
#         azure_endpoint=ENDPOINT,
#     )
#     print("向量索引已初始化完成。")
#     return FAISS.from_documents(processed_documents, embeddings)

# 4. 檢索功能
def retrieve_top_by_exhibition(db, query, k=4):
    """檢索與查詢相關的展覽。
    Args:
        db (FAISS): 向量索引。
        query (str): 查詢字符串。
        k (int, optional): 返回的結果數量。預設為 4。
    Returns:
        list: 最相關的展覽結果。
    """
    results = db.similarity_search_with_score(query, k = k*2)
    grouped_results = {}
    for doc, score in results:
        exhibition = doc.metadata.get("title", "未知展覽")
        if exhibition not in grouped_results or grouped_results[exhibition]["score"] > score:
            grouped_results[exhibition] = {"doc": doc, "score": score}
    
    print(f"檢索結果: {grouped_results}")
    return sorted(grouped_results.values(), key=lambda x: x["score"])[:k]

# 5. 根據 _id 查詢展覽標題進行切換
def find_exhibition_title_by_id(documents, input_id):
    """
    根據使用者輸入的 MongoDB Atlas 格式 _id 查詢展覽標題。

    Args:
        documents (list): 包含展覽數據的 Document 列表。
        input_id (str): 使用者輸入的 _id 字串。

    Returns:
        str: 對應的展覽標題，若未找到則返回提示訊息。
    """
    for document in documents:
        # 提取 _id 的 $oid 值
        # document_id = document.metadata["_id"].get("$oid") if isinstance(document.metadata["_id"], dict) else None
        # if document_id == input_id:
        #     return document.metadata.get("title", "未找到標題")
        if str(document.metadata["_id"]) == input_id:
            return document.metadata.get("title", "未找到標題")
        # print(f"未找到對應的展覽，請檢查 _id 是否正確。")
    return "未找到對應的展覽"

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
        Based on the user's interest in "{input}", recommend 3 exhibitions from the provided context.
        Your recommendations should:
        1.The recommendations have similar points.
        2.Include the exhibition name and unique highlights in **a single sentence with no more than 50 words**..
        3.Translate all responses into Traditional Chinese.
        Please use the following structure for your answer:
        1. 展覽名稱: [Name]
           獨特亮點: [Highlights]
        2. 展覽名稱: [Name]
           獨特亮點: [Highlights]
        3. 展覽名稱: [Name]
           獨特亮點: [Highlights]
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
            page_content=f"展覽名稱: {doc.metadata.get('title', '未知展覽')}\n"
                         f"日期: {doc.metadata.get('date', '未知日期')}\n"
                         f"內容: {doc.page_content}",
            metadata=doc.metadata
        )
        for doc in context_docs
    ]
    print(f"生成推薦中{context_docs}")
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
        r"\d+\.\s*展覽名稱:\s*(.*?)\s*獨特亮點:\s*(.*)"
    )
    matches = pattern.findall(llm_result)

    highlight_map = {}
    for name, highlight in matches:
        # 去掉前後空白
        exhibition_name = name.strip()
        exhibition_highlight = highlight.strip()
        highlight_map[exhibition_name] = exhibition_highlight

    return highlight_map

def save_llm_results_as_dict(llm_result, context_docs):
    """解析 LLM 推薦結果並保存為字典。

    Args:
        llm_result (str): LLM 的答案字符串。
        context_docs (list): 與推薦相關的文檔上下文。

    Returns:
        dict: 包含推薦結果的字典。
    """
    # 第一步：先用剛才定義好的函式，取得「展覽名稱 -> highlight」的 map
    highlight_map = parse_llm_response(llm_result)

    doc_id_insert = {}
    # 🔴 原本是 for doc in context_docs[1:]:
    for doc in context_docs:  
        if "_id" in doc.metadata:
            _id = doc.metadata["_id"]
            name = doc.metadata.get("title", "未知展覽")
            matched_highlight = highlight_map.get(name, "")
            # 如果 matched_highlight 為空，代表 LLM 沒有提供它的亮點，就直接跳過
            if not matched_highlight:
                continue
            doc_id_insert[_id] = {
                "title": name,
                "highlight": matched_highlight
            }

    print("解析結果:doc_id_insert ", doc_id_insert)
    print(f"解析結果：type", type(doc_id_insert))
    return doc_id_insert

    # 第二步：建立 doc_id_insert 結構，並且依展覽名稱對應到 highlight
    # doc_id_insert = {}
    # for doc in context_docs[1:]:  # 你的原始程式中好像是從 context_docs[1:] 開始
    #     if "_id" in doc.metadata:
    #         _id = doc.metadata["_id"]
    #         title = doc.metadata.get("title", "未知展覽")

    #         # 根據「title」到 highlight_map 查找對應的亮點
    #         matched_highlight = highlight_map.get(title, "")

    #         doc_id_insert[_id] = {
    #             "title": title,
    #             "highlight": matched_highlight
    #         }
    
    # print("解析結果:doc_id_insert ", doc_id_insert)
    # print(f"解析結果：type", type(doc_id_insert))
    # return doc_id_insert


# 8. 保存推薦結果為 dict
# def save_llm_results_as_dict(llm_result, context_docs):
#     """解析 LLM 推薦結果並保存為字典。

#     Args:
#         llm_result (str): LLM 的答案字符串。
#         context_docs (list): 與推薦相關的文檔上下文。

#     Returns:
#         dict: 包含推薦結果的字典。
#     """
#     parsed_results = {}

#     # 建立標題到 _id 的映射，僅處理包含 _id 的文檔
#     doc_id_insert = {
#         doc.metadata["_id"]: {"title": doc.metadata.get("title", "未知展覽")}
#         for doc in context_docs[1:] if "_id" in doc.metadata
#     }
#     print(f"建立doc_id_insert: {doc_id_insert}")
    
#     highlight = [hl.split('獨特亮點: ')[-1].strip() for hl in llm_result.split('\n\n')]
#     print(f"highlight: {highlight}")

#     # # 移除說明性文字，確保高亮與店家對應################################################################
#     if len(highlight) > len(doc_id_insert):
#         highlight = highlight[1:]  # 略過第一項

#     for i, (doc_id, doc_info) in enumerate(doc_id_insert.items()):
#         doc_info["highlight"] = highlight[i]
    
#     # print("解析結果:parsed_results ", parsed_results)
#     # print(f"解析結果： ", type(parsed_results))
#     print("解析結果:doc_id_insert ", doc_id_insert)
#     print(f"解析結果：type", type(doc_id_insert))
#     return doc_id_insert

# def save_llm_results_as_dict(llm_result, context_docs):
#     """解析 LLM 推薦結果並保存為字典。

#     Args:
#         llm_result (str): LLM 的答案字符串。
#         context_docs (list): 與推薦相關的文檔上下文。

#     Returns:
#         dict: 包含推薦結果的字典。
#     """
#     # 構建 doc_id_insert 字典
#     doc_id_insert = {
#         doc.metadata["_id"]: {"title": doc.metadata.get("title", "未知展覽")}
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

# def save_llm_results_as_dict(llm_result, context_docs):
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

#     for i, (doc_id, doc_info) in enumerate(doc_id_insert.items()):
#         doc_info["highlight"] = highlight[i]
    
#     print("解析結果:doc_id_insert ", doc_id_insert)
#     print(f"解析結果：type", type(doc_id_insert))
#     return doc_id_insert

def initialize_environment_exh(config_path="config.ini"):
    """初始化嵌入向量和 FAISS 資料庫。

    Args:
        config_path (str): 配置檔案路徑。
        faiss_path (str): FAISS 資料庫的路徑。

    Returns:
        object: 已加載的 FAISS 資料庫。
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

    # 加載 FAISS 向量資料庫

    faiss_path = os.path.join("exhibition_db_faiss")
    try:
        prebuilt_faiss = FAISS.load_local(
            faiss_path,
            embeddings,
            "index",
            allow_dangerous_deserialization=True
        )
        print(f"成功加載 FAISS 資料庫: {faiss_path}")
        return prebuilt_faiss
    except Exception as e:
        print(f"加載 FAISS 資料庫失敗: {e}")
        raise

# 使用者互動處理
def user_interaction(db, latest_love_exhib_id):
    """處理使用者輸入，進行檢索和推薦。
    Args:
        db (FAISS): 已初始化的向量索引。
    """

    # 根據我的最愛最新一筆的_id，蘇哥請改這裡
    # latest_love_exhib_id = "677c94b5cd8df06c18b4f021"

    # 查詢展覽標題
    documents_db = load_dataset_from_mongodb()
    user_input = find_exhibition_title_by_id(documents_db, latest_love_exhib_id)
    print(f"對應的展覽標題: {user_input}")

    # 構建查詢
    query = f"I enjoy {user_input} ，Which exhibitions could you suggest to me?"
    
    # 執行檢索
    top_results = retrieve_top_by_exhibition(db, query)
    
    # 打印檢索結果
    print("檢索結果：")
    for result in top_results:
        doc = result["doc"]
        score = result["score"]
        print(f"展覽名稱: {doc.metadata['title']}")
        print(f"段落內容: {doc.page_content}")
        print(f"相似度分數: {score}")
        print("-" * 50)
    
    # 初始化 LLM 並生成推薦結果
    document_chain = initialize_llm()
    llm_result = generate_llm_recommendations(document_chain, query, [r["doc"] for r in top_results]) #???
    print(f"問題{query}")

    # 打印並保存 LLM 推薦結果
    print("LLM Answer: ", llm_result)
    print("LLM Answer type: ", type(llm_result))
    return save_llm_results_as_dict(llm_result, [r["doc"] for r in top_results])


if __name__ == "__main__":
    # 後台初始化
    # db = backend_initialize() #初始化執行時間約為 8 秒
    # 查看db的佔記憶體的大小
    # db_size = sys.getsizeof(db)
    # print(f"向量索引佔用的記憶體大小: {db_size} bytes")
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=config["AzureOpenAI"]["DEPLOYMENT_NAME_EMBEDDING_LARGE"],
        openai_api_version=config["AzureOpenAI"]["VERSION"],
        api_key=config["AzureOpenAI"]["KEY"],
        azure_endpoint=config["AzureOpenAI"]["ENDPOINT"],
    )
    # 使用者互動
    prebuilt_faiss=FAISS.load_local("exhibition_db_faiss", 
                                    embeddings, 
                                    "index",
                                    allow_dangerous_deserialization=True)
    print({prebuilt_faiss})
    
    # 使用者互動
    user_interaction(prebuilt_faiss, latest_love_exhib_id='') #請改user_input，執行時間約為 5 秒
    


