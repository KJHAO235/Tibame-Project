
from linebot.v3.messaging import (
    PostbackAction,
    QuickReply,
    QuickReplyItem
)

exhibitions = {
    # "Art": "藝術展",
    "Anime": "動漫展",
    # "Book": "書展",
    # "Car": "車展",
    "Technology": "科技展",
    # "Cultural and Creative": "文創展",
    # "Furniture": "家具展",
    "Food": "食品展",
    "Pet": "寵物展",
    "Wedding": "婚紗展",
    "Travel": "旅遊展",
    # "Design": "設計展",
    "Other": "其他展"
}

def make_quick_reply_exhi(fav_exhi: list = []):
    # 喜歡的展覽種類 quick reply

    quick_reply_exhi = QuickReply(items=[
            QuickReplyItem(
                action=PostbackAction(label=exhibitions[exhibition], data=exhibition),
            )
            for exhibition in exhibitions.keys()
            if exhibition not in fav_exhi
        ])
    # check = [exhibition for exhibition in exhibitions.keys() if exhibition not in fav_exhi]
    # print("這裡是 check：", check)
    return quick_reply_exhi


def recommend_like_or_not():
    # 推薦展覽是否喜歡 quick reply

    quick_reply = QuickReply(items=[
        QuickReplyItem(
            imageUrl="https://png.pngtree.com/png-clipart/20190516/original/pngtree-vector-like-icon-png-image_4013523.jpg",
            action=PostbackAction(label="喜歡這個推薦", data="like", )),
        QuickReplyItem(
            imageUrl="https://cdn-icons-png.flaticon.com/512/259/259436.png",
            action=PostbackAction(label="不喜歡", data="dislike"))
    ])

    return quick_reply