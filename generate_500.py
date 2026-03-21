import random
import pandas as pd

random.seed(42)

ASPECTS = ["product", "price", "service", "variety"]
LABELS = ["neg", "neu", "pos"]

PRODUCT_POS = ["เคสสวยมาก", "วัสดุดี", "งานประกอบแน่น", "กันกระแทกดี", "สีตรงปก", "คุณภาพดีเกินคาด"]
PRODUCT_NEG = ["วัสดุไม่ค่อยดี", "งานประกอบหลวม", "สีไม่ตรงปก", "คุณภาพต่ำ", "เป็นรอยง่าย", "กันกระแทกไม่ดี"]
PRODUCT_NEU = ["เคสธรรมดา", "คุณภาพโอเค", "ใช้งานได้ทั่วไป", "ดูมาตรฐาน", "ไม่ได้เด่นมาก", "พอใช้ได้"]

PRICE_POS = ["ราคาคุ้มมาก", "ราคาโอเค", "มีโปรดี", "ลดราคาให้", "คุ้มค่ากับราคา", "ราคาไม่แพง"]
PRICE_NEG = ["ราคาแพงไปหน่อย", "แพงเกินคุณภาพ", "คิดแพง", "ราคาสูง", "ไม่ค่อยคุ้ม", "แพงกว่าที่คิด"]
PRICE_NEU = ["ราคาเฉยๆ", "ราคาใกล้เคียงร้านอื่น", "แล้วแต่คนมอง", "ราคากลางๆ", "ไม่ได้ถูกไม่ได้แพง", "ราคามาตรฐาน"]

SERVICE_POS = ["พนักงานแนะนำดี", "บริการดีมาก", "พูดจาสุภาพ", "ช่วยเลือกได้ดี", "ใส่ใจลูกค้า", "ตอบคำถามละเอียด"]
SERVICE_NEG = ["พนักงานไม่ค่อยสนใจ", "พูดไม่ดี", "บริการแย่", "ตอบช้า", "ไม่ค่อยช่วย", "หน้าตาไม่รับแขก"]
SERVICE_NEU = ["บริการปกติ", "โอเคตามมาตรฐาน", "ไม่ได้มีอะไรพิเศษ", "พนักงานทั่วไป", "พอใช้ได้", "เฉยๆ"]

VARIETY_POS = ["ของมีหลายรุ่น", "มีของครบ", "สีครบ", "รุ่นเยอะมาก", "ของให้เลือกเยอะ", "มีเคสหลายแบบ"]
VARIETY_NEG = ["ของมีน้อย", "รุ่นที่หาไม่มี", "สีที่อยากได้หมด", "ของไม่ค่อยครบ", "ตัวเลือกน้อย", "ของขาดบ่อย"]
VARIETY_NEU = ["มีของพอสมควร", "แล้วแต่ช่วง", "บางรุ่นมีบางรุ่นไม่มี", "ของมีบ้าง", "ไม่แน่ใจเรื่องสต็อก", "พอมีให้เลือก"]

def pick_phrase(label, pos_list, neu_list, neg_list):
    if label == "pos":
        return random.choice(pos_list)
    if label == "neg":
        return random.choice(neg_list)
    return random.choice(neu_list)

def make_row():

    weights = {"pos": 0.35, "neu": 0.30, "neg": 0.35}
    labels = random.choices(list(weights.keys()), weights=list(weights.values()), k=4)
    y = dict(zip(ASPECTS, labels))

    parts = [
        pick_phrase(y["product"], PRODUCT_POS, PRODUCT_NEU, PRODUCT_NEG),
        pick_phrase(y["price"], PRICE_POS, PRICE_NEU, PRICE_NEG),
        pick_phrase(y["service"], SERVICE_POS, SERVICE_NEU, SERVICE_NEG),
        pick_phrase(y["variety"], VARIETY_POS, VARIETY_NEU, VARIETY_NEG),
    ]

    for i, a in enumerate(ASPECTS):
        if random.random() < 0.25: 
            parts[i] = None
            y[a] = "neu"

    text = " ".join([p for p in parts if p is not None]).strip()
    if not text:
 
        text = random.choice(PRODUCT_NEU)
        y["product"] = "neu"

    return {"text": text, **y}

rows = [make_row() for _ in range(500)]
df = pd.DataFrame(rows)

# บันทึกไฟล์
out_path = "data/absa_mobile_shop.csv"
df.to_csv(out_path, index=False, encoding="utf-8-sig")

print("✅ Generated:", out_path)
print("Total rows:", len(df))
print(df.head(5))