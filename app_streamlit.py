import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F


MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
ASPECTS = ["สินค้า", "ราคา", "บริการ", "ความหลากหลาย"]
ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}

MAX_LEN = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



class MultiHeadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.heads = nn.ModuleList([nn.Linear(hidden, 3) for _ in range(4)])

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]
        cls = self.dropout(cls)
        logits = torch.stack([head(cls) for head in self.heads], dim=1)  # (B,4,3)
        return logits



@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = MultiHeadModel().to(DEVICE)
    model.load_state_dict(torch.load("absa_model.pt", map_location=DEVICE))
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()



def get_color(label):
    if label == "Positive":
        return "#2ecc71"  # green
    if label == "Negative":
        return "#e74c3c"  # red
    return "#f1c40f"      # yellow


def get_emoji(label):
    if label == "Positive":
        return "🟢"
    if label == "Negative":
        return "🔴"
    return "🟡"


def label_th(label_en: str):
    if label_en == "Positive":
        return "บวก"
    if label_en == "Negative":
        return "ลบ"
    return "กลาง"


def stars_from(pred_id: int, conf01: float) -> int:
    """
    ดาว 1–5:
    - Positive: ดาวสูงตามความมั่นใจ
    - Neutral: ดาวกลาง ๆ
    - Negative: ดาวต่ำตามความมั่นใจ (ยิ่งมั่นใจว่าแย่ ดาวยิ่งน้อย)
    """

    base = int(round(1 + conf01 * 4))
    base = max(1, min(5, base))

    if pred_id == 2:       # pos
        return base
    elif pred_id == 1:     # neu

        return int(round(2 + conf01 * 1.5))  # 2..4
    else:                  # neg
        return max(1, 6 - base)              # conf สูง -> ดาวน้อย


def render_stars(n: int) -> str:
    n = max(1, min(5, n))
    return "⭐" * n + "☆" * (5 - n)



st.set_page_config(page_title="ABSA รีวิวร้านมือถือ", layout="wide")

st.title("📱 วิเคราะห์รีวิวร้านมือถือ/เคสมือถือ")
st.markdown("ระบบวิเคราะห์ความคิดเห็นแยกตามหัวข้อ (Aspect-Based Sentiment Analysis)")

# init session state
if "review_text" not in st.session_state:
    st.session_state.review_text = ""

# input section
st.markdown("### ✍️ ใส่รีวิว")

b1, b2, b3, b4 = st.columns(4)
if b1.button("✨ รีวิวดี", key="ex_good"):
    st.session_state.review_text = "เคสสวยมาก วัสดุดี ราคาโอเค พนักงานแนะนำดี ของมีให้เลือกเยอะ"
    st.rerun()
if b2.button("⚠️ รีวิวกลาง", key="ex_neu"):
    st.session_state.review_text = "เคสโอเคตามราคา บริการปกติ ของมีพอประมาณ"
    st.rerun()
if b3.button("🔥 รีวิวแย่", key="ex_bad"):
    st.session_state.review_text = "วัสดุไม่ค่อยดี ราคาแพง พนักงานไม่ค่อยสนใจ รุ่นที่หาไม่มี"
    st.rerun()
if b4.button("🧹 ล้าง", key="ex_clear"):
    st.session_state.review_text = ""
    st.rerun()

review = st.text_area("พิมพ์รีวิวของคุณที่นี่", height=120, key="review_text")

analyze_clicked = st.button("🔍 วิเคราะห์", key="analyze_btn")

if analyze_clicked:
    text = st.session_state.review_text.strip()
    if text == "":
        st.warning("กรุณาพิมพ์รีวิวก่อนกดวิเคราะห์")
    else:
        enc = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)      # (1,4,3)
            probs = F.softmax(logits, dim=-1)              # (1,4,3)


        pred_ids, conf_list = [], []
        for i in range(4):
            pred = torch.argmax(probs[0, i]).item()
            conf = probs[0, i][pred].item()  # 0..1
            pred_ids.append(pred)
            conf_list.append(conf)

        pos_cnt = sum(1 for x in pred_ids if x == 2)
        neg_cnt = sum(1 for x in pred_ids if x == 0)
        neu_cnt = sum(1 for x in pred_ids if x == 1)


        scores = []
        for pred, conf in zip(pred_ids, conf_list):
            if pred == 2:
                s = 1.0
            elif pred == 0:
                s = -1.0
            else:
                s = 0.0
            scores.append(s * conf)

        overall = sum(scores) / max(1e-9, sum(conf_list))  # -1..1
        overall_0_100 = int(round((overall + 1) * 50))      # 0..100

        st.markdown("## 📊 ผลการวิเคราะห์")

        st.markdown(
            f"""
            <div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:10px;">
                <div style="padding:6px 12px; border-radius:999px; background:#173b22; border:1px solid #2ecc71;">🟢 บวก: {pos_cnt}</div>
                <div style="padding:6px 12px; border-radius:999px; background:#3a330f; border:1px solid #f1c40f;">🟡 กลาง: {neu_cnt}</div>
                <div style="padding:6px 12px; border-radius:999px; background:#3a1513; border:1px solid #e74c3c;">🔴 ลบ: {neg_cnt}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # overall banner
        if overall_0_100 >= 70:
            st.success(f"✅ คะแนนภาพรวม: {overall_0_100}/100 (โดยรวมค่อนข้างดี)")
        elif overall_0_100 <= 30:
            st.error(f"⛔ คะแนนภาพรวม: {overall_0_100}/100 (โดยรวมค่อนข้างแย่)")
        else:
            st.warning(f"⚠️ คะแนนภาพรวม: {overall_0_100}/100 (โดยรวมคละกัน)")

        st.progress(overall_0_100 / 100)

        # cards
        col1, col2 = st.columns(2)

        for i, aspect in enumerate(ASPECTS):
            pred = pred_ids[i]
            conf01 = conf_list[i]
            confidence = conf01 * 100

            label_en = ID2LABEL[pred]
            label_show = label_th(label_en)
            color = get_color(label_en)
            emoji = get_emoji(label_en)

            stars = stars_from(pred, conf01)
            stars_text = render_stars(stars)

            card_html = f"""
            <div style="
                background-color:#1e1e1e;
                padding:18px;
                border-radius:14px;
                margin-bottom:14px;
                border-left:8px solid {color};
            ">
                <div style="display:flex; align-items:center; justify-content:space-between;">
                    <h4 style="margin:0;">{emoji} {aspect}</h4>
                    <div style="font-size:18px;">{stars_text}</div>
                </div>
                <div style="margin-top:8px; font-size:20px; font-weight:700; color:{color};">
                    {label_show}
                </div>
                <div style="opacity:0.9; margin-top:4px;">
                    ความมั่นใจ: {confidence:.1f}%
                </div>
            </div>
            """
            target_col = col1 if i % 2 == 0 else col2
            target_col.markdown(card_html, unsafe_allow_html=True)
            target_col.progress(min(max(conf01, 0.0), 1.0))

st.markdown("---")
st.caption("Developed with WangchanBERTa + PyTorch + Streamlit")