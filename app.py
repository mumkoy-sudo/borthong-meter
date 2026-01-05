import streamlit as st
import pandas as pd
import datetime
import re

import numpy as np
import cv2
from PIL import Image

from google.cloud import vision
from google.oauth2 import service_account
import gspread

# =========================================================
# 0) CONFIG
# =========================================================
KEY_FILE = "credentials.json"             # ใช้ตอนรัน local
SHEET_NAME = "Bothong_Meter_Data"
APP_TITLE = "💧⚡ บ่อทอง เรสซิเด้นท์ (Meter OCR Pro)"

# กำหนด “เพดานการใช้งาน” เพื่อจับความผิดปกติ (ปรับได้ตามจริง)
DEFAULT_MAX_JUMP_WATER = 200     # หน่วย/รอบ
DEFAULT_MAX_JUMP_ELEC  = 5000    # หน่วย/รอบ

st.set_page_config(page_title="บ่อทอง เรสซิเด้นท์", page_icon="📝", layout="centered")


# =========================================================
# 1) CONNECTIONS (Vision + Google Sheet)
# =========================================================
@st.cache_resource
def init_connection():
    try:
        my_scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/cloud-platform",
        ]

        if "gcp_service_account" in st.secrets:
            creds = service_account.Credentials.from_service_account_info(
                st.secrets["gcp_service_account"], scopes=my_scopes
            )
        else:
            creds = service_account.Credentials.from_service_account_file(
                KEY_FILE, scopes=my_scopes
            )

        vision_client = vision.ImageAnnotatorClient(credentials=creds)
        gc = gspread.authorize(creds)
        sh = gc.open(SHEET_NAME)
        return vision_client, sh
    except Exception as e:
        st.error(f"เชื่อมต่อไม่สำเร็จ: {e}")
        return None, None


vision_client, sh = init_connection()


# =========================================================
# 2) IMAGE UTILS (ROI + PREPROCESS + AUTO DETECT)
# =========================================================
def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv_img: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def clamp_bbox(x1, y1, x2, y2, w, h):
    x1 = int(max(0, min(x1, w-1)))
    x2 = int(max(1, min(x2, w)))
    y1 = int(max(0, min(y1, h-1)))
    y2 = int(max(1, min(y2, h)))
    if x2 <= x1: x2 = min(w, x1 + 1)
    if y2 <= y1: y2 = min(h, y1 + 1)
    return x1, y1, x2, y2

def crop_cv(cv_img: np.ndarray, bbox):
    x1, y1, x2, y2 = bbox
    return cv_img[y1:y2, x1:x2].copy()

def preprocess_roi_for_ocr(cv_roi: np.ndarray, meter_type: str) -> np.ndarray:
    """
    เพิ่มความคม/คอนทราสต์ + ลดเงา/สะท้อน เพื่อช่วย OCR
    """
    img = cv_roi.copy()

    # ขยายก่อนเพื่อให้ตัวเลขใหญ่ขึ้น
    img = cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ลด noise
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # เพิ่มคอนทราสต์ด้วย CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # ทำ threshold
    if meter_type == "ไฟฟ้า":
        # ไฟฟ้ามักพื้นหลังสว่าง ตัวเลขดำ
        th = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 7
        )
    else:
        # น้ำกลมมีเงา/โค้ง ใช้ threshold ที่ทนกว่า
        th = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            41, 10
        )

    # ปรับ morphology เล็กน้อย
    kernel = np.ones((2, 2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

def find_yellow_label_bbox(cv_img: np.ndarray):
    """
    หา bbox ของป้ายเหลือง (เลขห้อง) แบบอัตโนมัติจากสี
    คืนค่า bbox (x1,y1,x2,y2) หรือ None
    """
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    # ช่วงสีเหลือง (ปรับได้)
    lower = np.array([15, 80, 80])
    upper = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # ช่วยลด noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # เลือกก้อนที่ใหญ่สุด
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 300:  # กันจุดเล็ก ๆ
        return None

    x, y, w, h = cv2.boundingRect(cnt)
    # ขยายกรอบเล็กน้อย
    pad = 8
    H, W = cv_img.shape[:2]
    return clamp_bbox(x-pad, y-pad, x+w+pad, y+h+pad, W, H)

def auto_detect_display_bbox(cv_img: np.ndarray, meter_type: str):
    """
    พยายามเดา “หน้าต่างตัวเลขมิเตอร์” อัตโนมัติ
    ใช้เป็นค่าเริ่มต้นให้ผู้ใช้ปรับต่อได้ (ไม่พึ่ง 100%)
    """
    H, W = cv_img.shape[:2]
    work = cv_img.copy()

    # จำกัดค้นหาในโซนกลาง-บน เพื่อตัดสติ๊กเกอร์ล่าง (serial/barcode)
    if meter_type == "ไฟฟ้า":
        y_top, y_bot = 0, int(H * 0.75)
    else:
        y_top, y_bot = 0, H  # น้ำกลมบางทีอยู่กลางล่างได้
    roi = work[y_top:y_bot, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 60, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < (W * H) * 0.01:
            continue

        ar = w / max(1, h)  # aspect ratio
        # หน้าต่างเลขมักเป็นสี่เหลี่ยมผืนผ้า
        if ar < 1.3 or ar > 6.5:
            continue

        # ตำแหน่ง: ใกล้กึ่งกลางแนวนอนมักดี
        cx = x + w / 2
        center_score = 1.0 - abs(cx - (W / 2)) / (W / 2)

        # สำหรับไฟฟ้า: เน้นโซนบน
        if meter_type == "ไฟฟ้า":
            pos_penalty = 1.0 - (y / max(1, (y_bot - y_top)))  # ยิ่งบนยิ่งดี
        else:
            pos_penalty = 0.7  # น้ำไม่ strict

        score = (area / (W * H)) * 3.0 + center_score + pos_penalty
        if score > best_score:
            best_score = score
            best = (x, y, x + w, y + h)

    if best is None:
        # fallback: กลางภาพ
        x1 = int(W * 0.20)
        x2 = int(W * 0.80)
        y1 = int(H * 0.35)
        y2 = int(H * 0.60)
        return (x1, y1, x2, y2)

    x1, y1, x2, y2 = best
    # แปลงกลับจาก roi coords
    y1 += y_top
    y2 += y_top

    # ขยายกรอบเล็กน้อย
    pad_x = int((x2 - x1) * 0.10)
    pad_y = int((y2 - y1) * 0.25)
    return clamp_bbox(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, W, H)

def encode_cv_to_bytes(cv_img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise ValueError("encode image failed")
    return buf.tobytes()


# =========================================================
# 3) OCR (GOOGLE VISION) + PARSING
# =========================================================
def vision_text(image_bytes: bytes) -> str:
    if vision_client is None:
        return ""
    img = vision.Image(content=image_bytes)
    # ใช้ text_detection (เพียงพอเมื่อเรา crop เฉพาะหน้าต่างเลขแล้ว)
    resp = vision_client.text_detection(image=img)
    if resp.text_annotations:
        return resp.text_annotations[0].description
    return ""

def normalize_ocr_text(s: str) -> str:
    # แก้ตัวสลับกันบ่อย
    return (
        s.replace("O", "0").replace("o", "0")
         .replace("I", "1").replace("l", "1").replace("|", "1")
         .replace("S", "5")  # บางภาพ
    )

def extract_digit_candidates(text: str, meter_type: str):
    """
    คืน list[int] ของ candidate เลขมิเตอร์ (ยังไม่เลือกตัวสุดท้าย)
    """
    t = normalize_ocr_text(text)
    t2 = t.replace(" ", "")  # รวมกรณี "1 4 6 1 3"

    raw = re.findall(r"\d+", t)
    merged = re.findall(r"\d+", t2)
    cands = set(raw + merged)

    nums = []
    for s in cands:
        if len(s) < 3 or len(s) > 6:
            continue
        nums.append(int(s))

    # กรองช่วงคร่าว ๆ เพื่อกัน serial/ปี
    if meter_type == "ไฟฟ้า":
        nums = [x for x in nums if 1000 <= x <= 999999]
    else:
        nums = [x for x in nums if 0 <= x <= 999999]
    return sorted(set(nums))

def choose_best_candidate(cands, prev, meter_type: str, max_jump: int):
    """
    เลือกเลขที่ดีที่สุดโดยอิง:
    1) ความยาวที่เหมาะสม (ไฟ 5-6 หลัก, น้ำ 4 หลัก)
    2) ต้อง >= prev (ถ้า prev มี)
    3) ใช้งาน (diff) ไม่เกิน max_jump (ถ้า prev มี)
    """
    if not cands:
        return 0, []

    scored = []
    for x in cands:
        # length preference
        L = len(str(x))
        if meter_type == "ไฟฟ้า":
            len_score = 3 if L == 5 else (2 if L == 6 else 0)
        else:
            len_score = 3 if L == 4 else (1 if L == 3 or L == 5 else 0)

        if prev and x < prev:
            valid = 0
            jump_ok = 0
            diff = prev - x
        else:
            valid = 1
            diff = x - prev if prev else 0
            jump_ok = 1 if (prev == 0 or diff <= max_jump) else 0

        # ให้คะแนน: valid + jump_ok + len_score และกันค่าโดด
        score = valid * 5 + jump_ok * 3 + len_score
        # ถ้า prev มี ให้ชอบ diff ที่เล็กและไม่ผิดปกติ
        if prev:
            score += max(0, 3 - min(3, diff / max(1, max_jump) * 3))

        scored.append((score, x, diff))

    scored.sort(reverse=True, key=lambda z: (z[0], z[1]))
    best = scored[0][1]
    # ส่งกลับ scored (เพื่อโชว์)
    return best, scored


def ocr_room_from_yellow_label(cv_img: np.ndarray) -> str:
    bbox = find_yellow_label_bbox(cv_img)
    if bbox is None:
        return ""
    roi = crop_cv(cv_img, bbox)
    proc = preprocess_roi_for_ocr(roi, meter_type="ไฟฟ้า")  # ใช้ threshold แบบไฟสำหรับเลขดำบนเหลือง
    txt = vision_text(encode_cv_to_bytes(proc))
    t = normalize_ocr_text(txt)
    # ห้องของคุณเป็น 4 หลักขึ้นต้น 10 เช่น 1018, 1020
    m = re.search(r"\b(10\d{2})\b", t.replace(" ", ""))
    if m:
        return m.group(1)
    # fallback 3-4 digits
    m2 = re.search(r"\b(\d{3,4})\b", t.replace(" ", ""))
    return m2.group(1) if m2 else ""


def ocr_meter_from_roi(cv_roi: np.ndarray, meter_type: str, prev: int, max_jump: int):
    proc = preprocess_roi_for_ocr(cv_roi, meter_type=meter_type)
    txt = vision_text(encode_cv_to_bytes(proc))
    cands = extract_digit_candidates(txt, meter_type=meter_type)
    best, scored = choose_best_candidate(cands, prev, meter_type=meter_type, max_jump=max_jump)
    return best, cands, scored, txt, proc


# =========================================================
# 4) SHEET OPS
# =========================================================
def get_last_reading(room, meter_type):
    if sh is None:
        return 0
    try:
        worksheet = sh.worksheet("Latest_Status")
        records = worksheet.get_all_records()
        df = pd.DataFrame(records)
        df["Room"] = df["Room"].astype(str)
        room = str(room)
        row = df[df["Room"] == room]
        if not row.empty:
            col_name = "Last_Water" if meter_type == "น้ำประปา" else "Last_Elec"
            v = row.iloc[0][col_name]
            return int(v) if str(v).strip() != "" else 0
        return 0
    except Exception:
        return 0

def save_data(room, m_type, prev, curr, usage):
    if sh is None:
        return
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_sheet = sh.worksheet("Logs")
    log_sheet.append_row([timestamp, room, m_type, prev, curr, usage])

    status_sheet = sh.worksheet("Latest_Status")
    cell = status_sheet.find(str(room))
    if cell:
        col_index = 2 if m_type == "น้ำประปา" else 3
        status_sheet.update_cell(cell.row, col_index, curr)
    else:
        new_water = curr if m_type == "น้ำประปา" else 0
        new_elec = curr if m_type == "ไฟฟ้า" else 0
        status_sheet.append_row([room, new_water, new_elec])


# =========================================================
# 5) UI
# =========================================================
st.title(APP_TITLE)

with st.expander("✅ คู่มือถ่ายภาพให้ AI อ่านแม่น (สำคัญมาก)"):
    st.markdown(
        """
**เป้าหมาย:** ให้ “หน้าต่างตัวเลข” ชัดและใหญ่ที่สุด และลดสิ่งรบกวน (Serial/บาร์โค้ด)

1) **ซูมเข้าหน้าต่างตัวเลข** ให้กินพื้นที่รูป ~40–60%  
2) **ถือกล้องให้ตรงฉาก** (ไม่เอียงซ้าย/ขวา/บน/ล่าง)  
3) **เลี่ยงแสงสะท้อน**: อย่าใช้แฟลชตรง ๆ, ขยับมุมเล็กน้อยให้ไม่มีเงาขาวทับเลข  
4) **แตะโฟกัสที่ตัวเลข** ก่อนถ่าย (ไม่ให้เบลอ)  
5) ถ้าแสงน้อย: เปิดไฟ/ใช้แสงด้านข้าง + เปิด HDR  
6) **อย่าให้สติ๊กเกอร์/บาร์โค้ดล่างเด่นกว่าเลขหน้าต่าง** (เพราะ OCR จะไปอ่าน Serial แทน)

> ทริค: ถ้าถ่ายไกล ให้ครอปก่อนอัปโหลด หรือใช้ตัวครอปในแอป (ด้านล่าง)
        """
    )

if sh is None:
    st.warning("⚠️ กำลังเชื่อมต่อระบบ...")
    st.stop()

meter_type = st.radio("เลือกประเภท:", ["น้ำประปา", "ไฟฟ้า"], horizontal=True)
max_jump = st.number_input(
    "เพดานหน่วยที่ยอมรับได้ต่อรอบ (ใช้จับอ่านผิด/เลขโดด)",
    min_value=1,
    value=DEFAULT_MAX_JUMP_WATER if meter_type == "น้ำประปา" else DEFAULT_MAX_JUMP_ELEC,
    step=10,
)

tab1, tab2 = st.tabs(["📸 ถ่ายรูป", "📂 อัปโหลดรูป"])
img_file = None

with tab1:
    camera_img = st.camera_input(f"ถ่ายรูปมิเตอร์{meter_type}")
    if camera_img:
        img_file = camera_img

with tab2:
    uploaded_img = st.file_uploader(
        f"เลือกรูปมิเตอร์{meter_type} จากเครื่อง", type=["jpg", "png", "jpeg"]
    )
    if uploaded_img:
        img_file = uploaded_img

ai_room = ""
ai_reading = 0
debug = {}

if img_file:
    pil_img = Image.open(img_file).convert("RGB")
    cv_img = pil_to_cv2(pil_img)
    H, W = cv_img.shape[:2]

    st.image(pil_img, caption="รูปต้นฉบับ", use_container_width=True)

    # --- 5.1 Auto room from yellow label ---
    with st.spinner("🤖 กำลังหาเลขห้องจากป้ายเหลือง..."):
        ai_room = ocr_room_from_yellow_label(cv_img)

    # --- 5.2 Get prev reading if room known (for better candidate selection) ---
    prev_guess = get_last_reading(ai_room, meter_type) if ai_room else 0

    st.subheader("1) ครอป “หน้าต่างตัวเลขมิเตอร์” (วิธีที่แม่นที่สุด)")
    st.caption("ระบบจะเดาโซนคร่าว ๆ ให้ก่อน คุณปรับสไลเดอร์ให้ครอบเฉพาะ “หน้าต่างตัวเลข” เท่านั้น")

    # auto bbox suggestion
    auto_bbox = auto_detect_display_bbox(cv_img, meter_type=meter_type)
    ax1, ay1, ax2, ay2 = auto_bbox

    # sliders (normalize to 0..1 for usability)
    with st.container():
        colA, colB = st.columns(2)
        with colA:
            x1p = st.slider("ซ้าย (x1)", 0.0, 1.0, float(ax1 / W), 0.01)
            x2p = st.slider("ขวา (x2)", 0.0, 1.0, float(ax2 / W), 0.01)
        with colB:
            y1p = st.slider("บน (y1)", 0.0, 1.0, float(ay1 / H), 0.01)
            y2p = st.slider("ล่าง (y2)", 0.0, 1.0, float(ay2 / H), 0.01)

    x1 = int(min(x1p, x2p) * W)
    x2 = int(max(x1p, x2p) * W)
    y1 = int(min(y1p, y2p) * H)
    y2 = int(max(y1p, y2p) * H)
    x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, W, H)
    roi_bbox = (x1, y1, x2, y2)

    cv_roi = crop_cv(cv_img, roi_bbox)
    st.image(cv2_to_pil(cv_roi), caption="ROI (ควรเห็นเฉพาะหน้าต่างตัวเลข)", use_container_width=True)

    # OCR ROI
    with st.spinner("🤖 OCR เฉพาะ ROI..."):
        ai_reading, candidates, scored, raw_txt, proc_img = ocr_meter_from_roi(
            cv_roi, meter_type=meter_type, prev=prev_guess, max_jump=max_jump
        )

    st.success("อ่านค่าเรียบร้อย! กรุณาตรวจสอบก่อนบันทึก")

    # debug expander
    with st.expander("🔍 ดูรายละเอียด AI (ข้อความ/ภาพที่ผ่านการปรับ/ตัวเลือกเลข)"):
        st.markdown("**OCR Text (ROI):**")
        st.text(raw_txt)
        st.markdown("**ภาพที่ผ่านการปรับก่อน OCR:**")
        st.image(cv2_to_pil(proc_img), use_container_width=True)
        st.markdown("**Candidates:** " + (", ".join(map(str, candidates)) if candidates else "-"))
        if scored:
            st.markdown("**คะแนนการเลือก (สูงสุด = เลือกเป็นคำตอบ):**")
            st.write(pd.DataFrame(scored, columns=["score", "value", "diff_vs_prev"]))

# ---------------------------------------------------------
# FORM: CONFIRM + SAVE
# ---------------------------------------------------------
with st.form("meter_form"):
    st.caption("👇 ตรวจสอบและแก้ไขตัวเลขได้ที่นี่")
    c1, c2 = st.columns(2)

    room_number = c1.text_input("เลขห้อง (ป้ายเหลือง)", value=ai_room if img_file else "")
    prev = get_last_reading(room_number, meter_type) if room_number else 0
    c1.info(f"ครั้งก่อน: {prev}")

    # ให้ผู้ใช้เลือก candidate ได้ ถ้ามี
    if img_file and 'candidates' in locals() and candidates:
        pick_mode = c2.radio("เลขมิเตอร์ (เลือกวิธีกรอก)", ["ใช้ค่าที่ AI แนะนำ", "เลือกจากรายการ", "กรอกเอง"], horizontal=True)
        if pick_mode == "ใช้ค่าที่ AI แนะนำ":
            current_reading = c2.number_input("เลขมิเตอร์", min_value=0, value=int(ai_reading), step=1)
        elif pick_mode == "เลือกจากรายการ":
            chosen = c2.selectbox("เลือกเลขที่ถูกต้อง", options=candidates, index=candidates.index(ai_reading) if ai_reading in candidates else 0)
            current_reading = int(chosen)
        else:
            current_reading = c2.number_input("เลขมิเตอร์", min_value=0, value=0, step=1)
    else:
        current_reading = c2.number_input(
            "เลขมิเตอร์",
            min_value=0,
            value=int(ai_reading) if img_file else 0,
            step=1,
            help="ถ้าเลขกำลังหมุน ให้ปัดเศษลง (ใช้ค่าน้อยกว่า)",
        )

    usage = 0
    if room_number:
        if current_reading >= prev:
            usage = current_reading - prev
        else:
            st.warning("⚠️ เลขปัจจุบันน้อยกว่าครั้งก่อน (อาจจดผิด/อ่านผิด)")
            usage = current_reading

        # anomaly check
        if prev and (current_reading - prev) > max_jump:
            st.error(f"❗ หน่วยกระโดดมากผิดปกติ (>{max_jump}) แนะนำให้เช็ค ROI/ถ่ายใหม่/เลือกเลขจากรายการ")

        st.metric("หน่วยที่ใช้", usage)

    submitted = st.form_submit_button("💾 บันทึกข้อมูล")
    if submitted:
        if room_number and current_reading > 0:
            save_data(room_number, meter_type, prev, int(current_reading), int(usage))
            st.success(f"บันทึกห้อง {room_number} เรียบร้อย!")
            st.balloons()
        else:
            st.error("กรุณากรอกข้อมูลให้ครบ (เลขห้อง และ เลขมิเตอร์)")
