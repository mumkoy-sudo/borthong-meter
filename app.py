import streamlit as st
import pandas as pd
from google.cloud import vision
from google.oauth2 import service_account
import gspread
import datetime
import re
import json

# --- 1. การตั้งค่าและเชื่อมต่อ ---
KEY_FILE = 'credentials.json' 
SHEET_NAME = 'Bothong_Meter_Data' 

st.set_page_config(page_title="บ่อทอง เรสซิเด้นท์", page_icon="📝")

# เชื่อมต่อ Google Cloud & Sheet
@st.cache_resource
def init_connection():
    try:
        my_scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/cloud-platform"
        ]
        
        creds = None
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

# *** บรรทัดนี้สำคัญมาก (ที่หายไปคราวที่แล้ว) ***
vision_client, sh = init_connection()

# --- 2. สูตรแกะตัวเลข V.3 (ไม่ต้องใช้ cv2) ---
def get_text_from_image(image_bytes):
    if vision_client is None: return ""
    image = vision.Image(content=image_bytes)
    # ใช้โหมด DOCUMENT_TEXT_DETECTION เพื่อให้อ่านตัวเลขที่ชิดกันได้ดีขึ้น
    response = vision_client.document_text_detection(image=image)
    if response.full_text_annotation:
        return response.full_text_annotation.text
    return ""

def extract_numbers(text, m_type):
    # 1. ล้างค่า Text ให้สะอาด
    text_clean = text.replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1')
    text_merged = text_clean.replace(" ", "") 
    
    # ดึงชุดตัวเลขทั้งหมดออกมา
    numbers_raw = re.findall(r'\d+', text_clean)
    numbers_merged = re.findall(r'\d+', text_merged)
    
    all_candidates = set(numbers_raw + numbers_merged)
    
    suggested_room = ""
    suggested_meter = 0
    meter_candidates = []
    
    # Blacklist เลขขยะ
    ignore_list = [
        220, 50, 15, 45, 100, 400, 
        2023, 2024, 2025, 2552, 2336, 
        2124, 65057, 6505, 
        1, 2, 33 
    ]

    # ขั้นตอนที่ 1: หาเลขห้อง
    for num_str in all_candidates:
        if len(num_str) == 4 and num_str.startswith("10"):
            suggested_room = num_str
            break 
    
    if not suggested_room:
        for num_str in all_candidates:
            if len(num_str) == 3 and int(num_str) < 500:
                suggested_room = num_str
                break

    # ขั้นตอนที่ 2: หาเลขมิเตอร์
    for num_str in all_candidates:
        if num_str == suggested_room: continue
        if len(num_str) < 3: continue
        val = int(num_str)
        if val in ignore_list: continue
        if len(num_str) > 6: continue # ตัด Serial Number ยาวๆ ทิ้ง
        meter_candidates.append(val)

    # ขั้นตอนที่ 3: เลือกเลขที่ดีที่สุด
    if meter_candidates:
        if m_type == 'ไฟฟ้า':
            # ไฟฟ้า: เน้น 5 หลัก (10000-99999)
            priority_candidates = [x for x in meter_candidates if 10000 <= x <= 99999]
            if priority_candidates:
                suggested_meter = max(priority_candidates)
            else:
                suggested_meter = max(meter_candidates)
        else: 
            # น้ำประปา: เน้น 4 หลัก
            priority_candidates = [x for x in meter_candidates if 1000 <= x <= 9999]
            if priority_candidates:
                suggested_meter = max(priority_candidates)
            else:
                suggested_meter = max(meter_candidates)

    return suggested_room, suggested_meter

def get_last_reading(room, meter_type):
    if sh is None: return 0
    try:
        worksheet = sh.worksheet("Latest_Status")
        records = worksheet.get_all_records()
        df = pd.DataFrame(records)
        df['Room'] = df['Room'].astype(str)
        room = str(room)
        row = df[df['Room'] == room]
        if not row.empty:
            col_name = 'Last_Water' if meter_type == 'น้ำประปา' else 'Last_Elec'
            return int(row.iloc[0][col_name])
        return 0 
    except: return 0

def save_data(room, m_type, prev, curr, usage):
    if sh is None: return
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_sheet = sh.worksheet("Logs")
    log_sheet.append_row([timestamp, room, m_type, prev, curr, usage])
    status_sheet = sh.worksheet("Latest_Status")
    cell = status_sheet.find(str(room))
    if cell:
        col_index = 2 if m_type == 'น้ำประปา' else 3
        status_sheet.update_cell(cell.row, col_index, curr)
    else:
        new_water = curr if m_type == 'น้ำประปา' else 0
        new_elec = curr if m_type == 'ไฟฟ้า' else 0
        status_sheet.append_row([room, new_water, new_elec])

# --- 3. หน้าจอแอป (UI) ---
st.title("💧⚡ บ่อทอง เรสซิเด้นท์")

if sh is None:
    st.warning("⚠️ กำลังเชื่อมต่อระบบ...")
else:
    meter_type = st.radio("เลือกประเภท:", ["น้ำประปา", "ไฟฟ้า"], horizontal=True)
    
    tab1, tab2 = st.tabs(["📸 ถ่ายรูป", "📂 อัปโหลดรูป"])
    img_file = None
    
    with tab1:
        camera_img = st.camera_input(f"ถ่ายรูปมิเตอร์{meter_type}")
        if camera_img: img_file = camera_img

    with tab2:
        uploaded_img = st.file_uploader(f"เลือกรูปมิเตอร์{meter_type} จากเครื่อง", type=['jpg', 'png', 'jpeg'])
        if uploaded_img: 
            st.image(uploaded_img, caption="รูปที่เลือก", width=300)
            img_file = uploaded_img

    ai_room = ""
    ai_reading = 0
    raw_text_debug = ""

    if img_file:
        bytes_data = img_file.getvalue()
        with st.spinner('🤖 AI กำลังแกะรอยตัวเลข (สูตร V.3)...'):
            raw_text_debug = get_text_from_image(bytes_data)
            ai_room, ai_reading = extract_numbers(raw_text_debug, meter_type)
        
        st.success("อ่านค่าเรียบร้อย! (กรุณาตรวจสอบก่อนบันทึก)")
        with st.expander("🔍 ดูสิ่งที่ AI อ่านได้ทั้งหมด"):
            st.text(raw_text_debug)

    with st.form("meter_form"):
        st.caption("👇 ตรวจสอบและแก้ไขตัวเลขได้ที่นี่")
        c1, c2 = st.columns(2)
        room_number = c1.text_input("เลขห้อง (ป้ายเหลือง)", value=ai_room)
        
        current_reading = c2.number_input(
            "เลขมิเตอร์", 
            min_value=0, 
            value=ai_reading, 
            help="ถ้าเลขกำลังหมุน ให้ปัดเศษลง (ใช้ค่าน้อยกว่า)"
        )
        
        prev = 0
        usage = 0
        
        if room_number:
            prev = get_last_reading(room_number, meter_type)
            st.info(f"ครั้งก่อน: **{prev}**")
            
            if current_reading >= prev: 
                usage = current_reading - prev
            else: 
                st.warning("⚠️ เลขปัจจุบันน้อยกว่าครั้งก่อน (จดผิดหรือเปล่าครับ?)")
                usage = current_reading 
            
            st.metric("หน่วยที่ใช้", usage)

        if st.form_submit_button("💾 บันทึกข้อมูล"):
            if room_number and current_reading > 0:
                save_data(room_number, meter_type, prev, current_reading, usage)
                st.success(f"บันทึกห้อง {room_number} เรียบร้อย!")
                st.balloons()
            else:
                st.error("กรุณากรอกข้อมูลให้ครบ (เลขห้อง และ เลขมิเตอร์)")
