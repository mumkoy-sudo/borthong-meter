import streamlit as st
import pandas as pd
from google.cloud import vision
from google.oauth2 import service_account
import gspread
import datetime
import re
import json

# --- 1. การตั้งค่าและเชื่อมต่อ (Configuration) ---
KEY_FILE = 'credentials.json' 
SHEET_NAME = 'Bothong_Meter_Data' 

st.set_page_config(page_title="บ่อทอง เรสซิเด้นท์", page_icon="📝")

# ฟังก์ชันเชื่อมต่อ Google Cloud & Sheet
@st.cache_resource
def init_connection():
    try:
        my_scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/cloud-platform"
        ]
        
        creds = None
        # เช็คว่ารันบน Cloud (Secrets) หรือ คอมตัวเอง (File)
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

# --- 2. ฟังก์ชันช่วยงาน ---
def get_text_from_image(image_bytes):
    if vision_client is None: return ""
    image = vision.Image(content=image_bytes)
    response = vision_client.text_detection(image=image)
    texts = response.text_annotations
    if texts: return texts[0].description
    return ""

def extract_numbers(text):
    numbers = re.findall(r'\d+', text)
    suggested_room = ""
    suggested_meter = 0
    for num in numbers:
        if len(num) == 3 and int(num) <= 360: 
            suggested_room = num
        elif len(num) >= 4: 
            suggested_meter = int(num)
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
    
    # --- ส่วนที่แก้ไข: เพิ่ม Tabs ให้เลือก ---
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
            
    # --- จบส่วนแก้ไข ---

    ai_room = ""
    ai_reading = 0

    if img_file:
        bytes_data = img_file.getvalue()
        with st.spinner('🤖 AI กำลังอ่านค่า...'):
            raw_text = get_text_from_image(bytes_data)
            ai_room, ai_reading = extract_numbers(raw_text)
        st.success("อ่านค่าจากรูปเสร็จสิ้น!")

    with st.form("meter_form"):
        c1, c2 = st.columns(2)
        room_number = c1.text_input("เลขห้อง", value=ai_room)
        current_reading = c2.number_input("เลขมิเตอร์", min_value=0, value=ai_reading)
        
        prev = 0
        usage = 0
        if room_number:
            prev = get_last_reading(room_number, meter_type)
            st.info(f"ครั้งก่อน: **{prev}**")
            if current_reading >= prev: usage = current_reading - prev
            else: usage = current_reading 
            st.metric("หน่วยที่ใช้", usage)

        if st.form_submit_button("💾 บันทึก"):
            save_data(room_number, meter_type, prev, current_reading, usage)
            st.success(f"บันทึกห้อง {room_number} เรียบร้อย!")
            st.balloons()