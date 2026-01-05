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

# --- 2. ฟังก์ชันช่วยงาน & สูตรแกะตัวเลข (แก้ใหม่ตามโจทย์) ---
def get_text_from_image(image_bytes):
    if vision_client is None: return ""
    image = vision.Image(content=image_bytes)
    response = vision_client.text_detection(image=image)
    texts = response.text_annotations
    if texts: return texts[0].description
    return ""

def extract_numbers(text):
    """
    สูตรแกะตัวเลขสำหรับ 'บ่อทอง เรสซิเด้นท์' โดยเฉพาะ
    1. ห้อง: ป้ายสีเหลือง 4 หลัก (เช่น 1018, 1020)
    2. มิเตอร์: ตัดเลขขยะ (220V, ปี 2024) ออก แล้วเลือกเลขที่เหลือที่น่าจะเป็นมิเตอร์ที่สุด
    """
    # 1. ล้างตัวอักษรที่ชอบอ่านผิด
    text = text.replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1')
    
    # 2. ดึงชุดตัวเลขทั้งหมดออกมา
    numbers = re.findall(r'\d+', text)
    
    suggested_room = ""
    suggested_meter = 0
    candidates = []
    
    # --- รายชื่อเลขขยะที่ต้องทิ้ง (Blacklist) ---
    # มาจากรูปมิเตอร์ Mitsubishi และ Sanwa ที่ส่งมา
    ignore_list = [
        220, 50, # 220V 50Hz
        15, 45, 100, 400, # Amp และ rev/kWh
        2023, 2024, 2025, # ปีผลิต
        2336, 2552, # เลข มอก. (2336-2552)
        33, # รุ่น MF-33E
        1, 2, # เลข Class 1 หรือ 2 เล็กๆ
    ]

    # ขั้นตอนที่ 1: คัดกรองตัวเลขเบื้องต้น
    for num_str in numbers:
        # ข้ามตัวเลขที่สั้นเกินไป (1-2 หลัก ไม่น่าใช่ทั้งห้องและมิเตอร์) 
        # ยกเว้นกรณีมิเตอร์น้ำหมุนรอบใหม่อาจจะเป็นเลขน้อยๆ ได้ แต่ส่วนใหญ่จะหลักร้อยขึ้น
        if len(num_str) < 3: 
            continue
            
        val = int(num_str)
        
        # ถ้าตัวเลขไปตรงกับเลขขยะ ให้ข้ามเลย
        if val in ignore_list:
            continue
            
        candidates.append(num_str)

    # ขั้นตอนที่ 2: ตามหา "เลขห้อง" (Priority สูงสุด)
    # จากรูป เลขห้องคือป้ายเหลือง 4 หลัก ขึ้นต้นด้วย '10'
    for c in candidates:
        if len(c) == 4 and c.startswith("10"):
            suggested_room = c
            candidates.remove(c) # เจอแล้วลบออกจากกองกลาง เพื่อไม่ให้สับสนกับเลขมิเตอร์
            break 
    
    # กรณีหาแบบ 4 หลักไม่เจอ (เผื่ออนาคต)
    if not suggested_room:
        for c in candidates:
            # ถ้าเจอเลข 3 หลัก ที่ไม่ใช่เลขมิเตอร์ (สมมติห้องไม่เกิน 500)
            if len(c) == 3 and int(c) < 500:
                suggested_room = c
                if c in candidates: candidates.remove(c)
                break

    # ขั้นตอนที่ 3: ตามหา "เลขมิเตอร์"
    # เลขที่เหลืออยู่ใน candidates ตอนนี้ น่าจะเป็นเลขมิเตอร์แล้ว
    if candidates:
        # แปลงเป็นตัวเลข
        possible_meters = [int(x) for x in candidates]
        
        # เลือกตัวที่มีค่า "มากที่สุด" ที่เหลืออยู่ 
        # (เพราะเลขมิเตอร์มักจะเยอะกว่าเลขห้อง หรือเลข Amp)
        suggested_meter = max(possible_meters)
        
        # *หมายเหตุเรื่องการปัดเศษ*: AI อ่าน Text จากภาพ มันจะอ่านตัวที่ชัดที่สุด
        # ถ้าเลขกำลังหมุน AI มักจะอ่านเป็นตัวเต็มตัวใดตัวหนึ่ง 
        # โค้ดนี้จะดึงค่าที่ AI อ่านได้มาแสดง แต่สุดท้ายคนต้องตรวจสอบอีกครั้ง

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

    if img_file:
        bytes_data = img_file.getvalue()
        with st.spinner('🤖 AI กำลังอ่านค่า (สูตรเฉพาะบ่อทอง)...'):
            raw_text = get_text_from_image(bytes_data)
            ai_room, ai_reading = extract_numbers(raw_text)
        
        st.success("อ่านค่าเรียบร้อย! โปรดตรวจสอบความถูกต้องก่อนบันทึก")
        
        # แสดงข้อความดิบ เผื่ออยากเช็คว่า AI เห็นอะไรบ้าง (ซ่อนไว้กดดูได้)
        with st.expander("ดูข้อมูลดิบที่ AI อ่านได้"):
            st.text(raw_text)

    with st.form("meter_form"):
        st.caption("ตรวจสอบตัวเลขด้านล่าง (แก้ไขได้ถ้า AI อ่านผิด)")
        c1, c2 = st.columns(2)
        room_number = c1.text_input("เลขห้อง (ป้ายเหลือง)", value=ai_room)
        # ใส่ help เพื่อเตือนเรื่องการปัดเศษ
        current_reading = c2.number_input("เลขมิเตอร์", min_value=0, value=ai_reading, help="ถ้าเลขกำลังหมุน ให้ใช้ค่าน้อยกว่า")
        
        prev = 0
        usage = 0
        if room_number:
            prev = get_last_reading(room_number, meter_type)
            st.info(f"ครั้งก่อน: **{prev}**")
            
            # คำนวณหน่วยที่ใช้
            if current_reading >= prev: 
                usage = current_reading - prev
            else: 
                # กรณีมิเตอร์วนรอบ (หายากแต่เผื่อไว้) หรือ จดผิด
                st.warning("⚠️ เลขปัจจุบันน้อยกว่าครั้งก่อน?")
                usage = current_reading 
            
            st.metric("หน่วยที่ใช้", usage)

        if st.form_submit_button("💾 บันทึกข้อมูล"):
            if room_number and current_reading > 0:
                save_data(room_number, meter_type, prev, current_reading, usage)
                st.success(f"บันทึกห้อง {room_number} เรียบร้อย!")
                st.balloons()
            else:
                st.error("กรุณาตรวจสอบข้อมูล (เลขห้อง หรือ เลขมิเตอร์ ห้ามเป็น 0)")
