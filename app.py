import streamlit as st
import pandas as pd
from google.cloud import vision
from google.oauth2 import service_account
import gspread
import datetime
import re

# ==========================================
# ⚙️ 1. ตั้งค่าและรายชื่อห้อง (Master Data)
# ==========================================
KEY_FILE = 'credentials.json' 
SHEET_NAME = 'Bothong_Meter_Data' 

# ตั้งราคาต่อหน่วย (สำหรับคำนวณเงินเบื้องต้น)
UNIT_PRICE_WATER = 18  # แก้ไขราคาค่าน้ำตรงนี้
UNIT_PRICE_ELEC = 7    # แก้ไขราคาค่าไฟตรงนี้

# สร้างรายชื่อห้อง 269 ห้อง ตามข้อมูลจริง
# 🟢 โซนเขียว (1): 1001-1032
zone_green = [str(x) for x in range(1001, 1033)]
# 🟠 โซนส้ม (2): 2001-2058
zone_orange = [str(x) for x in range(2001, 2059)]
# 🔘 โซนเทา (3): 3001-3043
zone_grey = [str(x) for x in range(3001, 3044)]
# 🔵 โซนฟ้า (4): 4001-4136
zone_blue = [str(x) for x in range(4001, 4137)]

# รวมเป็นรายชื่อห้องทั้งหมด
ALL_ROOMS = zone_green + zone_orange + zone_grey + zone_blue

st.set_page_config(page_title="บ่อทอง เรสซิเด้นท์", page_icon="🏢", layout="centered")

# ==========================================
# 🔌 2. เชื่อมต่อระบบ
# ==========================================
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

# ==========================================
# 🧠 3. ฟังก์ชันคำนวณต่างๆ
# ==========================================

def get_text_from_image(image_bytes):
    if vision_client is None: return ""
    image = vision.Image(content=image_bytes)
    response = vision_client.document_text_detection(image=image)
    if response.full_text_annotation:
        return response.full_text_annotation.text
    return ""

def extract_numbers(text, m_type):
    """แกะตัวเลขและตรวจสอบกับรายชื่อห้องจริง"""
    # ล้างค่า
    text_clean = text.replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1')
    text_merged = text_clean.replace(" ", "") 
    numbers_raw = re.findall(r'\d+', text_clean)
    numbers_merged = re.findall(r'\d+', text_merged)
    all_candidates = set(numbers_raw + numbers_merged)
    
    suggested_room = ""
    suggested_meter = 0
    meter_candidates = []
    
    ignore_list = [220, 50, 15, 45, 100, 400, 2023, 2024, 2025, 2552, 2336, 2124, 65057, 6505, 1, 2, 33]

    # --- 1. หาเลขห้อง (ต้องตรงกับรายชื่อจริงเท่านั้น) ---
    found_in_master = [n for n in all_candidates if n in ALL_ROOMS]
    if found_in_master:
        suggested_room = found_in_master[0] # เอาตัวแรกที่เจอ
    
    # ถ้าไม่เจอ ลองเดา 4 หลัก
    if not suggested_room:
        for n in all_candidates:
            if len(n) == 4 and n.startswith(('1','2','3','4')):
                suggested_room = n
                break

    # --- 2. หาเลขมิเตอร์ ---
    for n in all_candidates:
        if n == suggested_room: continue
        if len(n) < 3 or len(n) > 6: continue
        if int(n) in ignore_list: continue
        meter_candidates.append(int(n))

    if meter_candidates:
        if m_type == 'ไฟฟ้า': # ไฟฟ้าเอาเลข 5 หลักก่อน
            prio = [x for x in meter_candidates if 10000 <= x <= 99999]
            suggested_meter = max(prio) if prio else max(meter_candidates)
        else: # น้ำเอาเลข 4 หลักก่อน
            prio = [x for x in meter_candidates if 1000 <= x <= 9999]
            suggested_meter = max(prio) if prio else max(meter_candidates)

    return suggested_room, suggested_meter

def check_progress(meter_type):
    """เช็กว่าเดือนนี้จดไปกี่ห้อง ขาดห้องไหนบ้าง"""
    if sh is None: return [], 0
    try:
        ws = sh.worksheet("Logs")
        records = ws.get_all_records()
        df = pd.DataFrame(records)
        
        # หาวันที่ปัจจุบัน (ปี-เดือน) เช่น 2024-05
        current_month = datetime.datetime.now().strftime("%Y-%m")
        df['Timestamp'] = df['Timestamp'].astype(str)
        
        # กรองข้อมูลเฉพาะเดือนนี้ และประเภทมิเตอร์นี้
        done_df = df[
            (df['Timestamp'].str.contains(current_month)) & 
            (df['Type'] == meter_type)
        ]
        
        # รายชื่อห้องที่จดแล้ว
        done_rooms = set(done_df['Room'].astype(str).unique())
        
        # รายชื่อห้องทั้งหมด
        all_rooms_set = set(ALL_ROOMS)
        
        # หาห้องที่ขาด (ทั้งหมด - ที่จดแล้ว)
        missing = sorted(list(all_rooms_set - done_rooms), key=lambda x: int(x))
        
        return missing, len(done_rooms)
    except Exception as e:
        return ALL_ROOMS, 0

def save_data(room, m_type, prev, curr, usage):
    if sh is None: return
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. ลงบันทึก Logs
    try:
        log_sheet = sh.worksheet("Logs")
    except:
        log_sheet = sh.add_worksheet(title="Logs", rows="1000", cols="20")
        log_sheet.append_row(["Timestamp", "Room", "Type", "Previous", "Current", "Usage"])

    log_sheet.append_row([timestamp, room, m_type, prev, curr, usage])
    
    # 2. อัปเดตสถานะล่าสุด
    try:
        status_sheet = sh.worksheet("Latest_Status")
    except:
        status_sheet = sh.add_worksheet(title="Latest_Status", rows="500", cols="5")
        status_sheet.append_row(["Room", "Last_Water", "Last_Elec"])

    try:
        cell = status_sheet.find(str(room))
        if cell:
            col_index = 2 if m_type == 'น้ำประปา' else 3
            status_sheet.update_cell(cell.row, col_index, curr)
        else:
            new_water = curr if m_type == 'น้ำประปา' else 0
            new_elec = curr if m_type == 'ไฟฟ้า' else 0
            status_sheet.append_row([room, new_water, new_elec])
    except:
        pass

# ==========================================
# 📱 4. หน้าจอแอป (UI)
# ==========================================
st.title("💧⚡ บ่อทอง เรสซิเด้นท์")
st.caption(f"ระบบบริหารจัดการมิเตอร์ (ทั้งหมด {len(ALL_ROOMS)} ห้อง)")

if sh is None:
    st.warning("⚠️ กำลังเชื่อมต่อฐานข้อมูล...")
else:
    # เลือกประเภท
    meter_type = st.radio("เลือกประเภทมิเตอร์:", ["น้ำประปา", "ไฟฟ้า"], horizontal=True)

    # --- 📊 Dashboard แสดงยอดคงเหลือ (ฟีเจอร์ข้อ 2) ---
    missing_list, count_done = check_progress(meter_type)
    total = len(ALL_ROOMS)
    percent = count_done / total if total > 0 else 0
    
    st.markdown("---")
    c1, c2 = st.columns([3, 1])
    with c1:
        st.subheader(f"📊 ความคืบหน้าเดือนนี้: {meter_type}")
    with c2:
        st.metric("จดแล้ว", f"{count_done}/{total}")

    st.progress(percent)
    
    # แสดงรายชื่อห้องที่ขาด
    if len(missing_list) > 0:
        with st.expander(f"❌ ดูรายชื่อห้องที่ยังไม่จด ({len(missing_list)} ห้อง)"):
            # แสดงแบบแบ่งกลุ่มให้อ่านง่าย
            st.write(", ".join(missing_list))
    else:
        st.success("🎉 เก่งมาก! จดครบทุกห้องแล้วครับ")
    st.markdown("---")

    # --- ส่วนทำงาน (ถ่ายรูป/กรอก) ---
    tab1, tab2 = st.tabs(["📸 ถ่ายรูป", "📂 อัปโหลดรูป"])
    img_file = None
    
    with tab1:
        camera_img = st.camera_input(f"ถ่ายรูปมิเตอร์{meter_type}")
        if camera_img: img_file = camera_img
    with tab2:
        uploaded_img = st.file_uploader("เลือกรูปจากเครื่อง", type=['jpg','png','jpeg'])
        if uploaded_img: 
            st.image(uploaded_img, width=300)
            img_file = uploaded_img

    ai_room = ""
    ai_reading = 0

    if img_file:
        bytes_data = img_file.getvalue()
        with st.spinner('🤖 AI กำลังอ่านค่า...'):
            raw_text = get_text_from_image(bytes_data)
            ai_room, ai_reading = extract_numbers(raw_text, meter_type)
        
        if ai_room in ALL_ROOMS:
            st.success(f"✅ AI พบห้อง {ai_room} ในระบบถูกต้อง")
        elif ai_room:
            st.warning(f"⚠️ AI อ่านได้ {ai_room} แต่ไม่พบในรายชื่อห้อง (โปรดตรวจสอบ)")
        
    with st.form("main_form"):
        c1, c2 = st.columns(2)
        room_number = c1.text_input("เลขห้อง", value=ai_room)
        current_reading = c2.number_input("เลขมิเตอร์", min_value=0, value=ai_reading)
        
        # ดึงค่าครั้งก่อน
        prev = 0
        try:
            ws = sh.worksheet("Latest_Status")
            records = ws.get_all_records()
            df_status = pd.DataFrame(records)
            df_status['Room'] = df_status['Room'].astype(str)
            row = df_status[df_status['Room'] == str(room_number)]
            if not row.empty:
                col = 'Last_Water' if meter_type == 'น้ำประปา' else 'Last_Elec'
                prev = int(row.iloc[0][col])
        except:
            prev = 0
            
        st.info(f"ครั้งก่อน: **{prev}**")
        usage = 0
        if current_reading >= prev:
            usage = current_reading - prev
        else:
            st.warning("⚠️ เลขมิเตอร์น้อยกว่าครั้งก่อน?")
            usage = current_reading
            
        # คำนวณค่าใช้จ่าย (ประมาณการ)
        st.divider()
        unit_price = UNIT_PRICE_WATER if meter_type == 'น้ำประปา' else UNIT_PRICE_ELEC
        est_cost = usage * unit_price
        
        m1, m2 = st.columns(2)
        m1.metric("หน่วยที่ใช้ (Usage)", f"{usage} หน่วย")
        m2.metric("ค่าใช้จ่ายประมาณ (บาท)", f"{est_cost:,.2f} ฿", help=f"คิดที่หน่วยละ {unit_price} บาท")
        
        if st.form_submit_button("💾 บันทึกข้อมูล"):
            if room_number not in ALL_ROOMS:
                st.error(f"❌ ห้อง {room_number} ไม่มีอยู่ในรายชื่อโครงการ! กรุณาตรวจสอบ")
            elif current_reading <= 0:
                st.error("❌ กรุณากรอกเลขมิเตอร์")
            else:
                save_data(room_number, meter_type, prev, current_reading, usage)
                st.success(f"บันทึกห้อง {room_number} เรียบร้อย!")
                st.rerun() # รีเฟรชหน้าจอเพื่ออัปเดต Dashboard ทันที
