import streamlit as st
import pandas as pd
from google.cloud import vision
from google.oauth2 import service_account
import gspread
import datetime
import re
import json

# ==========================================
# ⚙️ ส่วนตั้งค่ารายชื่อห้อง (Master Data)
# ==========================================
KEY_FILE = 'credentials.json' 
SHEET_NAME = 'Bothong_Meter_Data' 

# สร้างรายชื่อห้องตามข้อมูลที่แจ้งมา (รวม 269 ห้อง)
# 🟢 โซนสีเขียว (เลข 1): 1001 - 1032 (32 ห้อง)
zone_green = [str(x) for x in range(1001, 1033)]

# 🟠 โซนสีส้ม (เลข 2): 2001 - 2058 (58 ห้อง)
zone_orange = [str(x) for x in range(2001, 2059)]

# 🔘 โซนสีเทา (เลข 3): 3001 - 3043 (43 ห้อง)
zone_grey = [str(x) for x in range(3001, 3044)]

# 🔵 โซนสีฟ้า (เลข 4): 4001 - 4136 (136 ห้อง)
zone_blue = [str(x) for x in range(4001, 4137)]

# รวมทั้งหมดเป็น Master List
ALL_ROOMS = zone_green + zone_orange + zone_grey + zone_blue

st.set_page_config(page_title="บ่อทอง เรสซิเด้นท์", page_icon="🏢")

# ==========================================
# 🔌 ส่วนเชื่อมต่อระบบ
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
# 🧠 ฟังก์ชันคำนวณและจัดการข้อมูล
# ==========================================

def get_text_from_image(image_bytes):
    if vision_client is None: return ""
    image = vision.Image(content=image_bytes)
    response = vision_client.document_text_detection(image=image)
    if response.full_text_annotation:
        return response.full_text_annotation.text
    return ""

def extract_numbers(text, m_type):
    """
    สูตรแกะตัวเลข V.4 (ใช้ Master Data ตรวจสอบ)
    """
    # 1. ทำความสะอาดข้อความ
    text_clean = text.replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1')
    text_merged = text_clean.replace(" ", "") 
    
    numbers_raw = re.findall(r'\d+', text_clean)
    numbers_merged = re.findall(r'\d+', text_merged)
    all_candidates = set(numbers_raw + numbers_merged)
    
    suggested_room = ""
    suggested_meter = 0
    meter_candidates = []
    
    # เลขขยะที่ต้องตัดทิ้ง
    ignore_list = [
        220, 50, 15, 45, 100, 400, 
        2023, 2024, 2025, 2552, 2336, 
        2124, 65057, 6505, 
        1, 2, 33 
    ]

    # --- Priority 1: หาเลขห้องจาก Master List (ALL_ROOMS) ---
    # ถ้าตัวเลขที่เจอ "มีอยู่จริง" ในรายชื่อห้อง ให้ฟันธงว่าเป็นเลขห้องทันที
    found_rooms = []
    for num_str in all_candidates:
        if num_str in ALL_ROOMS:
            found_rooms.append(num_str)
    
    # ถ้าเจอเลขห้องใน List จริง
    if found_rooms:
        # ถ้าเจอหลายห้อง (เช่น เจอทั้ง 1001 และ 1002 ในรูปเดียว) ให้เดาว่าเป็นห้องที่อยู่ตรงกลางภาพ 
        # (แต่ในที่นี้เอาตัวแรกไปก่อน หรือให้ user เลือกเองถ้าผิด)
        suggested_room = found_rooms[0]
    
    # ถ้าไม่เจอใน List จริง (อาจจะอ่านผิดเพี้ยนไปบ้าง) ลองเดาจาก Pattern 4 หลัก
    if not suggested_room:
        for num_str in all_candidates:
            if len(num_str) == 4 and num_str.startswith(('1', '2', '3', '4')):
                # ลองเช็คว่ามันใกล้เคียงกับห้องที่มีไหม (ข้ามไปก่อน เอาแค่รูปแบบ)
                suggested_room = num_str
                break

    # --- Priority 2: หาเลขมิเตอร์ ---
    for num_str in all_candidates:
        # ข้ามเลขห้องที่เจอไปแล้ว
        if num_str == suggested_room: continue
        # ข้ามเลขสั้นๆ หรือยาวเกินไป
        if len(num_str) < 3: continue
        if len(num_str) > 6: continue 
        
        val = int(num_str)
        if val in ignore_list: continue
        
        meter_candidates.append(val)

    if meter_candidates:
        if m_type == 'ไฟฟ้า':
            # ไฟฟ้า เน้น 5 หลัก
            priority = [x for x in meter_candidates if 10000 <= x <= 99999]
            suggested_meter = max(priority) if priority else max(meter_candidates)
        else: 
            # น้ำประปา เน้น 4 หลัก
            priority = [x for x in meter_candidates if 1000 <= x <= 9999]
            suggested_meter = max(priority) if priority else max(meter_candidates)

    return suggested_room, suggested_meter

def sort_latest_status():
    if sh is None: return
    try:
        ws = sh.worksheet("Latest_Status")
        data = ws.get_all_records()
        if not data: return

        df = pd.DataFrame(data)
        try:
            # แปลงเป็นตัวเลขเพื่อเรียงลำดับ (เพราะห้องเราเป็นตัวเลขทั้งหมดแล้ว)
            df['Room_Int'] = df['Room'].astype(int)
            df = df.sort_values(by='Room_Int')
            df = df.drop(columns=['Room_Int'])
        except:
            df = df.sort_values(by='Room')

        ws.clear()
        ws.update([df.columns.values.tolist()] + df.values.tolist())
    except Exception as e:
        print(f"Sort Error: {e}")

def save_data(room, m_type, prev, curr, usage):
    if sh is None: return
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_sheet = sh.worksheet("Logs")
    log_sheet.append_row([timestamp, room, m_type, prev, curr, usage])
    
    status_sheet = sh.worksheet("Latest_Status")
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
        new_water = curr if m_type == 'น้ำประปา' else 0
        new_elec = curr if m_type == 'ไฟฟ้า' else 0
        status_sheet.append_row([room, new_water, new_elec])
    
    sort_latest_status()

def check_missing_rooms(meter_type):
    if sh is None: return [], 0
    try:
        ws = sh.worksheet("Logs")
        records = ws.get_all_records()
        df = pd.DataFrame(records)
        
        current_month = datetime.datetime.now().strftime("%Y-%m")
        df['Timestamp'] = df['Timestamp'].astype(str)
        
        filtered = df[
            (df['Timestamp'].str.contains(current_month)) & 
            (df['Type'] == meter_type)
        ]
        
        recorded_rooms = set(filtered['Room'].astype(str).unique())
        all_rooms_set = set(str(r) for r in ALL_ROOMS)
        
        missing = sorted(list(all_rooms_set - recorded_rooms), key=lambda x: int(x))
        total_recorded = len(recorded_rooms)
        
        return missing, total_recorded
    except:
        return ALL_ROOMS, 0

# ==========================================
# 📱 ส่วนหน้าจอแอป (UI)
# ==========================================
st.title("💧⚡ บ่อทอง เรสซิเด้นท์")

if sh is None:
    st.warning("⚠️ กำลังเชื่อมต่อระบบ...")
else:
    meter_type = st.radio("เลือกประเภท:", ["น้ำประปา", "ไฟฟ้า"], horizontal=True)

    # --- ส่วนตรวจสอบความคืบหน้า (Dashboard) ---
    with st.expander(f"📊 ตรวจสอบยอดจดมิเตอร์ ({meter_type})", expanded=True):
        missing_list, count_done = check_missing_rooms(meter_type)
        total_rooms = len(ALL_ROOMS)
        
        # Progress Bar
        progress = count_done / total_rooms if total_rooms > 0 else 0
        st.progress(progress)
        st.write(f"✅ จดไปแล้ว: **{count_done}** / {total_rooms} ห้อง")
        
        if missing_list:
            st.warning(f"❌ เหลืออีก: **{len(missing_list)}** ห้อง")
            # แสดงรายชื่อห้องที่ขาด แบบย่อๆ (ถ้าเยอะเกินให้ซ่อน)
            if len(missing_list) > 20:
                st.caption(f"ตัวอย่างห้องที่ขาด: {', '.join(missing_list[:10])} ... และอีก {len(missing_list)-10} ห้อง")
            else:
                st.info(f"ห้องที่ขาด: {', '.join(missing_list)}")
        else:
            st.success("🎉 สุดยอด! ครบทุกห้องแล้วครับ")

    st.divider()

    # --- ส่วนถ่ายรูป / อัปโหลด ---
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
        with st.spinner('🤖 AI กำลังอ่านค่า (ตรวจสอบรายชื่อห้องจริง)...'):
            raw_text = get_text_from_image(bytes_data)
            ai_room, ai_reading = extract_numbers(raw_text, meter_type)
        
        if ai_room in ALL_ROOMS:
            st.success(f"✅ AI เจอห้อง {ai_room} ในระบบ!")
        elif ai_room:
            st.warning(f"⚠️ AI อ่านได้เลข {ai_room} แต่ไม่พบในรายชื่อห้อง (โปรดตรวจสอบ)")
        else:
            st.warning("⚠️ AI ไม่เห็นเลขห้องที่ชัดเจน")

    # --- แบบฟอร์มบันทึก ---
    with st.form("meter_form"):
        c1, c2 = st.columns(2)
        room_number = c1.text_input("เลขห้อง", value=ai_room)
        current_reading = c2.number_input("เลขมิเตอร์", min_value=0, value=ai_reading)
        
        prev = 0
        usage = 0
        
        if room_number:
            # ดึงค่าเก่า
            try:
                ws_status = sh.worksheet("Latest_Status")
                records = ws_status.get_all_records()
                df_status = pd.DataFrame(records)
                df_status['Room'] = df_status['Room'].astype(str)
                row = df_status[df_status['Room'] == str(room_number)]
                if not row.empty:
                    col_name = 'Last_Water' if meter_type == 'น้ำประปา' else 'Last_Elec'
                    prev = int(row.iloc[0][col_name])
            except:
                prev = 0

            st.info(f"ครั้งก่อน: **{prev}**")
            
            if current_reading >= prev: 
                usage = current_reading - prev
            else: 
                st.warning("⚠️ เลขปัจจุบันน้อยกว่าครั้งก่อน (ตรวจสอบอีกครั้ง)")
                usage = current_reading 
            
            st.metric("หน่วยที่ใช้", usage)

        if st.form_submit_button("💾 บันทึกข้อมูล"):
            # ตรวจสอบว่าห้องถูกต้องไหม
            if room_number not in ALL_ROOMS:
                st.error(f"❌ ห้อง {room_number} ไม่อยู่ในรายชื่อโครงการ! (กรุณาแก้เลขห้องให้ถูก)")
            elif current_reading <= 0:
                st.error("❌ เลขมิเตอร์ต้องมากกว่า 0")
            else:
                save_data(room_number, meter_type, prev, current_reading, usage)
                st.success(f"บันทึกห้อง {room_number} เรียบร้อย!")
                st.rerun()
