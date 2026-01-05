def extract_numbers(text):
    """แกะตัวเลขโดยปรับสูตรให้เข้ากับมิเตอร์บ่อทอง"""
    # 1. ทำความสะอาดข้อความ ลบตัวอักษรแปลกๆ
    text = text.replace('O', '0').replace('o', '0') # แก้ O เป็น 0 เผื่อ AI อ่านผิด
    
    # 2. หาชุดตัวเลขทั้งหมดในภาพ
    numbers = re.findall(r'\d+', text)
    
    suggested_room = ""
    suggested_meter = 0
    
    candidates = []
    
    # 3. กรองตัวเลข (Filter)
    for num_str in numbers:
        # ลบเลข 0 นำหน้า (เช่น 0145 -> 145)
        val = int(num_str)
        
        # --- กฎการตัดเลขขยะ (Noise Filter) ---
        # ถ้าเจอเลขพวกนี้ ให้ข้ามไปเลย (เลข Voltage, Hz, ปีผลิต, มอก.)
        if val in [220, 50, 15, 45, 2023, 2024, 2025, 2552, 2336]:
            continue
            
        # เก็บตัวเลขที่น่าสนใจไว้ (ต้องมี 3 หลักขึ้นไป)
        if len(num_str) >= 3:
            candidates.append(num_str)

    # 4. ตามหาเลขห้อง (Priority 1)
    # กฎ: เลขห้องของบ่อทอง น่าจะเป็น 4 หลัก และขึ้นต้นด้วย "10" (เช่น 1018, 1020)
    for c in candidates:
        if len(c) == 4 and c.startswith("10"):
            suggested_room = c
            candidates.remove(c) # เจอแล้วลบออกจากกองกลาง
            break # เจอแล้วหยุดหา
    
    # ถ้าหาแบบ 4 หลักไม่เจอ ลองหาแบบ 3 หลัก (เผื่อมีห้อง 101-999)
    if not suggested_room:
        for c in candidates:
            if len(c) == 3 and int(c) < 500: # สมมติว่าห้องไม่เกิน 500
                suggested_room = c
                candidates.remove(c)
                break

    # 5. ตามหาเลขมิเตอร์ (Priority 2)
    # เอาตัวเลขที่เหลือ มาดูว่าตัวไหนค่ามากที่สุด หรือดูเป็นไปได้มากที่สุด
    if candidates:
        # แปลงเป็นตัวเลขแล้วหาตัวที่ดูเป็นมิเตอร์ที่สุด (ส่วนใหญ่เลขมิเตอร์จะเยอะกว่าเลขห้อง)
        # หรือถ้าเหลือตัวเดียว ก็เอาตัวนั้นเลย
        possible_meters = [int(x) for x in candidates]
        suggested_meter = max(possible_meters) # เดาว่าเลขมิเตอร์น่าจะค่าสูงสุดที่เหลืออยู่

    return suggested_room, suggested_meter
