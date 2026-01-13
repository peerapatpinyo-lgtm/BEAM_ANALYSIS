import pandas as pd
import numpy as np

def prepare_load_dataframe(user_loads_df, n_spans, spans, params, f_dl=1.4, f_ll=1.7):
    """
    Processor สำหรับจัดการ Load:
    หน้าที่:
    1. รับ Load รวม (User Input + Self-weight จาก app.py)
    2. แปลงหน่วยให้เป็น N (Newton) ทั้งหมด
    3. คูณ Load Factor (1.4 สำหรับ DL, 1.7 สำหรับ LL)
    4. จัด Format ให้ตรงกับที่ Solver ต้องการ
    """
    
    # 1. ป้องกันกรณีไม่มี Load ส่งมาเลย
    if user_loads_df is None or user_loads_df.empty:
        # ส่งตารางว่างกลับไป เพื่อไม่ให้โปรแกรม Error
        return pd.DataFrame(columns=['span_index', 'type', 'mag', 'dist', 'd_start'])

    processed_loads = []

    # 2. วนลูปจัดการ Load ทีละรายการ
    for _, load in user_loads_df.iterrows():
        
        # --- A. เลือก Factor ตามประเภท ---
        case_type = load.get('case', 'DL') # ถ้าไม่ระบุ ถือเป็น DL
        
        if case_type in ['DL', 'SW', 'Dead', 'Superimposed Dead']:
            factor = f_dl  # ปกติคือ 1.4
        elif case_type in ['LL', 'Live']:
            factor = f_ll  # ปกติคือ 1.7
        else:
            factor = 1.0   # กรณีอื่นๆ หรือ Service Load

        # --- B. จัดการเรื่องหน่วย (Unit Conversion) ---
        raw_mag = float(load['mag'])
        
        # Logic การเช็คหน่วย:
        # - ถ้าค่า load > 500 สันนิษฐานว่าเป็นหน่วย N (เช่น SW ที่ app คำนวณมา = 2400)
        # - ถ้าค่า load น้อยๆ สันนิษฐานว่าเป็น kN (เช่น User กรอก 10, 20) -> คูณ 1000
        if raw_mag > 500.0:
            mag_N = raw_mag
        else:
            mag_N = raw_mag * 1000.0

        # --- C. คำนวณ Ultimate Load (คูณ Factor) ---
        factored_mag_N = mag_N * factor

        # --- D. เก็บข้อมูลลง List ---
        processed_loads.append({
            'span_index': int(load['span_index']),
            'type': load['type'],                       # 'P' (Point) หรือ 'U' (Uniform)
            'mag': factored_mag_N,                      # ค่าที่คูณ Factor และเป็นหน่วย N แล้ว
            'dist': float(load.get('dist', 0)),         # ความยาว Load (สำหรับ UDL)
            'd_start': float(load.get('d_start', 0)),   # ระยะเริ่ม Load
            'case_origin': case_type                    # เก็บไว้ตรวจสอบได้
        })

    # 3. ส่งผลลัพธ์กลับเป็น DataFrame
    return pd.DataFrame(processed_loads)
