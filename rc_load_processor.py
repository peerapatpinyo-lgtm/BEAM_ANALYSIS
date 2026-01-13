import pandas as pd
import numpy as np

def prepare_load_dataframe(user_loads_df, n_spans, spans, params, f_dl=1.4, f_ll=1.7):
    """
    Processor สำหรับจัดการ Load:
    1. รับ Load ทั้งหมด (User Inputs + Self-weight จาก app.py)
    2. จัดการเรื่องหน่วย (Convert kN -> N)
    3. คูณ Load Factor (U = f_dl*DL + f_ll*LL)
    """
    
    # ถ้าไม่มี Load เลย ให้คืนค่า DataFrame ว่างๆ กลับไป
    if user_loads_df is None or user_loads_df.empty:
        return pd.DataFrame(columns=['span_index', 'type', 'mag', 'dist', 'd_start'])

    processed_loads = []

    # วนลูปเช็ค Load ทีละรายการ
    for _, load in user_loads_df.iterrows():
        # 1. เช็คประเภท Load เพื่อระบุ Factor
        case_type = load.get('case', 'DL')
        
        # กำหนด Factor
        if case_type in ['DL', 'SW', 'Dead', 'Superimposed Dead']:
            factor = f_dl
        elif case_type in ['LL', 'Live']:
            factor = f_ll
        else:
            factor = 1.0

        # 2. จัดการหน่วย (Unit Handling) & คูณ Factor
        raw_mag = float(load['mag'])
        
        # [Smart Unit Check]
        # app.py ส่ง Self-weight มาเป็น N/m (ค่าจะหลักพัน เช่น 3600)
        # แต่ User Input มักใส่เป็น kN/m (ค่าจะหลักสิบ เช่น 10, 20)
        # เราจึงใช้เงื่อนไขนี้แยกแยะเพื่อแปลงให้เป็น N ทั้งหมด
        if raw_mag > 500.0:
            # สันนิษฐานว่าเป็นหน่วย N/m หรือ N แล้ว (เช่น SW) -> ไม่ต้องคูณ 1000
            mag_N = raw_mag
        else:
            # สันนิษฐานว่าเป็นหน่วย kN/m หรือ kN (User Input) -> แปลงเป็น N
            mag_N = raw_mag * 1000.0

        factored_mag_N = mag_N * factor

        # 3. เตรียมข้อมูลส่งให้ Solver
        processed_loads.append({
            'span_index': int(load['span_index']),
            'type': load['type'],                   # 'P' or 'U'
            'mag': factored_mag_N,                  # หน่วย N (Factored)
            'dist': float(load.get('dist', 0)),     
            'd_start': float(load.get('d_start', 0)), 
            'case_origin': case_type
        })

    # ส่งค่ากลับเป็น DataFrame
    return pd.DataFrame(processed_loads)
