# rc_design_engine.py
import numpy as np
from rc_utils import get_beta1

# -----------------------------------------------------------
# 1. Function คำนวณเหล็กเสริมที่ต้องการ (Required Steel)
# -----------------------------------------------------------
def get_as_req(Mu_kNm, d_eff_mm, fc, fy, b_mm):
    """
    Calculate Required Steel Area based on ACI 318
    MUST RETURN EXACTLY 3 VALUES: (as_req, rho, is_fail)
    """
    if Mu_kNm == 0: 
        return 0.0, 0.0, False
        
    Mu = abs(Mu_kNm) * 1e6 # แปลงหน่วยเป็น N-mm
    phi = 0.9 
    
    # คำนวณ Rn
    Rn = Mu / (phi * b_mm * d_eff_mm**2)
    term_inside = 1 - (2 * Rn) / (0.85 * fc)
    
    # ตรวจสอบว่าหน้าตัดรับไหวไหม (ถ้า term_inside < 0 แปลว่าต้องขยายหน้าตัด หรือใส่เหล็กอัด)
    if term_inside < 0:
        return 0.0, 0.0, True 

    # คำนวณ Rho ที่ต้องการ
    rho = (0.85 * fc / fy) * (1 - np.sqrt(term_inside))
    as_req_calc = rho * b_mm * d_eff_mm
    
    # คำนวณ As_min ตาม ACI
    as_min = max((0.25 * np.sqrt(fc) / fy) * b_mm * d_eff_mm, (1.4 / fy) * b_mm * d_eff_mm)
    
    as_final = max(as_req_calc, as_min)
    
    return float(as_final), float(rho), False

# -----------------------------------------------------------
# 2. Function ช่วยคำนวณตำแหน่งจุดศูนย์ถ่วงกลุ่มเหล็ก (Layer Properties)
# -----------------------------------------------------------
def calculate_layer_properties(layers, b, h, cover, stir_db=10, is_top=False):
    """
    Helper to calculate centroid (d) and extreme tension depth (dt) for multi-layer steel.
    Assume 25mm clear spacing between layers.
    Args:
        stir_db (float): Stirrup diameter (Default 10mm to prevent crash)
    """
    if not layers:
        return 0.0, h - cover, h - cover 

    Ast_total = 0.0
    moment_area_sum = 0.0
    
    # จุดเริ่มต้น: ผิวคอนกรีต -> Cover -> ผิวในเหล็กปลอก
    current_y_base = cover + stir_db 
    
    # ตัวแปรช่วยคำนวณตำแหน่งชั้นถัดไป
    prev_center_y = 0.0
    prev_db = 0.0
    
    # เก็บตำแหน่งแกนเหล็กชั้นนอกสุด (Layer 0) เพื่อหา dt
    first_layer_center = 0.0
    
    for i, lay in enumerate(layers):
        n = lay['n']
        db = lay['db']
        
        if n <= 0: continue
        
        area = n * (np.pi * (db/2)**2)
        
        # คำนวณตำแหน่งจุดศูนย์กลางของเหล็กชั้นนี้ (วัดจากผิวรับแรงดึงเข้ามา)
        if i == 0:
            # ชั้นแรก (ติดผิว): Cover + ปลอก + รัศมีเหล็กแกน
            center_y = current_y_base + db/2
            first_layer_center = center_y
        else:
            # ชั้นถัดไป: Center เดิม + รัศมีเดิม + ช่องว่าง 25mm + รัศมีใหม่
            spacing = 25.0 
            center_y = prev_center_y + (prev_db/2) + spacing + (db/2)
            
        Ast_total += area
        moment_area_sum += area * center_y
        
        # Update ค่าสำหรับรอบถัดไป
        prev_center_y = center_y
        prev_db = db

    if Ast_total == 0:
        return 0.0, 0.0, 0.0

    # Centroid (y_bar) วัดจากผิวรับแรงดึง
    y_bar = moment_area_sum / Ast_total
    
    # Effective Depth (d) = h - y_bar
    d = h - y_bar
    
    # Extreme Tension Depth (dt) = h - Center ของเหล็กชั้นนอกสุด
    dt = h - first_layer_center
        
    return float(Ast_total), float(d), float(dt)

# -----------------------------------------------------------
# 3. Function หลักคำนวณกำลังรับโมเมนต์ (Phi Mn)
# -----------------------------------------------------------
def get_phi_Mn_details(layers, b, h, fc, fy, cover, stir_db=10):
    """
    Calculate Moment Capacity (Phi Mn) for Multi-Layer Steel
    **FIXED**: Added default stir_db=10 to prevent 'missing argument' error.
    """
    # 1. Calculate Group Properties
    Ast, d, dt = calculate_layer_properties(layers, b, h, cover, stir_db)
    
    if Ast == 0: 
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    # 2. Calculate Block Depth (a)
    # T = C => Ast * fy = 0.85 * fc * b * a
    a = (Ast * fy) / (0.85 * fc * b)
    beta1 = get_beta1(fc)
    c = a / beta1
    
    # 3. Check Section Fail (Block exceeds effective depth significantly)
    if a >= d: 
        return 0.0, float(Ast), float(a), 0.0, float(c), -1.0 

    # 4. Calculate Strain at Extreme Tension Steel (dt)
    # ACI 318 ใช้ dt ในการเช็ค Strain (ไม่ใช่ d)
    strain_t = 0.003 * (dt - c) / c if c > 0 else 999.0 

    # 5. Calculate Phi
    if strain_t >= 0.005:
        phi = 0.90
    elif strain_t <= 0.002:
        phi = 0.65
    else:
        phi = 0.65 + 0.25 * ((strain_t - 0.002) / 0.003)

    # 6. Calculate Mn (Moment about centroid of steel group)
    Mn = Ast * fy * (d - a/2)
    phi_Mn = phi * Mn / 1e6 # แปลงเป็น kN-m
    
    return float(phi_Mn), float(Ast), float(a), float(Mn), float(c), float(strain_t)

# -----------------------------------------------------------
# 4. Function ตรวจสอบแรงเฉือน (Shear Check)
# -----------------------------------------------------------
def check_shear_details(Vu_kN, b, d, fc, fy, stir_db=10, spacing=150):
    """
    Check Shear Capacity
    """
    if d <= 0: 
        return "FAIL (Invalid d)", 0.0, 0.0, 0.0, 0.0, 0.0
    
    Vu = abs(Vu_kN) * 1000 # N
    phi = 0.75
    
    Vc = 0.17 * np.sqrt(fc) * b * d
    phi_Vc = phi * Vc
    
    Av = 2 * (np.pi * (stir_db/2)**2)
    s = max(spacing, 1.0)
    Vs = (Av * fy * d) / s
    phi_Vs = phi * Vs
    
    phi_Vn = (phi_Vc + phi_Vs) / 1000 # kN
    
    is_ok = (phi_Vn * 1000) >= Vu
    
    if not is_ok:
        status = f"FAIL (Vu={abs(Vu_kN):.1f} > φVn={phi_Vn:.1f})"
    else:
        status = "OK"

    return status, float(phi_Vn), float(phi_Vc/1000), float(phi_Vs/1000), float(Vc), float(Vs)
