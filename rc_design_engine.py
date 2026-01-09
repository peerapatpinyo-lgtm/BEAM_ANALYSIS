# rc_design_engine.py
import numpy as np
from rc_utils import get_beta1

def get_as_req(Mu_kNm, d_eff_mm, fc, fy, b_mm):
    """
    Calculate Required Steel Area based on ACI 318
    MUST RETURN EXACTLY 3 VALUES: (as_req, rho, is_fail)
    """
    # 1. จัดการกรณี Moment เป็น 0
    if Mu_kNm == 0: 
        return 0.0, 0.0, False
        
    Mu = abs(Mu_kNm) * 1e6 # หน่วย N-mm
    phi = 0.9 
    
    # 2. คำนวณ Rn และตรวจสอบหน้าตัด
    Rn = Mu / (phi * b_mm * d_eff_mm**2)
    term_inside = 1 - (2 * Rn) / (0.85 * fc)
    
    # 3. จัดการกรณีหน้าตัดเล็กเกินไป (Section Fail)
    if term_inside < 0:
        return 0.0, 0.0, True 

    # 4. คำนวณพื้นที่เหล็ก
    rho = (0.85 * fc / fy) * (1 - np.sqrt(term_inside))
    as_req_calc = rho * b_mm * d_eff_mm
    
    # 5. พื้นที่เหล็กขั้นต่ำ (As_min)
    as_min = max((0.25 * np.sqrt(fc) / fy) * b_mm * d_eff_mm, (1.4 / fy) * b_mm * d_eff_mm)
    
    as_final = max(as_req_calc, as_min)
    
    # ส่งคืน 3 ค่าตามที่ app.py ต้องการเป๊ะๆ
    return float(as_final), float(rho), False

def get_phi_Mn_details(n, db, d_eff, b, fc, fy):
    """
    Calculate Moment Capacity (Phi Mn)
    MUST RETURN EXACTLY 6 VALUES: (phi_Mn, Ast, a, Mn, c, strain_t)
    """
    Ast = n * (np.pi * (db/2)**2)
    if Ast == 0: 
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    a = (Ast * fy) / (0.85 * fc * b)
    beta1 = get_beta1(fc)
    c = a / beta1
    
    # ตรวจสอบความลึก block
    if a >= d_eff: 
        return 0.0, float(Ast), float(a), 0.0, float(c), -1.0 

    # คำนวณ Strain
    strain_t = 0.003 * (d_eff - c) / c if c > 0 else 999.0 

    # หาค่า Phi
    if strain_t >= 0.005:
        phi = 0.90
    elif strain_t <= 0.002:
        phi = 0.65
    else:
        phi = 0.65 + 0.25 * ((strain_t - 0.002) / 0.003)

    Mn = Ast * fy * (d_eff - a/2)
    phi_Mn = phi * Mn / 1e6 # kN-m
    
    # ส่งคืน 6 ค่า
    return float(phi_Mn), float(Ast), float(a), float(Mn), float(c), float(strain_t)

def check_shear_details(Vu_kN, b, d, fc, fy, stir_db, spacing):
    """
    Check Shear Capacity
    MUST RETURN EXACTLY 6 VALUES: (status, phi_Vn, phi_Vc, phi_Vs, Vc, Vs)
    """
    if d <= 0: 
        return "FAIL (Invalid d)", 0.0, 0.0, 0.0, 0.0, 0.0
    
    Vu = abs(Vu_kN) * 1000 # N
    phi = 0.75 # ACI Shear
    
    Vc = 0.17 * np.sqrt(fc) * b * d
    phi_Vc = phi * Vc
    
    Av = 2 * (np.pi * (stir_db/2)**2)
    s = max(spacing, 1.0)
    Vs = (Av * fy * d) / s
    phi_Vs = phi * Vs
    
    phi_Vn = (phi_Vc + phi_Vs) / 1000 # kN
    
    is_ok = (phi_Vn * 1000) >= Vu
    
    # สร้าง Status พร้อมระบุหน่วยเปรียบเทียบ
    if not is_ok:
        status = f"FAIL (Vu={abs(Vu_kN):.1f} > φVn={phi_Vn:.1f} kN)"
    else:
        status = "OK"

    # ส่งคืน 6 ค่า
    return status, float(phi_Vn), float(phi_Vc/1000), float(phi_Vs/1000), float(Vc), float(Vs)
