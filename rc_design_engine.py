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

def calculate_layer_properties(layers, b, h, cover, stir_db, is_top=False):
    """
    Helper to calculate centroid (d) and extreme tension depth (dt) for multi-layer steel.
    Assume 25mm clear spacing between layers.
    layers = [{'n': 2, 'db': 16}, {'n': 2, 'db': 20}] (Ordered from Outer to Inner)
    """
    if not layers:
        return 0.0, h, h # No steel

    Ast_total = 0.0
    moment_area_sum = 0.0
    
    # ระยะจากผิวคอนกรีตถึงจุดศูนย์กลางเหล็กแต่ละชั้น
    # Layer 0 คือชั้นนอกสุด (ติดผิว), Layer 1 คือชั้นถัดเข้าไป
    current_y = cover + stir_db # Start at inside of stirrup
    
    extreme_center = 0.0 # Keep track of outer-most layer center for dt
    
    for i, lay in enumerate(layers):
        n = lay['n']
        db = lay['db']
        
        if n <= 0: continue
        
        area = n * (np.pi * (db/2)**2)
        
        # Calculate center of this layer
        if i == 0:
            center_dist = current_y + db/2
            extreme_center = center_dist
        else:
            # Previous layer center + prev_db/2 + spacing + current_db/2
            prev_db = layers[i-1]['db']
            spacing = 25.0 # Standard min clear spacing
            # Distance from previous center to this center
            step = (prev_db / 2) + spacing + (db / 2)
            center_dist = extreme_center + step # This logic assumes stacking linear relative to 1st layer, simplistic but robust
            # Update explicit calculation for stacking:
            # Better: Calculate Y from surface cumulatively
            # But simpler: Just add spacing to previous Y
             
        # Re-calc strictly:
        # Layer 0 center: cover + stir + db/2
        # Layer 1 center: Layer 0 center + db0/2 + 25 + db1/2
        
        if i == 0:
            y_loc = cover + stir_db + db/2
            extreme_y = y_loc
        else:
            prev_db = layers[i-1]['db']
            y_loc = extreme_y + (prev_db/2) + 25.0 + (db/2) # Add spacing
            extreme_y = y_loc # Update for next loop (Wait, this is moving inwards)
            
            # Correct Logic: 
            # We need absolute distance from the compression face? 
            # No, let's calculate distance from Tension Face first (y_bottom), then convert to d.
        
        Ast_total += area
        moment_area_sum += area * y_loc
        
        # Store for next iteration reference if needed
        # (In this simple loop, re-calculating y_loc based on previous is fine)

    if Ast_total == 0:
        return 0.0, 0.0, 0.0

    # Centroid from Tension Face
    y_bar = moment_area_sum / Ast_total
    
    # Effective Depth (d) = h - y_bar
    d = h - y_bar
    
    # Extreme Tension Depth (dt) = h - (Distance to center of outer-most layer)
    # Layer 0 is outer-most
    first_layer_db = layers[0]['db']
    dist_to_first_center = cover + stir_db + first_layer_db/2
    dt = h - dist_to_first_center
    
    if is_top:
        # Check logic for Top bars? Same math, just flipped reference.
        # d is distance from Bottom face (Compression) to Centroid of Top Steel
        pass 
        
    return Ast_total, d, dt

def get_phi_Mn_details(layers, b, h, fc, fy, cover, stir_db):
    """
    Calculate Moment Capacity (Phi Mn) for Multi-Layer Steel
    layers: list of dict [{'n':.., 'db':..}, ...]
    MUST RETURN EXACTLY 6 VALUES: (phi_Mn, Ast, a, Mn, c, strain_t)
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
    # ACI 318: epsilon_t is based on dt, not d
    strain_t = 0.003 * (dt - c) / c if c > 0 else 999.0 

    # 5. Calculate Phi
    if strain_t >= 0.005:
        phi = 0.90
    elif strain_t <= 0.002:
        phi = 0.65
    else:
        phi = 0.65 + 0.25 * ((strain_t - 0.002) / 0.003)

    # 6. Calculate Mn (Moment about centroid of steel group)
    # Note: Using d (centroid) for moment arm is correct for group resultant
    Mn = Ast * fy * (d - a/2)
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
        status = f"FAIL (Mu={abs(Vu_kN):.1f} > φVn={phi_Vn:.1f} kN)"
    else:
        status = "OK"

    # ส่งคืน 6 ค่า
    return status, float(phi_Vn), float(phi_Vc/1000), float(phi_Vs/1000), float(Vc), float(Vs)
