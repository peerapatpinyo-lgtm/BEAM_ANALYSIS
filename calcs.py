# calcs.py
import numpy as np

def normalize_section_units(b_input, h_input):
    """
    ตรวจสอบและแปลงหน่วยอัตโนมัติ (Meters -> Millimeters)
    """
    # จัดการความกว้าง (b)
    if b_input < 10:
        b_mm = b_input * 1000
    else:
        b_mm = b_input
        
    # จัดการความลึก (h)
    if h_input < 10:
        h_mm = h_input * 1000
    else:
        h_mm = h_input
        
    return b_mm, h_mm

def get_beta1(fc):
    """Calculate Beta1 factor according to ACI 318"""
    if fc <= 28:
        return 0.85
    elif fc >= 55:
        return 0.65
    else:
        return 0.85 - 0.05 * (fc - 28) / 7

def get_as_req(Mu_kNm, d_eff_mm, fc, fy, b_mm):
    """Calculate Required Steel Area based on ACI 318"""
    if Mu_kNm == 0: return 0.0, 0.0, False
    Mu = abs(Mu_kNm) * 1e6 # N-mm
    phi = 0.9 
    
    Rn = Mu / (phi * b_mm * d_eff_mm**2)
    
    term_inside = 1 - (2 * Rn) / (0.85 * fc)
    
    if term_inside < 0:
        return 0.0, 0.0, True # Section too small (Fail)

    rho = (0.85 * fc / fy) * (1 - np.sqrt(term_inside))
    as_req = rho * b_mm * d_eff_mm
    
    # Minimum Steel
    as_min1 = (0.25 * np.sqrt(fc) / fy) * b_mm * d_eff_mm
    as_min2 = (1.4 / fy) * b_mm * d_eff_mm
    as_min = max(as_min1, as_min2)
    
    return max(as_req, as_min), rho, False

def get_phi_Mn_details(n, db, d_eff, b, fc, fy):
    """Calculate Moment Capacity with Full Safety Checks"""
    Ast = n * (np.pi * (db/2)**2)
    if Ast == 0: return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Whitney Stress Block
    a = (Ast * fy) / (0.85 * fc * b)
    beta1 = get_beta1(fc)
    c = a / beta1
    
    # Check: Over-reinforced (a >= d)
    if a >= d_eff: 
        return 0.0, Ast, a, 0.0, c, -1.0 

    # Strain
    if c > 0:
        strain_t = 0.003 * (d_eff - c) / c
    else:
        strain_t = 999.0

    # Phi Factor
    if strain_t >= 0.005:
        phi = 0.90
    elif strain_t <= 0.002:
        phi = 0.65
    else:
        phi = 0.65 + 0.25 * ((strain_t - 0.002) / 0.003)

    Mn = Ast * fy * (d_eff - a/2)
    phi_Mn = phi * Mn / 1e6 # kNm
    
    return phi_Mn, Ast, a, Mn, c, strain_t

def check_shear_details(Vu_kN, b, d, fc, fy, stir_db, spacing):
    """Check Shear Capacity and Spacing"""
    if d <= 0: return "FAIL (Invalid d)", 0,0,0,0,0
    
    Vu = abs(Vu_kN) * 1000 # N
    
    # Vc
    Vc = 0.17 * np.sqrt(fc) * b * d
    phi = 0.85
    phi_Vc = phi * Vc
    
    # Vs
    Av = 2 * (np.pi * (stir_db/2)**2)
    if spacing <= 0: spacing = 1000
    
    Vs = (Av * fy * d) / spacing
    phi_Vs = phi * Vs
    
    phi_Vn = phi_Vc + phi_Vs
    
    # Max Spacing Check
    threshold = 0.33 * np.sqrt(fc) * b * d
    if Vs <= threshold:
        s_max_limit = min(d/2, 600)
    else:
        s_max_limit = min(d/4, 300)
        
    is_strength_ok = phi_Vn >= Vu
    is_spacing_ok = spacing <= s_max_limit
    
    if not is_strength_ok:
        status = "FAIL (Strength)"
    elif not is_spacing_ok:
        status = f"FAIL (Space > {s_max_limit:.0f})"
    else:
        status = "OK"

    return status, phi_Vn/1000, phi_Vc/1000, phi_Vs/1000, Vc, Vs
