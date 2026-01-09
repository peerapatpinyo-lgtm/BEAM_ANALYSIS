# rc_design_engine.py
import numpy as np
from rc_utils import get_beta1

def get_as_req(Mu_kNm, d_eff_mm, fc, fy, b_mm):
    """
    Calculate Required Steel Area based on ACI 318
    Returns: 3 values -> (as_final, rho, is_fail)
    """
    if Mu_kNm == 0: return 0.0, 0.0, False
    Mu = abs(Mu_kNm) * 1e6 # N-mm
    phi = 0.9 
    
    # Check Min/Max is usually done after, but here we calculate pure required As
    Rn = Mu / (phi * b_mm * d_eff_mm**2)
    
    # Formula: rho = (0.85*fc/fy) * [1 - sqrt(1 - 2*Rn / (0.85*fc))]
    term_inside = 1 - (2 * Rn) / (0.85 * fc)
    
    if term_inside < 0:
        return 0.0, 0.0, True # Section too small (Fail)

    rho = (0.85 * fc / fy) * (1 - np.sqrt(term_inside))
    as_req = rho * b_mm * d_eff_mm
    
    # Minimum Steel (ACI 9.6.1.2) [mm2]
    as_min1 = (0.25 * np.sqrt(fc) / fy) * b_mm * d_eff_mm
    as_min2 = (1.4 / fy) * b_mm * d_eff_mm
    as_min = max(as_min1, as_min2)
    
    # Compare As_req vs As_min
    as_final = max(as_req, as_min)
    
    return as_final, rho, False

def get_phi_Mn_details(n, db, d_eff, b, fc, fy):
    """
    Calculate Moment Capacity (Phi Mn) with Full Safety Checks
    Returns: 6 values -> (phi_Mn, Ast, a, Mn, c, strain_t)
    """
    Ast = n * (np.pi * (db/2)**2)
    if Ast == 0: return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    # 1. Whitney Stress Block
    a = (Ast * fy) / (0.85 * fc * b)
    beta1 = get_beta1(fc)
    c = a / beta1
    
    # --- CRITICAL SAFETY CHECK: a >= d ---
    if a >= d_eff: 
        return 0.0, Ast, a, 0.0, c, -1.0 

    # 2. Strain Calculation
    if c > 0:
        strain_t = 0.003 * (d_eff - c) / c
    else:
        strain_t = 999.0 

    # 3. Phi Factor Calculation (ACI 318)
    if strain_t >= 0.005:
        phi = 0.90
    elif strain_t <= 0.002:
        phi = 0.65
    else:
        phi = 0.65 + 0.25 * ((strain_t - 0.002) / 0.003)

    # 4. Nominal Moment (Mn)
    Mn = Ast * fy * (d_eff - a/2)
    phi_Mn = phi * Mn / 1e6 # Convert to kNm
    
    return phi_Mn, Ast, a, Mn, c, strain_t

def check_shear_details(Vu_kN, b, d, fc, fy, stir_db, spacing):
    """
    Check Shear Capacity and ACI Max Spacing Requirements
    Returns: 6 values -> (status, phi_Vn, phi_Vc, phi_Vs, Vc, Vs)
    """
    if d <= 0: return "FAIL (Invalid d)", 0,0,0,0,0
    
    Vu = abs(Vu_kN) * 1000 # Convert kN to N
    phi = 0.75 # ACI 318 Shear Phi
    
    # 1. Concrete Capacity (Vc)
    Vc = 0.17 * np.sqrt(fc) * b * d
    phi_Vc = phi * Vc
    
    # 2. Steel Capacity (Vs)
    Av = 2 * (np.pi * (stir_db/2)**2) # 2 legs
    if spacing <= 0: spacing = 1000 
    
    Vs = (Av * fy * d) / spacing
    phi_Vs = phi * Vs
    
    phi_Vn = (phi_Vc + phi_Vs) / 1000 # Convert to kN
    
    # 3. Maximum Spacing Check (ACI 318)
    threshold = 0.33 * np.sqrt(fc) * b * d
    if Vs <= threshold:
        s_max_limit = min(d/2, 600)
    else:
        s_max_limit = min(d/4, 300)
        
    # Evaluation with units
    is_strength_ok = (phi_Vn * 1000) >= Vu
    is_spacing_ok = spacing <= s_max_limit
    
    if not is_strength_ok:
        status = f"FAIL (Vu={abs(Vu_kN):.1f} > φVn={phi_Vn:.1f} kN)"
    elif not is_spacing_ok:
        status = f"FAIL (s={spacing:.0f} > s_max={s_max_limit:.0f} mm)"
    else:
        status = "OK"

    return status, phi_Vn, phi_Vc/1000, phi_Vs/1000, Vc, Vs
