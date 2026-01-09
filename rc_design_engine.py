# rc_design_engine.py
import numpy as np
from rc_utils import get_beta1

def get_as_req(Mu_kNm, d_eff_mm, fc, fy, b_mm):
    """
    Calculate Required Steel Area (As) based on ACI 318
    Returns: 3 values (max_as, rho, is_fail)
    Units: Mu [kN-m], d_eff [mm], fc/fy [MPa], b [mm]
    """
    if Mu_kNm == 0: return 0.0, 0.0, False
    Mu = abs(Mu_kNm) * 1e6 # Convert kN-m to N-mm
    phi = 0.9 
    
    # Calculate Coefficient of Resistance (Rn) [N/mm2]
    Rn = Mu / (phi * b_mm * d_eff_mm**2)
    
    # Check Section Capacity
    term_inside = 1 - (2 * Rn) / (0.85 * fc)
    if term_inside < 0:
        return 0.0, 0.0, True # Section too small (Fail)

    # Reinforcement Ratio (rho)
    rho = (0.85 * fc / fy) * (1 - np.sqrt(term_inside))
    as_req = rho * b_mm * d_eff_mm # [mm2]
    
    # Minimum Steel (As_min) - ACI 9.6.1.2 [mm2]
    as_min1 = (0.25 * np.sqrt(fc) / fy) * b_mm * d_eff_mm
    as_min2 = (1.4 / fy) * b_mm * d_eff_mm
    as_min = max(as_min1, as_min2)
    
    # Compare As_req vs As_min
    as_final = max(as_req, as_min)
    
    return as_final, rho, False

def get_phi_Mn_details(n, db, d_eff, b, fc, fy):
    """
    Calculate Moment Capacity (Phi Mn) with Full Safety Checks
    Returns: 6 values (phi_Mn, Ast, a, Mn, c, strain_t)
    Units: phi_Mn [kN-m], Ast [mm2], a/c [mm], Mn [N-mm]
    """
    # Total Steel Area Provided [mm2]
    Ast = n * (np.pi * (db/2)**2)
    if Ast == 0: return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    # 1. Whitney Stress Block Depth (a) [mm]
    a = (Ast * fy) / (0.85 * fc * b)
    beta1 = get_beta1(fc)
    c = a / beta1 # Neutral axis depth [mm]
    
    # --- CRITICAL SAFETY CHECK: a >= d ---
    if a >= d_eff: 
        return 0.0, Ast, a, 0.0, c, -1.0 

    # 2. Strain Calculation at Extreme Tension Steel
    if c > 0:
        strain_t = 0.003 * (d_eff - c) / c
    else:
        strain_t = 999.0 

    # 3. Strength Reduction Factor (phi) - ACI 318
    if strain_t >= 0.005:
        phi = 0.90 # Tension Controlled
    elif strain_t <= 0.002:
        phi = 0.65 # Compression Controlled
    else:
        # Transition zone
        phi = 0.65 + 0.25 * ((strain_t - 0.002) / 0.003)

    # 4. Design Moment Capacity (phi_Mn) [kN-m]
    Mn = Ast * fy * (d_eff - a/2) # [N-mm]
    phi_Mn = phi * Mn / 1e6 # Convert to kN-m
    
    return phi_Mn, Ast, a, Mn, c, strain_t

def check_shear_details(Vu_kN, b, d, fc, fy, stir_db, spacing):
    """
    Check Shear Capacity and ACI Max Spacing Requirements
    Returns: 6 values (status, phi_Vn, phi_Vc, phi_Vs, Vc, Vs)
    Units: Vu/phi_Vn/phi_Vc/phi_Vs [kN], Vc/Vs [N]
    """
    if d <= 0: return "FAIL (Invalid d)", 0, 0, 0, 0, 0
    
    Vu = abs(Vu_kN) * 1000 # Convert kN to N
    phi = 0.75 # ACI 318-19 Shear Phi Factor
    
    # 1. Concrete Shear Strength (Vc) [N]
    Vc = 0.17 * np.sqrt(fc) * b * d
    phi_Vc = phi * Vc
    
    # 2. Steel Shear Strength (Vs) [N]
    Av = 2 * (np.pi * (stir_db/2)**2) # 2 legs [mm2]
    if spacing <= 0: spacing = 1000 # Safety factor
    
    Vs = (Av * fy * d) / spacing
    phi_Vs = phi * Vs
    
    # Total Design Shear Strength (phi_Vn) [kN]
    phi_Vn_total = (phi_Vc + phi_Vs) / 1000
    
    # 3. Maximum Spacing Check (ACI 318)
    threshold = 0.33 * np.sqrt(fc) * b * d
    if Vs <= threshold:
        s_max_limit = min(d/2, 600)
    else:
        s_max_limit = min(d/4, 300)
        
    # Final Evaluation (Strength & Spacing)
    is_strength_ok = (phi_Vn_total * 1000) >= Vu
    is_spacing_ok = spacing <= s_max_limit
    
    if not is_strength_ok:
        status = f"FAIL (Vu={abs(Vu_kN):.1f} > φVn={phi_Vn_total:.1f} kN)"
    elif not is_spacing_ok:
        status = f"FAIL (s={spacing:.0f} > s_max={s_max_limit:.0f} mm)"
    else:
        status = "OK"

    return status, phi_Vn_total, phi_Vc/1000, phi_Vs/1000, Vc, Vs
