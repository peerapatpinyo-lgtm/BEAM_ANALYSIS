import numpy as np

def design_beam_flexure(Mu, b, d, fc, fy, phi=0.9):
    """
    Calculates required steel Area (As) for singly reinforced beam.
    Units: Mu (kNm), b, d (m), fc, fy (MPa)
    Returns: As_req (mm^2), rho, status_dict
    """
    Mu_Nmm = Mu * 1e6
    b_mm = b * 1000
    d_mm = d * 1000
    
    # 1. Check Max Capacity (Steel Yielding Limit)
    # beta1 calculation
    if fc <= 30: beta1 = 0.85
    elif fc >= 55: beta1 = 0.65
    else: beta1 = 0.85 - 0.05 * (fc - 30) / 7
    
    rho_min = max(0.25 * np.sqrt(fc) / fy, 1.4 / fy)
    rho_b = 0.85 * beta1 * (fc / fy) * (600 / (600 + fy))
    rho_max = 0.75 * rho_b # Common practical limit
    
    # 2. Solve for Required Rho (Iterative or Quadratic)
    # Mn = As * fy * (d - a/2)
    # Mu/phi = rho * b * d * fy * (d - 0.59 * rho * d * fy / fc)
    # Rn = Mu / (phi * b * d^2)
    Rn = Mu_Nmm / (phi * b_mm * d_mm**2)
    
    try:
        rho_req = (0.85 * fc / fy) * (1 - np.sqrt(1 - (2 * Rn) / (0.85 * fc)))
    except:
        return 0, 0, {"status": "Fail", "msg": "Section too small (Compression Fail)"}
        
    As_req = rho_req * b_mm * d_mm
    
    status = "OK"
    msg = "Design Pass"
    if rho_req < rho_min:
        rho_req = rho_min
        As_req = rho_min * b_mm * d_mm
        msg = "Used Min Steel"
    elif rho_req > rho_max:
        status = "Warning"
        msg = "High Steel (Exceeds rho_max)"
        
    return As_req, rho_req, {"status": status, "msg": msg}

def check_shear(Vu, b, d, fc, fy, phi=0.85):
    """
    Returns required stirrup spacing.
    Units: Vu (kN), b, d (m)
    """
    Vu_N = Vu * 1000
    b_mm = b * 1000
    d_mm = d * 1000
    
    Vc = 0.17 * np.sqrt(fc) * b_mm * d_mm
    phi_Vc = phi * Vc
    
    req_s = None
    status = "OK"
    
    if Vu_N <= phi_Vc / 2:
        status = "No Shear Reinforcement Needed"
        req_s = 600 # Max spacing
    elif Vu_N <= phi_Vc:
        status = "Min Shear Reinforcement"
        # Av min check
        req_s = 300 # Example max spacing
    else:
        # Vs needed
        Vs = (Vu_N - phi_Vc) / phi
        # Try RB6 (2 legs) -> Av = 2 * 28 = 56 mm2
        Av = 56.5
        s_req = (Av * fy * d_mm) / Vs
        req_s = s_req
        
        if Vs > 0.66 * np.sqrt(fc) * b_mm * d_mm:
            status = "Fail (Section too small for Shear)"
            req_s = 0
            
    return req_s, status
