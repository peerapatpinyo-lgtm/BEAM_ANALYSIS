import numpy as np

def design_span_expert(mu_pos, mu_neg, vu, b_m, h_m, fc, fy, fyt, cover_mm, db_main, db_stirrup):
    phi_m, phi_v = 0.90, 0.75
    b, h = b_m * 1000, h_m * 1000
    d = h - cover_mm - db_stirrup - (db_main / 2)
    
    def calc_flexure(mu_knm):
        if abs(mu_knm) < 0.1: 
            return {"n": 2, "as_req": 0, "status": "Min Steel", "et": 0.005, "a": 0, "k": 0}
        
        mu_n = (abs(mu_knm) * 1e6) / phi_m
        k = mu_n / (b * d**2)
        m = fy / (0.85 * fc)
        
        check_val = 1 - (2 * m * k / fy)
        if check_val < 0: return {"status": "FAIL: OVER-REINFORCED", "n": 0, "as_req": 0, "a": 0, "et": 0, "k": k}
            
        rho = (1/m) * (1 - np.sqrt(max(0, check_val)))
        rho_min = max(0.25 * np.sqrt(fc) / fy, 1.4 / fy)
        rho_final = max(rho, rho_min)
        as_req = rho_final * b * d
        
        a = (as_req * fy) / (0.85 * fc * b)
        beta1 = max(0.65, 0.85 - 0.05 * (fc - 28) / 7)
        c = a / beta1
        et = ((d - c) / c) * 0.003 
        
        n_bars = max(2, int(np.ceil(as_req / (np.pi * (db_main**2) / 4))))
        return {"n": n_bars, "as_req": as_req, "k": k, "a": a, "et": et, "status": "OK", "mu_val": mu_knm}

    # Shear Calculation
    vc = (0.17 * np.sqrt(fc) * b * d) / 1000 # Concrete Shear Strength (kN)
    phi_vc = phi_v * vc
    vs_req = (vu / phi_v) - vc if vu > (phi_vc * 0.5) else 0
    s_max = min(d/2, 300)
    
    return {
        "pos": calc_flexure(mu_pos),
        "neg": calc_flexure(mu_neg),
        "spacing": int(s_max), 
        "d": d, "vu": vu, "phi_vc": phi_vc
    }
