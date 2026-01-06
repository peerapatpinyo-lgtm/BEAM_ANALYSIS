import numpy as np

def design_span_detailed(mu_pos, mu_neg, vu, b_m, h_m, fc, fy, fyt, cover_mm, db_main, db_stirrup):
    phi_m, phi_v = 0.90, 0.75
    b, h = b_m * 1000, h_m * 1000
    d = h - cover_mm - db_stirrup - (db_main / 2)
    
    def calc_flexure(mu_knm):
        if abs(mu_knm) < 0.1: 
            return {"n": 2, "as_req": 0, "k": 0, "rho": 0, "a": 0, "c": 0, "et": 0.005, "status": "Min Steel"}
        
        mu_n = (abs(mu_knm) * 1e6) / phi_m
        k = mu_n / (b * d**2)
        m = fy / (0.85 * fc)
        check_val = 1 - (2 * m * k / fy)
        
        if check_val < 0: return {"status": "FAIL: SECTION TOO SMALL"}
            
        rho = (1/m) * (1 - np.sqrt(check_val))
        rho_min = max(0.25 * np.sqrt(fc) / fy, 1.4 / fy)
        rho_final = max(rho, rho_min)
        as_req = rho_final * b * d
        
        # Engineering parameters
        a = (as_req * fy) / (0.85 * fc * b)
        beta1 = max(0.65, 0.85 - 0.05 * (fc - 28) / 7)
        c = a / beta1
        et = ((d - c) / c) * 0.003 # Tension strain
        
        n_bars = max(2, int(np.ceil(as_req / (np.pi * (db_main**2) / 4))))
        return {"n": n_bars, "as_req": as_req, "k": k, "rho": rho_final, "a": a, "c": c, "et": et, "status": "OK"}

    res_pos = calc_flexure(mu_pos)
    res_neg = calc_flexure(mu_neg)
    
    # Shear Design
    vc = (1/6) * np.sqrt(fc) * b * d / 1000
    phi_vc = phi_v * vc
    vs_req = (vu / phi_v) - vc if vu > (phi_vc * 0.5) else 0
    asv = 2 * (np.pi * (db_stirrup**2) / 4)
    s = min(d/2, 300)
    if vs_req > 0:
        s = min((asv * fyt * d) / (vs_req * 1000), s)
        
    return {
        "pos": res_pos, "neg": res_neg, "spacing": int(s), 
        "d": d, "b": b, "h": h, "mu_pos": mu_pos, "mu_neg": mu_neg, "vu": vu, "vc": vc
    }
