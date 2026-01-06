import numpy as np

def design_span_expert(mu_pos, mu_neg, vu, b_m, h_m, fc, fy, cover_mm, db_main):
    phi_m, phi_v = 0.90, 0.75
    b, h = b_m * 1000, h_m * 1000
    db_stirrup = 9 # สมมติเหล็กปลอก 9mm
    d = h - cover_mm - db_stirrup - (db_main / 2)
    
    def calc_flexure(mu_knm):
        if abs(mu_knm) < 0.5: # แรงน้อยมาก ให้ใส่เหล็กขั้นต่ำ
            rho_min = max(0.25 * np.sqrt(fc) / fy, 1.4 / fy)
            as_req = rho_min * b * d
            return {"n": 2, "as_req": as_req, "status": "Minimum Steel", "et": 0.005}
        
        mu_n = (abs(mu_knm) * 1e6) / phi_m
        rn = mu_n / (b * d**2)
        m = fy / (0.85 * fc)
        
        # Check if section can handle the load
        if rn > (0.85 * fc * (0.375 * 0.85)): # Simplified limit for tension-controlled
            return {"n": 0, "status": "OVER-REINFORCED: Increase Section", "et": 0}
            
        rho = (1/m) * (1 - np.sqrt(1 - (2 * m * rn / fy)))
        rho_min = max(0.25 * np.sqrt(fc) / fy, 1.4 / fy)
        rho_final = max(rho, rho_min)
        as_req = rho_final * b * d
        
        # Strain check
        a = (as_req * fy) / (0.85 * fc * b)
        c = a / 0.85
        et = ((d - c) / c) * 0.003
        
        n_bars = max(2, int(np.ceil(as_req / (np.pi * (db_main**2) / 4))))
        return {"n": n_bars, "as_req": as_req, "status": "OK", "et": et}

    res_pos = calc_flexure(mu_pos)
    res_neg = calc_flexure(mu_neg)
    vc = (0.17 * np.sqrt(fc) * b * d) / 1000
    
    return {
        "pos": res_pos, "neg": res_neg, "vu": vu, "phi_vc": phi_v * vc,
        "b": b, "h": h, "d": d, "mu_pos": mu_pos, "mu_neg": mu_neg
    }
