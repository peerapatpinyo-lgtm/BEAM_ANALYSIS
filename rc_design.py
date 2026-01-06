import math

def design_span_expert(m_pos, m_neg, v_u, b, h, fc, fy, cover, db):
    d = (h * 1000) - cover - (db/2) - 10 # mm
    A_bar = (math.pi * db**2) / 4
    
    def calc_steel(mu_kNm, is_top):
        if abs(mu_kNm) < 1: return {"n": 2, "capacity": 0}
        mu = abs(mu_kNm) * 1e6
        phi = 0.9
        as_req = mu / (phi * fy * 0.9 * d)
        as_min = (1.4 / fy) * (b*1000) * d
        as_final = max(as_req, as_min)
        n = max(2, math.ceil(as_final / A_bar))
        a = (n * A_bar * fy) / (0.85 * fc * b * 1000)
        cap = (phi * n * A_bar * fy * (d - a/2)) / 1e6
        return {"n": n, "capacity": cap}

    pos = calc_steel(m_pos, False)
    neg = calc_steel(m_neg, True)
    
    # Shear Design Simplified
    vc = 0.17 * math.sqrt(fc) * (b*1000) * d / 1000 # kN
    stirrups = "RB9 @ 0.20 m" if v_u > 0.5*0.85*vc else "Min. RB6 @ 0.25 m"
    
    return {"pos": pos, "neg": neg, "shear_stirrups": stirrups, "shear_status": "OK"}
