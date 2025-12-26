import numpy as np

def get_default_factors(code_name):
    if "WSD" in code_name: return 1.0, 1.0, "WSD"
    elif "ACI" in code_name: return 1.2, 1.6, "SDM"
    else: return 1.4, 1.7, "SDM"

def calculate_rc_design(max_M, max_V, fc, fy, b, h, cov, method, unit_sys, main_bar_area, stirrup_area):
    # Convert Units to MKS (kg, cm)
    if "kN" in unit_sys:
        Mu_calc = max_M * 1000 * 100 / 9.80665
        Vu_calc = max_V * 1000 / 9.80665
        fc_c, fy_c = fc * 10.197, fy * 10.197
    else:
        Mu_calc, Vu_calc = max_M * 100, max_V
        fc_c, fy_c = fc, fy

    d = h - cov - 0.9
    result = {}

    # Flexure
    if method == "SDM":
        phi_b = 0.9
        Rn = Mu_calc / (phi_b * b * d**2)
        term = 1 - (2*Rn)/(0.85*fc_c)
        if term < 0:
            result.update({'As_req': 9999, 'msg_flex': "❌ Section too small"})
        else:
            rho = (0.85*fc_c/fy_c) * (1 - np.sqrt(term))
            As_req = max(rho, 14/fy_c) * b * d
            result.update({'As_req': As_req, 'msg_flex': "✅ Moment OK"})
    else:
        As_req = Mu_calc / (0.5 * fy_c * 0.875 * d)
        result.update({'As_req': As_req, 'msg_flex': "✅ Moment OK (WSD)"})

    result['nb'] = 0 if result['As_req'] == 9999 else max(2, int(np.ceil(result['As_req'] / main_bar_area)))

    # Shear
    Vc = 0.53 * np.sqrt(fc_c) * b * d
    if method == "SDM": Vc *= 0.85
    
    if Vu_calc > Vc:
        Vs_req = Vu_calc - Vc
        s_req = (2 * stirrup_area * fy_c * d) / Vs_req
        s = int(5 * round(min(s_req, d/2, 60)/5)) or 5
        result.update({'stirrup_text': f"@{s} cm", 'msg_shear': "⚠️ Shear Reinf."})
    else:
        result.update({'stirrup_text': f"@{int(d/2)} cm (Min)", 'msg_shear': "✅ Min Shear"})
        
    return result
