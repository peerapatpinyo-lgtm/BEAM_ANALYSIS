import numpy as np

def calculate_advanced_rc(mu_pos, mu_neg, vu, b_m, h_m, fc, fy):
    phi_m, phi_v = 0.90, 0.75
    b, h = b_m * 1000, h_m * 1000  # mm
    d = h - 50  # mm
    report = []
    
    # --- Optimization Parameters ---
    # ค่า k_max สำหรับ Tension Controlled (โดยประมาณ)
    k_max = 0.20 * fc  # ขีดจำกัดที่คานยังประหยัดและปลอดภัย
    k_min = 0.05 * fc  # ขีดจำกัดที่คานเริ่มใหญ่เกินไป
    
    def calc_steel(mu_knm, label):
        if abs(mu_knm) < 0.1: return 0, 0, 0, "Minimal"
        mu_n_mm = (abs(mu_knm) * 1e6) / phi_m
        k = mu_n_mm / (b * d**2)
        m = fy / (0.85 * fc)
        
        check_val = 1 - (2 * m * k / fy)
        if check_val < 0: return 0, k, 0, "CRITICAL"
        
        rho = (1/m) * (1 - np.sqrt(check_val))
        rho_min = max(0.25 * np.sqrt(fc) / fy, 1.4 / fy)
        rho_final = max(rho, rho_min)
        as_req = rho_final * b * d
        
        # คำนวณจำนวนเส้น (ใช้ DB20 พื้นที่ 314 mm2)
        n_bars = max(2, int(np.ceil(as_req / 314)))
        return as_req, k, n_bars, "OK"

    as_bot, k_bot, n_bot, stat_bot = calc_steel(mu_pos, "Bottom")
    as_top, k_top, n_top, stat_top = calc_steel(mu_neg, "Top")

    # --- Optimization Report ---
    max_k = max(k_bot, k_top)
    if stat_bot == "CRITICAL" or stat_top == "CRITICAL":
        opt_status = "🔴 OVER-REINFORCED"
        opt_desc = f"หน้าตัดเล็กเกินไป (k = {max_k:.2f} > {k_max:.2f}). คอนกรีตจะระเบิดก่อนเหล็กสละตัว"
        suggestion = f"เพิ่มความหนาคาน (h) เป็น {int(h+100)} mm"
    elif max_k > k_max:
        opt_status = "🟡 UNECONOMICAL"
        opt_desc = f"หน้าตัดค่อนข้างเล็ก (k = {max_k:.2f}). ต้องใช้เหล็กเสริมจำนวนมาก"
        suggestion = "ควรเพิ่มหน้าตัดเล็กน้อยเพื่อลดปริมาณเหล็ก"
    elif max_k < k_min:
        opt_status = "🔵 OVER-SIZED"
        opt_desc = f"หน้าตัดใหญ่เกินไป (k = {max_k:.2f} < {k_min:.2f}). สิ้นเปลืองคอนกรีต"
        suggestion = "สามารถลดขนาดหน้าตัดเพื่อประหยัดต้นทุน"
    else:
        opt_status = "🟢 OPTIMIZED"
        opt_desc = "หน้าตัดมีความสมดุลระหว่างคอนกรีตและเหล็กเสริม"
        suggestion = "คงขนาดหน้าตัดนี้ไว้"

    # Shear Stirrups
    vc = (1/6) * np.sqrt(fc) * b * d / 1000
    vs_req = (vu / phi_v) - vc if vu > (phi_v * vc * 0.5) else 0
    spacing = min(200, d/2)
    if vs_req > 0:
        spacing = min((127 * fy * d) / (vs_req * 1000), d/2, 600)

    return {
        'as_bot': as_bot, 'n_bot': n_bot,
        'as_top': as_top, 'n_top': n_top,
        'spacing': spacing,
        'opt_status': opt_status, 'opt_desc': opt_desc, 'suggestion': suggestion,
        'b': b, 'h': h
    }
