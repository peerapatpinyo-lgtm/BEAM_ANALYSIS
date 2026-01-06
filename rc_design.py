import numpy as np

def calculate_advanced_rc(mu_pos, mu_neg, vu, b_m, h_m, fc, fy):
    phi_m, phi_v = 0.90, 0.75
    b, h = b_m * 1000, h_m * 1000  # mm
    d = h - 50  # mm
    report = []
    optimize_msg = ""

    def calc_as(mu_knm):
        if abs(mu_knm) < 0.1: return 0, 0, "OK"
        mu_n_mm = (abs(mu_knm) * 1e6) / phi_m
        rn = mu_n_mm / (b * d**2)
        m = fy / (0.85 * fc)
        
        # Check for Section Size Optimization
        # rho_ideal ควรอยู่ช่วง 0.5% - 1.2% สำหรับหน้าตัดที่ประหยัดและปลอดภัย
        rho = (1/m) * (1 - np.sqrt(max(0, 1 - (2 * m * rn / fy))))
        rho_min = max(0.25 * np.sqrt(fc) / fy, 1.4 / fy)
        rho_final = max(rho, rho_min)
        
        status = "OK"
        if rn > (0.25 * fc): status = "TOO_SMALL"
        elif rho_final < rho_min: status = "TOO_LARGE"
        
        return rho_final * b * d, rho_final, status

    as_bot, rho_bot, stat_bot = calc_as(mu_pos)
    as_top, rho_top, stat_top = calc_as(mu_neg)

    # --- Automatic Section Optimization Logic ---
    if stat_bot == "TOO_SMALL" or stat_top == "TOO_SMALL":
        new_h = np.sqrt((max(abs(mu_pos), abs(mu_neg)) * 1e6 / phi_m) / (0.15 * fc * b)) + 50
        optimize_msg = f"⚠️ **หน้าตัดเล็กเกินไป:** แนะนำให้เพิ่มความหนา (h) เป็นอย่างน้อย {int(np.ceil(new_h/50)*50)} mm"
    elif rho_bot < 0.003 and rho_top < 0.003:
        optimize_msg = "💡 **หน้าตัดใหญ่เกินความจำเป็น:** สามารถลดขนาดหน้าตัดเพื่อประหยัดงบประมาณได้"
    else:
        optimize_msg = "✅ **หน้าตัดมีความเหมาะสม:** ปริมาณเหล็กเสริมอยู่ในช่วงที่ประหยัด (Economical Design)"

    # Shear
    vc = (1/6) * np.sqrt(fc) * b * d / 1000
    vs_req = (vu / phi_v) - vc if vu > (phi_v * vc * 0.5) else 0
    spacing = min(200, d/2)
    if vs_req > 0:
        spacing = min((127 * fy * d) / (vs_req * 1000), d/2, 600)

    report.append(f"**Section Analysis:**")
    report.append(f"- Bottom Steel: {as_bot:.0f} mm² (ρ={rho_bot:.4f})")
    report.append(f"- Top Steel: {as_top:.0f} mm² (ρ={rho_top:.4f})")
    report.append(f"- Stirrup: RB9 @ {int(spacing)} mm")
    
    return {
        'as_top': as_top, 'as_bot': as_bot, 'spacing': spacing,
        'report': report, 'optimize_msg': optimize_msg
    }
