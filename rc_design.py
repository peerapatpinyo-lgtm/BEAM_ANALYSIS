import math

def get_beta1(fc):
    if fc <= 28: return 0.85
    if fc >= 55: return 0.65
    return 0.85 - 0.05 * (fc - 28) / 7

def design_section(Mu_kNm, Vu_kN, b_mm, h_mm, fc, fy):
    logs = []
    # 1. Setup
    phi_m = 0.90
    phi_v = 0.85
    cover = 40
    db = 16
    d = h_mm - cover - 9 - (db/2) # 9mm stirrup
    
    Mu = abs(Mu_kNm) * 1.2 * 1e6 # Factor 1.2 for Mu
    Vu = abs(Vu_kN) * 1.6 * 1000 # Factor 1.6 for Vu
    
    beta1 = get_beta1(fc)
    m = fy / (0.85 * fc)
    Rn = Mu / (phi_m * b_mm * d**2)
    
    logs.append(f"### FLEXURAL DESIGN REPORT")
    logs.append(f"- Factor $m = {m:.2f}$, $R_n = {Rn:.3f}$ MPa")
    
    # 2. Section Check
    check_val = 1 - (2 * m * Rn) / fy
    if check_val < 0:
        d_req = math.sqrt(Mu / (phi_m * b_mm * 0.2 * fc))
        logs.append(f"❌ **Error: Section too small!**")
        logs.append(f"💡 Suggestion: Increase Height to at least {d_req+60:.0f} mm")
        return None, logs

    # 3. Steel Calculation
    rho_req = (1/m) * (1 - math.sqrt(check_val))
    As_min = max(0.25 * math.sqrt(fc) / fy, 1.4 / fy) * b_mm * d
    As_final = max(rho_req * b_mm * d, As_min)
    
    n_bars = max(2, math.ceil(As_final / (math.pi * (db/2)**2)))
    
    logs.append(f"- $A_{{s,req}} = {rho_req * b_mm * d:.1f}$ mm²")
    logs.append(f"- $A_{{s,min}} = {As_min:.1f}$ mm²")
    logs.append(f"👉 **Select {n_bars}-DB16**")

    # 4. Shear Design
    Vc = (0.17 * math.sqrt(fc) * b_mm * d) / 1000 # kN
    logs.append(f"### SHEAR DESIGN REPORT")
    logs.append(f"- $\phi V_c = {phi_v * Vc:.2f}$ kN vs $V_u = {Vu/1000:.2f}$ kN")
    
    if Vu <= (phi_v * Vc / 2):
        stirrup = "RB6 @ 0.20 m (Min)"
    elif Vu <= (phi_v * Vc):
        stirrup = "RB6 @ 0.15 m (Min)"
    else:
        # Simplified Stirrup calculation
        Vs = (Vu/phi_v) - (Vc*1000)
        Av = 2 * (math.pi * (9/2)**2) # RB9 2-legs
        s = (Av * 240 * d) / Vs
        s = min(s, d/2, 300)
        stirrup = f"RB9 @ {math.floor(s/10)*10/1000:.2f} m"
    
    logs.append(f"👉 **Stirrup: {stirrup}**")
    
    return {"n_bars": n_bars, "stirrup": stirrup}, logs
