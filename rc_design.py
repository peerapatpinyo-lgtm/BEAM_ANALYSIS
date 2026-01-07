import math

def design_section(Mu_kNm, Vu_kN, b_mm, h_mm, fc, fy):
    logs = []
    phi_m = 0.9
    phi_v = 0.85
    cover = 40
    db = 16
    d = h_mm - cover - 9 - (db/2)
    
    # Factor Loads
    Mu = abs(Mu_kNm) * 1.2 * 1e6  # N-mm
    Vu = abs(Vu_kN) * 1.6 * 1000   # N
    
    m = fy / (0.85 * fc)
    Rn = Mu / (phi_m * b_mm * d**2)
    
    logs.append("### RC Design (Timoshenko Analysis Basis)")
    logs.append(f"- Factor $R_n = {Rn:.3f}$ MPa")

    # Check for over-reinforced
    if Rn > (0.85 * fc * 0.2):
        return None, ["❌ Error: Section too small! Concrete crushed."]

    # Steel Calculation
    rho = (1/m) * (1 - math.sqrt(1 - (2 * m * Rn)/fy))
    As_min = max(0.25 * math.sqrt(fc) / fy, 1.4 / fy) * b_mm * d
    As_final = max(rho * b_mm * d, As_min)
    
    n_bars = max(2, math.ceil(As_final / (math.pi * (db/2)**2)))
    
    logs.append(f"- $A_{{s,req}} = {rho * b_mm * d:.1f}$ mm²")
    logs.append(f"- $A_{{s,min}} = {As_min:.1f}$ mm²")
    logs.append(f"✅ **Select {n_bars}-DB{db}**")
    
    # Shear Design
    Vc = (0.17 * math.sqrt(fc) * b_mm * d)
    if Vu > (phi_v * Vc):
        stirrup = "RB9 @ 0.15 m"
    else:
        stirrup = "RB6 @ 0.20 m (Min)"
    logs.append(f"✅ **Stirrup: {stirrup}**")

    return {"n_bars": n_bars, "stirrup": stirrup}, logs
