import numpy as np
import math

def get_rebar_area(size_str):
    """Return area (cm2) based on standard Thai sizes"""
    db_map = {
        'RB6': 0.28, 'RB9': 0.64,
        'DB12': 1.13, 'DB16': 2.01, 'DB20': 3.14, 'DB25': 4.91, 'DB28': 6.16
    }
    return db_map.get(size_str, 2.01)

def calculate_flexure_sdm(Mu_kgm, section_name, b, h, cover, params):
    """
    Strength Design Method (SDM) Calculation with detailed logging.
    """
    logs = []
    logs.append(f"<b>--- Design for {section_name} ---</b>")
    
    # 1. Constants
    phi = 0.90
    fc = params['fc']
    fy = params['fy']
    Mu = abs(Mu_kgm) * 100 # Convert kg-m -> kg-cm
    
    logs.append(f"Mu = {Mu_kgm:.2f} kg-m")
    logs.append(f"Section {b}x{h} cm, fc'={fc} ksc, fy={fy} ksc")

    # 2. Effective Depth (d)
    # Estimate: cover + stirrup(0.9) + half_bar(1.0) approx 4-5 cm
    d_est = h - cover - 2.0 
    logs.append(f"Effective depth (d) ≈ {d_est:.2f} cm")
    
    if Mu <= 1e-3:
        return {'Type': section_name, 'Mu': 0, 'Bars': "Min Steel", 'Status': 'OK', 'Log': logs + ["Moment is negligible."]}

    # 3. Calculate Rn
    # Mn_req = Mu / phi
    Mn_req = Mu / phi
    Rn = Mn_req / (b * d_est**2)
    logs.append(f"Required Mn = {Mn_req:,.2f} kg-cm")
    logs.append(f"Rn = {Rn:.2f} ksc")
    
    # 4. Check Rho Required
    # rho = (1/m) * (1 - sqrt(1 - 2*m*Rn/fy))
    beta1 = 0.85 if fc <= 280 else max(0.65, 0.85 - 0.05*((fc-280)/70))
    m = fy / (0.85 * fc)
    
    term = 1 - (2 * m * Rn / fy)
    if term < 0:
        logs.append(f"<span style='color:red'>Error: Section too small (Rn too high). Increase Depth.</span>")
        return {'Type': section_name, 'Mu': Mu_kgm, 'Bars': "SIZE ERR", 'Status': 'Fail', 'Log': logs}
        
    rho_req = (1/m) * (1 - math.sqrt(term))
    logs.append(f"Rho required = {rho_req:.5f}")
    
    # 5. Check Limits (Rho_min, Rho_max)
    rho_min = max(14/fy, 0.25*math.sqrt(fc)/fy)
    rho_bal = (0.85 * beta1 * fc / fy) * (6120 / (6120 + fy))
    rho_max = 0.75 * rho_bal
    
    logs.append(f"Rho min = {rho_min:.5f}, Rho max = {rho_max:.5f}")
    
    if rho_req > rho_max:
        logs.append("<span style='color:red'>Fail: Rho > Rho_max (Brittle Failure)</span>")
        status = "Over Reinforced"
    else:
        status = "OK"

    # 6. Area of Steel
    rho_design = max(rho_req, rho_min)
    As_req = rho_design * b * d_est
    logs.append(f"As required = {As_req:.2f} cm2")
    
    # 7. Bar Selection
    bar_size = params['main_bar']
    area_one = get_rebar_area(bar_size)
    num_bars = math.ceil(As_req / area_one)
    num_bars = max(num_bars, 2) # Minimum 2 bars
    
    As_prov = num_bars * area_one
    logs.append(f"Selected: <b>{num_bars}-{bar_size}</b> (As={As_prov:.2f} cm2)")
    
    return {
        'Type': section_name,
        'Mu': Mu_kgm,
        'Bars': f"{num_bars}-{bar_size}",
        'Status': status,
        'Log': logs
    }

def calculate_shear_capacity(Vu_kg, b, h, cover, params):
    """Design Stirrups"""
    logs = []
    logs.append(f"Vu = {Vu_kg:.2f} kg")
    
    phi = 0.85
    fc = params['fc']
    fy = 2400 # Assume SR24 for stirrups
    d = h - cover - 2.0
    
    Vc = 0.53 * math.sqrt(fc) * b * d
    phiVc = phi * Vc
    logs.append(f"Capacity of Concrete (phi*Vc) = {phiVc:.2f} kg")
    
    stirrup_info = ""
    
    if Vu_kg <= phiVc / 2:
        stirrup_info = "Theoretically None (Use Min)"
        logs.append("Vu < 0.5*phiVc -> No shear steel required.")
    elif Vu_kg <= phiVc:
        stirrup_info = "Min Stirrups RB6 @ 20"
        logs.append("Vu < phiVc -> Use minimum stirrups.")
    else:
        # Calculate Vs
        Vs_req = (Vu_kg - phiVc) / phi
        logs.append(f"Vs Required = {Vs_req:.2f} kg")
        
        # Check Section Size
        if Vs_req > 2.1 * math.sqrt(fc) * b * d:
             stirrup_info = "Section Too Small!"
             logs.append("Error: Vs too high, increase concrete size.")
        else:
             # Spacing for RB6 (2 legs) -> Av = 0.56 cm2
             Av = 0.56
             s_calc = Av * fy * d / Vs_req
             s_max = d/2
             s_final = min(s_calc, s_max, 30) # Max 30 cm
             # Round to nearest 2.5 cm
             s_final = math.floor(s_final / 2.5) * 2.5
             stirrup_info = f"RB6 @ {s_final:.1f} cm"
             logs.append(f"Calculated spacing = {s_calc:.2f} cm -> Use {stirrup_info}")

    return stirrup_info, logs
