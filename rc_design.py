import numpy as np

def get_rebar_area(size_str):
    """Return area of rebar in cm2 based on standard Thai sizes"""
    db_map = {
        'DB12': 1.13, 'DB16': 2.01, 'DB20': 3.14, 'DB25': 4.91, 'DB28': 6.16
    }
    return db_map.get(size_str, 2.01)

def calculate_flexure_sdm(Mu, section_name, b, h, cover, params):
    """
    Design for Flexure (Strength Design Method)
    Mu: Ultimate Moment (kg-m) (+ or -)
    """
    Mu_abs = abs(Mu)
    fc = params['fc']
    fy = params['fy']
    phi_b = 0.90  # Flexure
    
    # Effective depth
    d = h - cover - 0.6 - 1.0 # approx (cover + stirrup + half_bar)
    if d <= 0: return {'Status': 'Error', 'Log': ['Depth too small']}
    
    logs = []
    logs.append(f"Section: {b}x{h} cm, d={d:.2f} cm")
    logs.append(f"Mu = {Mu_abs:.2f} kg-m")
    
    # Required Mn
    Mn_req = Mu_abs / phi_b * 100 # kg-cm
    
    # Check Max Reinforcement (Simple rho_max approx 0.75 rho_b)
    # Beta1
    beta1 = 0.85 if fc <= 280 else max(0.65, 0.85 - 0.05*((fc-280)/70))
    rho_b = (0.85 * beta1 * fc / fy) * (6120 / (6120 + fy))
    rho_max = 0.75 * rho_b
    rho_min = max(14/fy, 0.25*np.sqrt(fc)/fy) # ACI Metric
    
    # Calculate Required As
    # Rn = Mn / (b * d^2)
    Rn = Mn_req / (b * d**2) # kg/cm2
    
    # Check if section is large enough (Rn must < Rn_max)
    m = fy / (0.85 * fc)
    try:
        rho_req = (1/m) * (1 - np.sqrt(1 - 2*m*Rn/fy))
    except:
        return {'Type': section_name, 'Mu': Mu, 'Bars': "Section Too Small", 'Status': 'Fail (Compression Failure)', 'Log': logs}

    As_req = rho_req * b * d
    As_min = rho_min * b * d
    As_final = max(As_req, As_min)
    
    logs.append(f"As required = {As_final:.2f} cm2")
    
    # Select Bars
    bar_size = params['main_bar']
    bar_area = get_rebar_area(bar_size)
    num_bars = int(np.ceil(As_final / bar_area))
    # Minimum 2 bars
    num_bars = max(num_bars, 2)
    
    real_As = num_bars * bar_area
    status = "OK" if rho_req <= rho_max else "Over Reinforced!"
    
    return {
        'Type': section_name,
        'Mu': Mu,
        'Bars': f"{num_bars}-{bar_size}",
        'As_prov': real_As,
        'Status': status,
        'Log': logs
    }

def calculate_shear_capacity(Vu, b, h, cover, params):
    """
    Design Shear Reinforcement (Stirrups)
    Vu: Ultimate Shear (kg)
    """
    fc = params['fc']
    fy = params['fy'] # usually use stirrup grade? Assuming same for now or add param
    phi_v = 0.85
    d = h - cover - 1.5
    
    Vc = 0.53 * np.sqrt(fc) * b * d # kg (Simple ACI metric)
    phiVc = phi_v * Vc
    
    logs = [f"Vu = {Vu:.2f} kg", f"phi*Vc = {phiVc:.2f} kg"]
    
    stirrup_txt = ""
    
    if Vu <= phiVc / 2:
        stirrup_txt = "Theoretical: None required"
    elif Vu <= phiVc:
        stirrup_txt = "Min Stirrups (e.g. RB6 @ 200)"
    else:
        # Vs needed
        Vs_req = (Vu - phiVc) / phi_v
        # Check max shear
        if Vs_req > 2.1 * np.sqrt(fc) * b * d:
            stirrup_txt = "Section Too Small (Shear)"
        else:
            # Calculate spacing for RB6 (Area = 0.56 for 2 legs)
            Av = 2 * 0.28 # 2 legs of 6mm = 0.56 cm2
            # s = Av * fy * d / Vs
            s_req = Av * 2400 * d / Vs_req # Assume RB6 is SR24
            s_final = min(s_req, d/2, 60) # Max spacing rules
            # Round down to nearest 5 or 2.5
            s_final = int(s_final // 2.5) * 2.5
            stirrup_txt = f"RB6 @ {s_final:.0f} cm"
            
    return Vc, phiVc, stirrup_txt, logs
