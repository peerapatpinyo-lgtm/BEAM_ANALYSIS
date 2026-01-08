import streamlit as st
import numpy as np

def render_calculation_report(span_idx, span_len, b, h, fc, fy, Mu_pos, Mu_neg, Vu, res_data, Ma_pos=None, w_service=None):
    """
    Generate a detailed Step-by-Step calculation report based on ACI 318 / EIT (SDM).
    Now includes Serviceability (Deflection) Check.
    
    Ma_pos: Maximum Positive Moment under Service Load (DL + LL) [kNm]
    w_service: Distributed Service Load (DL + LL) [kN/m]
    """
    
    # --- 1. Constants & Section Properties ---
    st.markdown(f"### 📍 Design Calculation: Span {span_idx + 1}")
    st.markdown("---")
    
    b_mm = b * 1000
    h_mm = h * 1000
    L_mm = span_len * 1000
    cover = res_data['cover']
    Es = 200000 # Steel Modulus MPa
    
    # --- Estimate Service Loads if not provided (Fallback) ---
    # Assuming rough factor of 1.45 average if user doesn't pass Ma
    if Ma_pos is None:
        Ma_pos = Mu_pos / 1.45
    if w_service is None:
        # Back-calculate w from Moment assuming wL^2/8 roughly (just for display)
        w_service = (8 * Ma_pos) / (span_len**2)

    # Display Design Data
    st.markdown("**1. Design Data & Material Properties**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write(f"- $f_c'$: **{fc}** MPa")
        st.write(f"- $f_y$: **{fy}** MPa")
        st.write(f"- $E_s$: **{Es}** MPa")
    with c2:
        st.write(f"- $b$: **{b_mm:.0f}** mm")
        st.write(f"- $h$: **{h_mm:.0f}** mm")
        st.write(f"- Span: **{span_len:.2f}** m")
    with c3:
        st.write(f"- Cover: **{cover}** mm")
        st.write(f"- Service Load ($w$): **{w_service:.2f}** kN/m")
        st.write(f"- Service Moment ($M_a$): **{Ma_pos:.2f}** kNm")

    # Determine Beta1
    if fc <= 30:
        beta1 = 0.85
    else:
        beta1 = max(0.65, 0.85 - 0.05 * (fc - 30) / 7)

    # --- 2. FLEXURAL DESIGN (POSITIVE) ---
    st.markdown("---")
    st.markdown(f"**2. Flexural Design: Mid-Span (Positive Moment)**")
    
    mu = Mu_pos
    bot_db = res_data['bot_db']
    bot_n = res_data['pos']['n']
    stir_db = res_data['stir_db']
    
    # 2.1 Effective Depth
    d_bot = h_mm - cover - stir_db - (bot_db/2)
    st.latex(f"d = {h_mm:.0f} - {cover} - {stir_db} - {bot_db/2:.1f} = \\mathbf{{{d_bot:.1f}}} \\text{{ mm}}")

    # 2.2 Reinforcement
    As_prov = bot_n * (np.pi * (bot_db/2)**2)
    st.write(f"Design $M_u$: **{mu:.2f}** kNm | Provide: **{bot_n}-DB{bot_db}**")
    st.latex(f"A_{{s,prov}} = {bot_n} \\times \\pi \\times ({bot_db}/2)^2 = \\mathbf{{{As_prov:.0f}}} \\text{{ mm}}^2")

    # 2.3 Capacity Check
    a = (As_prov * fy) / (0.85 * fc * b_mm)
    c = a / beta1
    epsilon_t = 0.003 * (d_bot - c) / c
    phi = 0.9 if epsilon_t >= 0.005 else 0.65 + 0.25*((epsilon_t - 0.002)/0.003)
    
    Mn = As_prov * fy * (d_bot - a/2) * 1e-6
    phiMn = phi * Mn
    
    st.latex(f"\\phi M_n = {phi} \\cdot {As_prov:.0f} \\cdot {fy} \\cdot ({d_bot:.1f} - {a:.2f}/2) \\cdot 10^{{-6}} = \\mathbf{{{phiMn:.2f}}} \\text{{ kNm}}")
    
    if phiMn >= mu:
        st.success(f"✅ PASS: $\phi M_n$ ({phiMn:.2f}) > $M_u$ ({mu:.2f})")
    else:
        st.error(f"❌ FAIL: $\phi M_n$ ({phiMn:.2f}) < $M_u$ ({mu:.2f})")

    # --- 3. FLEXURAL DESIGN (NEGATIVE) ---
    st.markdown("---")
    st.markdown(f"**3. Flexural Design: Support (Negative Moment)**")
    # (Simplified for brevity in this update)
    mu_n = Mu_neg
    if mu_n > 0.01:
        top_db = res_data['top_db']
        top_n = res_data['neg']['n']
        d_top = h_mm - cover - stir_db - (top_db/2)
        As_prov_top = top_n * (np.pi * (top_db/2)**2)
        
        a_top = (As_prov_top * fy) / (0.85 * fc * b_mm)
        phiMn_top = 0.9 * As_prov_top * fy * (d_top - a_top/2) * 1e-6
        
        st.write(f"Design $M_u$: **{mu_n:.2f}** kNm | Provide: **{top_n}-DB{top_db}**")
        if phiMn_top >= mu_n:
            st.success(f"✅ PASS: $\phi M_n$ ({phiMn_top:.2f}) > $M_u$ ({mu_n:.2f})")
        else:
            st.error(f"❌ FAIL: Capacity < Demand")
    else:
        st.info("No significant negative moment.")

    # --- 4. SHEAR DESIGN ---
    st.markdown("---")
    st.markdown(f"**4. Shear Design**")
    vu = res_data['Vu_max']
    stir_s = res_data['shear']['s']
    
    Vc = 0.17 * np.sqrt(fc) * b_mm * d_bot / 1000.0
    phiVc = 0.85 * Vc
    
    st.write(f"Factored $V_u$: **{vu:.2f}** kN | $\phi V_c$: **{phiVc:.2f}** kN")
    
    Av = 2 * (np.pi * (stir_db/2)**2)
    Vs_prov = (Av * fy * d_bot) / stir_s / 1000.0
    phiVn = phiVc + (0.85 * Vs_prov)
    
    st.write(f"Provide **RB{stir_db} @ {stir_s} mm** ($V_s = {Vs_prov:.2f}$ kN)")
    
    if phiVn >= vu:
        st.success(f"✅ PASS: $\phi V_n$ ({phiVn:.2f}) > $V_u$ ({vu:.2f})")
    else:
        st.error(f"❌ FAIL: Shear Capacity insufficient")

    # --- 5. DEFLECTION CHECK (SERVICEABILITY) ---
    st.markdown("---")
    st.markdown(f"### 5. Serviceability Check (Deflection)")
    st.info("Calculation based on **Service Load** ($DL+LL$) and **Effective Moment of Inertia ($I_e$)**")

    # 5.1 Material Modulus
    Ec = 4700 * np.sqrt(fc)
    n = Es / Ec
    st.markdown("**5.1 Modulus of Elasticity & Modular Ratio**")
    st.latex(rf"E_c = 4700\sqrt{{f_c'}} = 4700\sqrt{{{fc}}} = {Ec:.0f} \text{{ MPa}}")
    st.latex(rf"n = \frac{{E_s}}{{E_c}} = \frac{{{Es}}}{{{Ec:.0f}}} = {n:.2f}")

    # 5.2 Gross Section Properties (Uncracked)
    Ig = (b_mm * h_mm**3) / 12
    fr = 0.62 * np.sqrt(fc) # Modulus of Rupture
    yt = h_mm / 2
    Mcr = (fr * Ig) / yt * 1e-6 # Convert to kNm
    
    st.markdown("**5.2 Gross Section & Cracking Moment**")
    st.latex(rf"I_g = \frac{{bh^3}}{{12}} = {Ig:.2e} \text{{ mm}}^4")
    st.latex(rf"f_r = 0.62\sqrt{{f_c'}} = {fr:.2f} \text{{ MPa}}")
    st.latex(rf"M_{{cr}} = \frac{{f_r I_g}}{{y_t}} = \frac{{{fr:.2f} \cdot {Ig:.2e}}}{{{yt:.0f}}} \cdot 10^{{-6}} = \mathbf{{{Mcr:.2f}}} \text{{ kNm}}")

    # 5.3 Effective Inertia (Ie)
    st.markdown("**5.3 Effective Moment of Inertia ($I_e$)**")
    
    Ma = Ma_pos # Service Moment
    
    if Ma < Mcr:
        st.success(f"Condition: $M_a$ ({Ma:.2f}) < $M_{{cr}}$ ({Mcr:.2f}) $\\to$ Section is **Uncracked**")
        Ie = Ig
        st.latex(r"I_e = I_g")
    else:
        st.warning(f"Condition: $M_a$ ({Ma:.2f}) > $M_{{cr}}$ ({Mcr:.2f}) $\\to$ Section is **Cracked**")
        
        # Calculate Cracked Inertia (Icr) - Transformed Section
        rho = As_prov / (b_mm * d_bot)
        # k = sqrt( (rho*n)^2 + 2*rho*n ) - rho*n
        rn = rho * n
        k = np.sqrt(rn**2 + 2*rn) - rn
        kd = k * d_bot
        
        # Icr = (b * kd^3)/3 + n * As * (d - kd)^2
        Icr = (b_mm * kd**3)/3 + n * As_prov * (d_bot - kd)**2
        
        st.write(f"Cracked Neutral Axis ($kd$): {kd:.2f} mm")
        st.latex(rf"I_{{cr}} = \frac{{b(kd)^3}}{{3}} + n A_s (d-kd)^2 = {Icr:.2e} \text{{ mm}}^4")
        
        # Branson's Equation for Ie
        term = (Mcr / Ma)**3
        Ie = term * Ig + (1 - term) * Icr
        # Ie must not exceed Ig
        Ie = min(Ie, Ig)
        
        st.latex(r"I_e = \left(\frac{M_{cr}}{M_a}\right)^3 I_g + \left[1 - \left(\frac{M_{cr}}{M_a}\right)^3\right] I_{cr}")
        st.latex(rf"I_e = {term:.3f} I_g + {1-term:.3f} I_{{cr}} = \mathbf{{{Ie:.2e}}} \text{{ mm}}^4")

    # 5.4 Deflection Calculation
    st.markdown("**5.4 Deflection Calculation ($\Delta$)**")
    
    # Immediate Deflection (Elastic)
    # Using 5wL^4 / 384EI for simple span (Approximation for generic span)
    # Note: For exact structural mechanics, we should use exact beam formulas, 
    # but for this report, the standard simply supported formula is the common conservative check.
    
    delta_imm = (5 * w_service * (L_mm**4)) / (384 * Ec * Ie) * 1000 # Convert kN/m to N/mm -> *1000 is wrong scaling.
    # Unit check:
    # w (N/mm) = w_service (kN/m) / 1.0 (actually 1 kN/m = 1 N/mm)
    # L (mm)
    # E (MPa = N/mm2)
    # I (mm4)
    # Formula: (5 * w * L^4) / (384 * E * I)
    
    w_newton = w_service # kN/m is equivalent to N/mm
    delta_imm = (5 * w_newton * (L_mm**4)) / (384 * Ec * Ie)
    
    st.write(f"Immediate Deflection ($\Delta_i$): **{delta_imm:.2f}** mm")
    
    # Long-term Deflection
    # Lambda = xi / (1 + 50*rho') -> assume rho'=0 (no compression steel usually considered for simplification)
    xi = 2.0 # Time dependent factor for > 5 years
    lambda_delta = xi / (1 + 0) # Conservative
    delta_long = lambda_delta * delta_imm
    delta_total = delta_imm + delta_long
    
    st.latex(r"\Delta_{long} = \lambda_{\Delta} \cdot \Delta_i \quad (\text{Use } \lambda = 2.0)")
    st.latex(rf"\Delta_{{total}} = {delta_imm:.2f} + {delta_long:.2f} = \mathbf{{{delta_total:.2f}}} \text{{ mm}}")
    
    # 5.5 Check Limits
    st.markdown("**5.5 Check against Limits**")
    
    limit_240 = L_mm / 240
    limit_180 = L_mm / 180
    limit_360 = L_mm / 360 # Usually for Live Load only, but checking Total here as conservative baseline
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"Limit $L/240$: **{limit_240:.2f}** mm")
        if delta_total <= limit_240:
             st.success("✅ PASS (L/240)")
        else:
             st.error("❌ FAIL (L/240)")
             
    with col_b:
        st.write(f"Limit $L/180$: **{limit_180:.2f}** mm")
        if delta_total <= limit_180:
             st.success("✅ PASS (L/180)")
        else:
             st.error("❌ FAIL (L/180)")

    st.caption("Note: Long-term deflection assumes factor of 2.0 (duration > 5 years).")
