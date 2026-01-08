# reporter.py
import streamlit as st
import numpy as np

def render_calculation_report(res):
    """
    Detailed Calculation Report with Step-by-Step Substitution.
    Visualizes the formulas and checks against ACI standards.
    """
    # --- UNPACK VARIABLES ---
    idx = res['span_id'] + 1
    L_m = res['L']
    b = res['b'] # mm
    h = res['h'] # mm
    cov = res['cover']
    fc = res['fc']
    fy = res['fy']
    
    # Loads (Ultimate & Service)
    Mu_pos = res['Mu_pos']
    Mu_neg = res['Mu_neg']
    Vu = res['Vu_max']
    Ma_pos = res['Ma_pos_svc']    # Service Moment
    delta_svc = res['delta_svc_mm'] # Elastic Deflection
    
    # Steel
    bot_n, bot_db = res['bot']['n'], res['bot']['db']
    top_n, top_db = res['top']['n'], res['top']['db']
    stir_db, stir_s = res['shear']['db'], res['shear']['s']

    # --- CONSTANTS ---
    Es = 200000.0 # MPa
    Ec = 4700 * np.sqrt(fc) # ACI 19.2.2.1
    n_mod = Es / Ec
    
    # Beta 1 Calculation (Detailed)
    if fc <= 28: beta1 = 0.85
    elif fc >= 55: beta1 = 0.65
    else: beta1 = 0.85 - 0.05 * (fc - 28) / 7

    # --- REPORT HEADER ---
    st.markdown(f"### 📘 Calculation Sheet: Span {idx} (L = {L_m:.2f} m)")
    st.markdown("---")

    # 1. DESIGN PARAMETERS
    st.markdown("**1. Design Parameters & Material Properties**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write(f"- Section: ${b:.0f} \\times {h:.0f}$ mm")
        st.write(f"- Cover ($C_c$): ${cov:.0f}$ mm")
    with c2:
        st.write(f"- $f_c'$: ${fc}$ MPa")
        st.write(f"- $f_y$: ${fy}$ MPa")
        st.write(f"- $\\beta_1$: ${beta1:.3f}$")
    with c3:
        st.latex(rf"E_c = 4700\sqrt{{f_c'}} = {Ec:.0f}\text{{ MPa}}")
        st.latex(rf"n = E_s/E_c = {Es:.0f}/{Ec:.0f} \approx {n_mod:.2f}")

    st.markdown("---")

    # 2. FLEXURAL DESIGN (POSITIVE MOMENT)
    st.markdown(f"#### 2. Flexural Design: Mid-Span (+Moment)")
    st.info(f"**Demand:** $M_u^{{+}} = {Mu_pos:.2f}$ kNm")

    # 2.1 Depth and Area
    d_bot = h - cov - stir_db - (bot_db/2)
    As_bot = bot_n * (np.pi * (bot_db/2)**2)
    
    col_l, col_r = st.columns([1.5, 1])
    with col_l:
        st.markdown("**2.1 Effective Depth ($d$) & Area ($A_s$)**")
        st.latex(rf"d = h - C_c - d_{{stir}} - \frac{{d_b}}{{2}} = {h} - {cov} - {stir_db} - \frac{{{bot_db}}}{{2}} = \mathbf{{{d_bot:.1f}}}\text{{ mm}}")
        st.latex(rf"A_s = {bot_n} \times \frac{{\pi \cdot {bot_db}^2}}{{4}} = \mathbf{{{As_bot:.0f}}}\text{{ mm}}^2")

        st.markdown("**2.2 Stress Block Depth ($a$) & Neutral Axis ($c$)**")
        a = (As_bot * fy) / (0.85 * fc * b)
        c_depth = a / beta1
        st.latex(rf"a = \frac{{A_s f_y}}{{0.85 f_c' b}} = \frac{{{As_bot:.0f} \cdot {fy}}}{{0.85 \cdot {fc} \cdot {b}}} = {a:.2f}\text{{ mm}}")
        st.latex(rf"c = a / \beta_1 = {a:.2f} / {beta1:.3f} = {c_depth:.2f}\text{{ mm}}")

        st.markdown("**2.3 Strain Check ($\epsilon_t$) & $\phi$ Factor**")
        strain_t = 0.003 * (d_bot - c_depth) / c_depth
        st.latex(rf"\epsilon_t = 0.003 \frac{{d-c}}{{c}} = 0.003 \frac{{{d_bot:.1f}-{c_depth:.2f}}}{{{c_depth:.2f}}} = \mathbf{{{strain_t:.4f}}}")
        
        if strain_t >= 0.005:
            phi = 0.9
            st.write("Since $\epsilon_t \ge 0.005$, Section is **Tension Controlled** -> $\phi = 0.9$")
        else:
            phi = 0.65 + 0.25 * (strain_t - 0.002)/0.003
            st.write(f"Transition Region -> $\phi = {phi:.3f}$")

        st.markdown("**2.4 Capacity Check ($\phi M_n$)**")
        Mn = As_bot * fy * (d_bot - a/2) * 1e-6
        phiMn = phi * Mn
        st.latex(rf"\phi M_n = \phi A_s f_y (d - a/2) = {phi:.2f} \cdot {As_bot:.0f} \cdot {fy} ({d_bot:.1f} - \frac{{{a:.2f}}}{{2}}) \cdot 10^{{-6}}")
        st.latex(rf"\phi M_n = \mathbf{{{phiMn:.2f}}}\text{{ kNm}}")

    with col_r:
        st.markdown("###### Status Check")
        if phiMn >= Mu_pos:
            st.success(f"✅ **PASS**\n\nCapacity: {phiMn:.2f}\n\nRequired: {Mu_pos:.2f}")
        else:
            st.error(f"❌ **FAIL**\n\nCapacity: {phiMn:.2f}\n\nRequired: {Mu_pos:.2f}")
        
        st.caption("ACI 318 Standard Check")

    st.markdown("---")

    # 3. SHEAR DESIGN
    st.markdown(f"#### 3. Shear Design")
    st.info(f"**Demand:** $V_u = {Vu:.2f}$ kN")

    # Vc Calculation
    Vc = 0.17 * np.sqrt(fc) * b * d_bot * 1e-3
    phi_shear = 0.85
    phiVc = phi_shear * Vc
    
    # Vs Calculation
    Av = 2 * (np.pi * (stir_db/2)**2)
    Vs = (Av * fy * d_bot) / stir_s * 1e-3
    phiVs = phi_shear * Vs
    
    phiVn = phiVc + phiVs

    # Max Spacing Check logic
    threshold_vs = 0.33 * np.sqrt(fc) * b * d_bot * 1e-3
    if Vs <= threshold_vs:
        s_max_limit = min(d_bot/2, 600)
    else:
        s_max_limit = min(d_bot/4, 300)

    col_l, col_r = st.columns([1.5, 1])
    with col_l:
        st.markdown("**3.1 Concrete Capacity ($\phi V_c$)**")
        st.latex(rf"\phi V_c = 0.85 \cdot 0.17 \sqrt{{f_c'}} b d = 0.85 \cdot 0.17 \sqrt{{{fc}}} \cdot {b} \cdot {d_bot:.1f} = \mathbf{{{phiVc:.2f}}}\text{{ kN}}")
        
        st.markdown("**3.2 Steel Capacity ($\phi V_s$)**")
        st.latex(rf"A_v (2\text{{ legs}}) = 2 \times \pi \cdot {stir_db}^2 / 4 = {Av:.1f}\text{{ mm}}^2")
        st.latex(rf"\phi V_s = \phi \frac{{A_v f_y d}}{{s}} = 0.85 \frac{{{Av:.1f} \cdot {fy} \cdot {d_bot:.1f}}}{{{stir_s}}} = \mathbf{{{phiVs:.2f}}}\text{{ kN}}")
        
        st.markdown("**3.3 Spacing Check ($S_{max}$)**")
        st.write(f"Check $V_s$ threshold: {Vs:.2f} kN vs {threshold_vs:.2f} kN")
        st.latex(rf"S_{{max}} = \min(d/2, 600) = \mathbf{{{s_max_limit:.0f}}}\text{{ mm}}")
        if stir_s > s_max_limit:
            st.error(f"❌ Spacing {stir_s} mm > Max {s_max_limit:.0f} mm")
        else:
            st.write(f"✅ Spacing {stir_s} mm $\le$ Max {s_max_limit:.0f} mm")

    with col_r:
        st.markdown("###### Total Capacity")
        st.latex(rf"\phi V_n = {phiVc:.2f} + {phiVs:.2f} = \mathbf{{{phiVn:.2f}}}\text{{ kN}}")
        
        if phiVn >= Vu and stir_s <= s_max_limit:
            st.success(f"✅ **PASS**")
        else:
            st.error(f"❌ **FAIL**")

    st.markdown("---")

    # 4. DEFLECTION (SERVICEABILITY)
    st.markdown(f"#### 4. Serviceability (Deflection)")
    st.info(f"Checking at Service Load: $M_a = {Ma_pos:.2f}$ kNm (Unfactored)")
    
    # 4.1 Properties
    Ig = (b * h**3) / 12
    fr = 0.62 * np.sqrt(fc) # ACI 318 Modulus of Rupture
    yt = h / 2
    Mcr = (fr * Ig / yt) * 1e-6
    
    st.markdown("**4.1 Cracking Moment ($M_{cr}$)**")
    st.latex(rf"f_r = 0.62\sqrt{{f_c'}} = 0.62\sqrt{{{fc}}} = {fr:.2f}\text{{ MPa}}")
    st.latex(rf"M_{{cr}} = \frac{{f_r I_g}}{{y_t}} = \frac{{{fr:.2f} \cdot {Ig:.2E}}}{{{yt:.0f}}} \cdot 10^{{-6}} = \mathbf{{{Mcr:.2f}}}\text{{ kNm}}")
    
    # 4.2 Effective Inertia (Ie) - Branson's Method
    if Ma_pos < 1e-3: # negligible load
        Ie = Ig
        st.write("No significant load.")
    elif Ma_pos < Mcr:
        Ie = Ig
        st.write(f"Since $M_a < M_{{cr}}$, section is **Uncracked**.")
        st.latex(rf"I_e = I_g = {Ig:.2E}\text{{ mm}}^4")
    else:
        # Calculate Icr (Transformed Section)
        rho = As_bot / (b * d_bot)
        rn = rho * n_mod
        k = np.sqrt(rn**2 + 2*rn) - rn
        kd = k * d_bot
        Icr = (b * kd**3)/3 + n_mod * As_bot * (d_bot - kd)**2
        
        st.write(f"Section Cracked ($M_a > M_{{cr}}$). Using Cracked Inertia.")
        st.latex(rf"kd = {kd:.1f}\text{{ mm}}, \quad I_{{cr}} = {Icr:.2E}\text{{ mm}}^4")
        
        # Branson's Formula
        term = (Mcr / Ma_pos)**3
        Ie = term * Ig + (1 - term) * Icr
        # Limit Ie <= Ig
        Ie = min(Ie, Ig)
        
        st.latex(rf"I_e = \left(\frac{{M_{{cr}}}}{{M_a}}\right)^3 I_g + \left[1 - \left(\frac{{M_{{cr}}}}{{M_a}}\right)^3\right] I_{{cr}}")
        st.latex(rf"I_e = {term:.3f} I_g + {(1-term):.3f} I_{{cr}} = \mathbf{{{Ie:.2E}}}\text{{ mm}}^4")

    # 4.3 Total Deflection
    st.markdown("**4.3 Long-Term Deflection**")
    # Immediate Deflection adjustment
    delta_immediate = delta_svc * (Ig / Ie)
    
    # Long term multiplier (lambda)
    # xi = 2.0 (5 years duration)
    # rho' = 0 (assume no compression steel for simple conservative check)
    lam = 2.0 / (1 + 50 * 0) 
    
    delta_long = delta_immediate * lam
    delta_total = delta_immediate + delta_long
    limit_L240 = L_m * 1000 / 240
    
    c1, c2 = st.columns(2)
    with c1:
        st.latex(rf"\Delta_{{inst}} = \Delta_{{elastic}} \times (I_g/I_e) = {delta_svc:.2f} \times {Ig/Ie:.2f} = {delta_immediate:.2f}\text{{ mm}}")
        st.latex(rf"\Delta_{{total}} = \Delta_{{inst}} (1 + \lambda) = {delta_immediate:.2f} (1 + 2.0) = \mathbf{{{delta_total:.2f}}}\text{{ mm}}")
    with c2:
        st.metric("Total Deflection", f"{delta_total:.2f} mm")
        st.metric("Limit (L/240)", f"{limit_L240:.2f} mm", delta_total - limit_L240, delta_color="inverse")

    if delta_total <= limit_L240:
        st.success("✅ **SERVICEABILITY PASS**")
    else:
        st.error("❌ **DEFLECTION EXCEEDED**")
