# reporter.py
import streamlit as st
import numpy as np

def render_calculation_report(res):
    """
    Renders a professional, step-by-step calculation report for RC Beam.
    Shows Formula -> Substitution -> Result.
    """
    # --- 1. UNPACK DATA ---
    idx = res['span_id'] + 1
    L = res['L']
    b = res['b']
    h = res['h']
    cov = res['cover']
    fc = res['fc']
    fy = res['fy']
    
    Mu_pos = res['Mu_pos']
    Mu_neg = res['Mu_neg']
    Vu = res['Vu']
    Ma_pos = res['Ma_pos']
    delta_svc = res['delta_svc']
    
    bot_n, bot_db = res['bot']['n'], res['bot']['db']
    top_n, top_db = res['top']['n'], res['top']['db']
    stir_db, stir_s = res['shear']['db'], res['shear']['s']

    # Constants
    Es = 200000.0 # MPa
    Ec = 4700 * np.sqrt(fc)
    n_mod = Es / Ec
    beta1 = 0.85 if fc <= 30 else max(0.65, 0.85 - 0.05 * (fc - 30) / 7)

    # --- START REPORT ---
    st.markdown(f"### 📍 Span {idx}: Analysis & Design Report")
    st.markdown("---")

    # 1. PROPERTIES
    st.markdown("**1. Design Parameters**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write(f"- Size: ${b:.0f} \\times {h:.0f}$ mm")
        st.write(f"- Length: ${L:.2f}$ m")
        st.write(f"- Cover: ${cov:.0f}$ mm")
    with c2:
        st.write(f"- $f_c'$: ${fc}$ MPa")
        st.write(f"- $f_y$: ${fy}$ MPa")
        st.write(f"- $\\beta_1$: ${beta1:.2f}$")
    with c3:
        st.write(f"- $E_c$: ${Ec:.0f}$ MPa")
        st.write(f"- $n = E_s/E_c$: ${n_mod:.2f}$")

    # 2. POSITIVE MOMENT
    st.markdown("---")
    st.markdown("#### 2. Positive Moment Check (Mid-Span)")
    st.write(f"**Demand:** $M_u^{{+}} = \\mathbf{{{Mu_pos:.2f}}}$ kNm")
    
    # Calculate d
    d_bot = h - cov - stir_db - (bot_db/2)
    As_bot = bot_n * (np.pi * (bot_db/2)**2)
    
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.markdown("**2.1 Effective Depth ($d$) & Area ($A_s$)**")
        st.latex(rf"d = h - C_c - d_v - d_b/2 = {h:.0f} - {cov} - {stir_db} - {bot_db}/2 = \mathbf{{{d_bot:.1f}}}\text{{ mm}}")
        st.latex(rf"A_s ({bot_n}\text{{-DB}}{bot_db}) = {bot_n} \times \pi \times ({bot_db}/2)^2 = \mathbf{{{As_bot:.0f}}}\text{{ mm}}^2")

        st.markdown("**2.2 Moment Capacity ($\phi M_n$)**")
        # a
        a = (As_bot * fy) / (0.85 * fc * b)
        st.latex(rf"a = \frac{{A_s f_y}}{{0.85 f_c' b}} = \frac{{{As_bot:.0f} \cdot {fy}}}{{0.85 \cdot {fc} \cdot {b:.0f}}} = {a:.2f}\text{{ mm}}")
        # Mn
        Mn = As_bot * fy * (d_bot - a/2) * 1e-6
        # Phi check (Simplified for report clarity, assuming tension controlled if well designed)
        c_depth = a / beta1
        dt = d_bot
        strain = 0.003 * (dt - c_depth)/c_depth
        phi = 0.9 if strain >= 0.005 else 0.65 + 0.25*(strain-0.002)/0.003
        phiMn = phi * Mn
        
        st.latex(rf"\phi M_n = \phi A_s f_y (d - a/2) = {phi:.2f} \cdot {As_bot:.0f} \cdot {fy} \cdot ({d_bot:.1f} - {a:.2f}/2) \cdot 10^{{-6}}")
        st.latex(rf"\phi M_n = \mathbf{{{phiMn:.2f}}}\text{{ kNm}}")

    with col_r:
        st.write("")
        st.write("")
        if phiMn >= Mu_pos:
            st.success(f"✅ PASS\n\nCap: {phiMn:.2f} > Req: {Mu_pos:.2f}")
        else:
            st.error(f"❌ FAIL\n\nCap: {phiMn:.2f} < Req: {Mu_pos:.2f}")

    # 3. NEGATIVE MOMENT
    st.markdown("---")
    st.markdown("#### 3. Negative Moment Check (Support)")
    st.write(f"**Demand:** $M_u^{{-}} = \\mathbf{{{Mu_neg:.2f}}}$ kNm")
    
    d_top = h - cov - stir_db - (top_db/2)
    As_top = top_n * (np.pi * (top_db/2)**2)
    
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.latex(rf"d = {d_top:.1f}\text{{ mm}}, \quad A_s({top_n}\text{{-DB}}{top_db}) = \mathbf{{{As_top:.0f}}}\text{{ mm}}^2")
        
        a_t = (As_top * fy) / (0.85 * fc * b)
        Mn_t = As_top * fy * (d_top - a_t/2) * 1e-6
        phiMn_t = 0.9 * Mn_t # Assume phi=0.9 for simplicity in quick report
        
        st.latex(rf"\phi M_n = 0.9 \cdot {As_top:.0f} \cdot {fy} \cdot ({d_top:.1f} - {a_t:.2f}/2) \cdot 10^{{-6}} = \mathbf{{{phiMn_t:.2f}}}\text{{ kNm}}")

    with col_r:
        if phiMn_t >= Mu_neg:
            st.success(f"✅ PASS\n\nCap: {phiMn_t:.2f} > Req: {Mu_neg:.2f}")
        else:
            st.error(f"❌ FAIL\n\nCap: {phiMn_t:.2f} < Req: {Mu_neg:.2f}")

    # 4. SHEAR
    st.markdown("---")
    st.markdown("#### 4. Shear Strength Check")
    
    Vc = 0.17 * np.sqrt(fc) * b * d_bot * 1e-3
    phiVc = 0.85 * Vc
    Av = 2 * (np.pi * (stir_db/2)**2)
    Vs = (Av * fy * d_bot) / stir_s * 1e-3
    phiVs = 0.85 * Vs
    phiVn = phiVc + phiVs
    
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.latex(rf"\phi V_c = 0.85 \cdot 0.17\sqrt{{{fc}}} \cdot {b:.0f} \cdot {d_bot:.1f} = \mathbf{{{phiVc:.2f}}}\text{{ kN}}")
        st.latex(rf"\phi V_s (\text{{RB}}{stir_db}@{stir_s}) = 0.85 \cdot \frac{{{Av:.1f} \cdot {fy} \cdot {d_bot:.1f}}}{{{stir_s}}} = \mathbf{{{phiVs:.2f}}}\text{{ kN}}")
        st.latex(rf"\phi V_n = {phiVc:.2f} + {phiVs:.2f} = \mathbf{{{phiVn:.2f}}}\text{{ kN}}")
    with col_r:
        if phiVn >= Vu:
            st.success(f"✅ PASS\n\nCap: {phiVn:.2f} > Req: {Vu:.2f}")
        else:
            st.error(f"❌ FAIL\n\nCap: {phiVn:.2f} < Req: {Vu:.2f}")

    # 5. SERVICEABILITY (Detailed)
    st.markdown("---")
    st.markdown("#### 5. Serviceability Check (Deflection)")
    st.info(f"Checking at Service Load ($M_a = {Ma_pos:.2f}$ kNm)")
    
    # 5.1 Cracking Moment
    Ig = (b * h**3) / 12
    fr = 0.62 * np.sqrt(fc)
    yt = h / 2
    Mcr = (fr * Ig / yt) * 1e-6
    
    st.markdown("**5.1 Cracking Moment ($M_{cr}$)**")
    st.latex(rf"I_g = \frac{{bh^3}}{{12}} = {Ig:.2E}\text{{ mm}}^4, \quad f_r = 0.62\sqrt{{{fc}}} = {fr:.2f}\text{{ MPa}}")
    st.latex(rf"M_{{cr}} = \frac{{f_r I_g}}{{y_t}} = \frac{{{fr:.2f} \cdot {Ig:.2E}}}{{{yt:.0f}}} = \mathbf{{{Mcr:.2f}}}\text{{ kNm}}")
    
    # 5.2 Effective Inertia
    st.markdown("**5.2 Effective Moment of Inertia ($I_e$)**")
    if Ma_pos < Mcr:
        Ie = Ig
        st.write(f"Since $M_a ({Ma_pos:.2f}) < M_{{cr}} ({Mcr:.2f})$, section is **Uncracked**.")
        st.latex(rf"I_e = I_g = {Ig:.2E}\text{{ mm}}^4")
    else:
        # Transformed Section
        rho = As_bot / (b * d_bot)
        rn = rho * n_mod
        k = np.sqrt(rn**2 + 2*rn) - rn
        kd = k * d_bot
        Icr = (b * kd**3)/3 + n_mod * As_bot * (d_bot - kd)**2
        
        st.write(f"Section Cracked ($M_a > M_{{cr}}$). Using Transformed Section ($kd={kd:.1f}$ mm).")
        st.latex(rf"I_{{cr}} = \frac{{b(kd)^3}}{{3}} + nA_s(d-kd)^2 = {Icr:.2E}\text{{ mm}}^4")
        
        # Branson
        term = (Mcr / Ma_pos)**3
        Ie = term * Ig + (1 - term) * Icr
        st.latex(rf"I_e = \left(\frac{{M_{{cr}}}}{{M_a}}\right)^3 I_g + \dots = \mathbf{{{Ie:.2E}}}\text{{ mm}}^4")

    # 5.3 Deflection
    st.markdown("**5.3 Deflection Calculation**")
    factor = Ig / Ie
    delta_i = delta_svc * factor
    
    # Long term
    xi = 2.0
    lam = xi / (1 + 50*0) # rho' = 0
    delta_lt = delta_i * lam
    delta_tot = delta_i + delta_lt
    
    st.write(f"- Elastic Deflection (from analysis, using $I_g$): $\\Delta_{{el}} = {delta_svc:.2f}$ mm")
    st.latex(rf"\Delta_{{inst}} = \Delta_{{el}} \times \frac{{I_g}}{{I_e}} = {delta_svc:.2f} \times {factor:.2f} = \mathbf{{{delta_i:.2f}}}\text{{ mm}}")
    st.latex(rf"\Delta_{{total}} = \Delta_{{inst}} (1 + \lambda) = {delta_i:.2f} (1 + {lam}) = \mathbf{{{delta_tot:.2f}}}\text{{ mm}}")
    
    # Check Limit
    limit = L * 1000 / 240
    col_res, col_lim = st.columns(2)
    with col_res:
         st.metric("Total Deflection", f"{delta_tot:.2f} mm")
    with col_lim:
         st.metric("Limit (L/240)", f"{limit:.2f} mm", delta_tot - limit, delta_color="inverse")
         
    if delta_tot <= limit:
        st.success("✅ Serviceability Check Passed")
    else:
        st.error("❌ Deflection Exceeds Limit")
