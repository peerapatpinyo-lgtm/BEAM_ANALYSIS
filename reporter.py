# ในไฟล์ reporter.py (ทับ code เดิมได้เลย)

import streamlit as st
import numpy as np

def render_calculation_report(res):
    """
    Detailed Calculation Report with Step-by-Step Substitution.
    """
    # ... (ส่วนประกาศตัวแปรเหมือนเดิม) ...
    idx = res['span_id'] + 1
    L_m = res['L']
    b = res['b'] 
    h = res['h'] 
    cov = res['cover']
    fc = res['fc']
    fy = res['fy']
    
    Mu_pos = res['Mu_pos']
    Mu_neg = res['Mu_neg']
    Vu = res['Vu_max']
    Ma_pos = res['Ma_pos_svc']   
    delta_svc = res['delta_svc_mm'] 
    
    bot_n, bot_db = res['bot']['n'], res['bot']['db']
    stir_db, stir_s = res['shear']['db'], res['shear']['s']

    Es = 200000.0 
    Ec = 4700 * np.sqrt(fc)
    n_mod = Es / Ec
    
    if fc <= 28: beta1 = 0.85
    elif fc >= 55: beta1 = 0.65
    else: beta1 = 0.85 - 0.05 * (fc - 28) / 7

    # --- REPORT HEADER ---
    st.markdown(f"### 📘 Calculation Sheet: Span {idx}")
    st.markdown("---")

    # 1. PARAMETERS
    st.markdown("**1. Design Parameters**")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"- Section: {b:.0f} x {h:.0f} mm")
        st.write(f"- Material: f'c={fc} MPa, fy={fy} MPa")
    with c2:
        st.write(f"- $\\beta_1$: {beta1:.3f}")
        st.latex(rf"E_c = {Ec:.0f}\text{{ MPa}}")

    st.markdown("---")

    # =========================================================
    # 2. FLEXURAL DESIGN (ULTIMATE STRENGTH)
    # =========================================================
    st.markdown(f"#### 2. Flexural Design (Ultimate Strength Design - USD)")
    st.markdown(f"*Load Factor used: {Mu_pos:.2f} kNm is the Factored Load ($M_u$)*")
    st.info(f"**Ultimate Demand:** $M_u^{{+}} = {Mu_pos:.2f}$ kNm")

    # 2.1 Depth and Area
    d_bot = h - cov - stir_db - (bot_db/2)
    As_bot = bot_n * (np.pi * (bot_db/2)**2)
    
    col_l, col_r = st.columns([1.5, 1])
    with col_l:
        st.markdown("**2.1 Effective Depth & Steel Area**")
        st.latex(rf"d = {h} - {cov} - {stir_db} - {bot_db}/2 = \mathbf{{{d_bot:.1f}}}\text{{ mm}}")
        st.latex(rf"A_s = {bot_n} \times \pi \cdot {bot_db}^2 / 4 = \mathbf{{{As_bot:.0f}}}\text{{ mm}}^2")

        st.markdown("**2.2 Stress Block & Strain**")
        a = (As_bot * fy) / (0.85 * fc * b)
        c_depth = a / beta1
        
        # Check logic: if a > d, print error immediately
        if a >= d_bot:
            st.error(f"❌ **CRITICAL ERROR:** $a$ ({a:.1f} mm) > $d$ ({d_bot:.1f} mm)")
            st.write("Section is too small for this amount of steel (Over-reinforced).")
            st.write("Concrete crushes before steel yields.")
            phi, phiMn = 0.0, 0.0
        else:
            st.latex(rf"a = \frac{{{As_bot:.0f} \cdot {fy}}}{{0.85 \cdot {fc} \cdot {b}}} = {a:.2f}\text{{ mm}}")
            
            if c_depth > 0:
                strain_t = 0.003 * (d_bot - c_depth) / c_depth
            else:
                strain_t = 0
                
            st.latex(rf"\epsilon_t = \mathbf{{{strain_t:.4f}}}")
            
            if strain_t >= 0.005: phi = 0.9
            elif strain_t <= 0.002: phi = 0.65
            else: phi = 0.65 + 0.25 * (strain_t - 0.002)/0.003
            
            # 2.3 Capacity
            Mn = As_bot * fy * (d_bot - a/2) * 1e-6
            phiMn = phi * Mn
            
            st.markdown("**2.3 Moment Capacity ($\phi M_n$)**")
            st.latex(rf"\phi M_n = {phi:.2f} \cdot {As_bot:.0f} \cdot {fy} ({d_bot:.1f} - {a:.2f}/2) \cdot 10^{{-6}}")
            st.latex(rf"\phi M_n = \mathbf{{{phiMn:.2f}}}\text{{ kNm}}")

    with col_r:
        st.markdown("###### Status Check")
        if phiMn >= Mu_pos and phiMn > 0:
            st.success(f"✅ **PASS**\n\nCap: {phiMn:.2f}\n\nReq: {Mu_pos:.2f}")
        else:
            st.error(f"❌ **FAIL**\n\nCap: {phiMn:.2f}\n\nReq: {Mu_pos:.2f}")
            if phiMn == 0:
                st.caption("Fail: Section geometry invalid")

    st.markdown("---")

    # ... (ส่วน Shear และ Deflection เหมือนเดิม) ...
    # 3. SHEAR
    st.markdown(f"#### 3. Shear Design (Ultimate)")
    # (Copy Shear Logic from previous answer here)
    # ...

    # 4. SERVICEABILITY
    st.markdown(f"#### 4. Serviceability (Deflection)")
    st.info(f"**Service Load:** $M_a = {Ma_pos:.2f}$ kNm (Unfactored)")
    # (Copy Deflection Logic from previous answer here)
    # ...
