import streamlit as st
import numpy as np

def render_calculation_report(res):
    """
    Detailed ACI 318-19 Calculation Report with Step-by-Step Substitution.
    """
    # --- Data Extraction ---
    idx = res['span_id'] + 1
    L_m = res['L']
    b = res['b'] 
    h = res['h'] 
    cov = res['cover']
    fc = res['fc']
    fy = res['fy']
    
    Mu = res['Mu_pos']
    Vu = res['Vu_max']
    Ma = res['Ma_pos_svc']   
    delta_svc = res['delta_svc_mm'] 
    
    bot_n, bot_db = res['bot']['n'], res['bot']['db']
    stir_db, stir_s = res['shear']['db'], res['shear']['s']

    # --- Constants & Material Properties ---
    Es = 200000.0 
    Ec = 4700 * np.sqrt(fc)
    n_mod = Es / Ec
    
    # Beta1 factor (ACI 318-19 Table 22.2.2.4.3)
    if fc <= 28: beta1 = 0.85
    elif fc >= 55: beta1 = 0.65
    else: beta1 = 0.85 - (0.05 * (fc - 28) / 7)

    st.markdown(f"## 📋 Engineering Calculation Sheet: Span {idx}")
    st.markdown(f"**Location:** $x = 0$ to $L = {L_m:.2f}$ m")
    st.divider()

    # =========================================================
    # 1. DESIGN PARAMETERS
    # =========================================================
    st.markdown("### 1. Materials & Geometry")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Concrete & Steel:**")
        st.latex(rf"f'_c = {fc} \text{{ MPa}}, \quad f_y = {fy} \text{{ MPa}}")
        st.latex(rf"E_s = 200,000 \text{{ MPa}}, \quad \beta_1 = {beta1:.3f}")
    with col2:
        st.write("**Section Dimensions:**")
        st.latex(rf"b = {b:.0f} \text{{ mm}}, \quad h = {h:.0f} \text{{ mm}}")
        st.latex(rf"Cover = {cov} \text{{ mm}}")

    # =========================================================
    # 2. FLEXURAL DESIGN (ACI 318-19)
    # =========================================================
    st.markdown("### 2. Flexural Strength (Positive Moment)")
    st.info(f"**Required Strength:** $M_u = {Mu:.2f}$ kNm")

    # 2.1 Effective Depth and Area
    d = h - cov - stir_db - (bot_db/2)
    As = bot_n * (np.pi * (bot_db/2)**2)
    As_min = (np.sqrt(fc)/(4*fy)) * b * d if (np.sqrt(fc)/(4*fy)) > (1.4/fy) else (1.4/fy) * b * d
    
    st.markdown("**2.1 Reinforcement Check**")
    st.latex(rf"d = h - d_{{cover}} - \phi_{{stirrup}} - \frac{{\phi_b}}{{2}} = {h} - {cov} - {stir_db} - \frac{{{bot_db}}}{{2}} = \mathbf{{{d:.1f}}}\text{{ mm}}")
    st.latex(rf"A_s = n \cdot \frac{{\pi \cdot \phi_b^2}}{{4}} = {bot_n} \cdot \frac{{\pi \cdot {bot_db}^2}}{{4}} = \mathbf{{{As:.1f}}}\text{{ mm}}^2")
    
    # 2.2 Capacity Calculation
    a = (As * fy) / (0.85 * fc * b)
    c = a / beta1
    strain_t = 0.003 * (d - c) / c
    
    # Phi calculation
    if strain_t >= 0.005: phi_flex = 0.90
    elif strain_t <= 0.002: phi_flex = 0.65
    else: phi_flex = 0.65 + 0.25 * (strain_t - 0.002)/0.003

    Mn = As * fy * (d - a/2) * 1e-6
    phiMn = phi_flex * Mn

    st.markdown("**2.2 Stress Block & Capacity**")
    st.latex(rf"a = \frac{{A_s \cdot f_y}}{{0.85 \cdot f'_c \cdot b}} = \frac{{{As:.1f} \cdot {fy}}}{{0.85 \cdot {fc} \cdot {b}}} = \mathbf{{{a:.2f}}}\text{{ mm}}")
    st.latex(rf"\epsilon_t = 0.003 \left( \frac{{d - c}}{{c}} \right) = \mathbf{{{strain_t:.4f}}} \rightarrow \phi = {phi_flex:.2f}")
    
    st.latex(rf"\phi M_n = \phi [A_s f_y (d - a/2)] = {phi_flex:.2f} [{As:.0f} \cdot {fy} ({d:.1f} - {a/2:.1f})] \cdot 10^{{-6}}")
    
    # Result Comparison
    if phiMn >= Mu:
        st.success(rf"**Design Summary:** $\phi M_n = {phiMn:.2f} \text{{ kNm}} \ge M_u = {Mu:.2f} \text{{ kNm}}$ ✅ **PASS**")
    else:
        st.error(rf"**Design Summary:** $\phi M_n = {phiMn:.2f} \text{{ kNm}} < M_u = {Mu:.2f} \text{{ kNm}}$ ❌ **FAIL**")

    # =========================================================
    # 3. SHEAR DESIGN (ACI 318-19)
    # =========================================================
    st.markdown("### 3. Shear Strength")
    st.info(f"**Required Strength:** $V_u = {Vu:.2f}$ kN")

    lambda_v = 1.0
    Vc = (0.17 * lambda_v * np.sqrt(fc) * b * d) / 1000  # kN
    
    # Stirrup capacity
    Av = 2 * (np.pi * (stir_db/2)**2) # 2 legs
    Vs = (Av * fy * d / stir_s) / 1000 # kN
    phiVn = 0.75 * (Vc + Vs)

    st.markdown("**3.1 Concrete & Steel Contribution**")
    st.latex(rf"V_c = 0.17 \lambda \sqrt{{f'_c}} b_w d = 0.17 \cdot 1.0 \cdot \sqrt{{{fc}}} \cdot {b} \cdot {d:.1f} = \mathbf{{{Vc:.2f}}}\text{{ kN}}")
    st.latex(rf"V_s = \frac{{A_v f_{{yt}} d}}{{s}} = \frac{{{Av:.1f} \cdot {fy} \cdot {d:.1f}}}{{{stir_s}}} = \mathbf{{{Vs:.2f}}}\text{{ kN}}")
    
    st.latex(rf"\phi V_n = 0.75(V_c + V_s) = 0.75({Vc:.2f} + {Vs:.2f}) = \mathbf{{{phiVn:.2f}}}\text{{ kN}}")

    if phiVn >= Vu:
        st.success(rf"**Design Summary:** $\phi V_n = {phiVn:.2f} \text{{ kN}} \ge V_u = {Vu:.2f} \text{{ kN}}$ ✅ **PASS**")
    else:
        st.error(rf"**Design Summary:** $\phi V_n = {phiVn:.2f} \text{{ kN}} < V_u = {Vu:.2f} \text{{ kN}}$ ❌ **FAIL**")

    # =========================================================
    # 4. SERVICEABILITY
    # =========================================================
    st.markdown("### 4. Serviceability (Deflection)")
    
    L_mm = L_m * 1000
    l_over_240 = L_mm / 240
    
    st.latex(rf"\delta_{{allow}} = L/240 = {L_mm:.0f}/240 = \mathbf{{{l_over_240:.2f}}}\text{{ mm}}")
    st.latex(rf"\delta_{{actual}} = \mathbf{{{delta_svc:.3f}}}\text{{ mm}}")

    if abs(delta_svc) <= l_over_240:
        st.success(f"**Deflection Check:** Actual < Allowable ✅ **PASS**")
    else:
        st.warning(f"**Deflection Check:** Actual > Allowable ⚠️ **EXCEEDED**")
