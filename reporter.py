# reporter.py
import streamlit as st
import numpy as np

def render_calculation_report(span_idx, span_len, b, h, fc, fy, Mu_pos, Mu_neg, Vu, res_data):
    """
    Generates a detailed Step-by-Step Calculation Sheet in English using LaTeX.
    """
    
    # --- 1. Extract & Convert Variables ---
    # Concrete Cover
    cover = 30 # mm
    
    # Provided Reinforcement Data
    top_n = res_data['neg']['n']
    top_db = res_data['top_db']
    bot_n = res_data['pos']['n']
    bot_db = res_data['bot_db']
    stir_db = res_data['stir_db']
    stir_s = res_data['shear']['s']

    # Geometry Conversions (m -> mm)
    b_mm = b * 1000
    h_mm = h * 1000
    
    # Effective Depth (d) Estimation
    # d = h - cover - stirrup - (bar_dia / 2)
    d_est = h_mm - cover - stir_db - (max(top_db, bot_db)/2) 

    # Area of Steel Provided (mm^2)
    As_top = top_n * (np.pi * (top_db**2) / 4)
    As_bot = bot_n * (np.pi * (bot_db**2) / 4)
    Av = 2 * (np.pi * (stir_db**2) / 4) # 2 legs for stirrups

    # Header
    st.markdown(f"### 📄 Detailed Calculation: Span {span_idx+1} (L = {span_len} m)")
    st.markdown("---")

    # =========================================================================
    # Section 1: Design Parameters
    # =========================================================================
    st.markdown("#### 1. Design Parameters")
    st.latex(r'''
    \begin{aligned}
    f'_c &= ''' + f"{fc:.2f}" + r''' \text{ MPa} \\
    f_y &= ''' + f"{fy:.0f}" + r''' \text{ MPa} \\
    b \times h &= ''' + f"{b_mm:.0f} \\times {h_mm:.0f}" + r''' \text{ mm} \\
    \text{Cover} &= ''' + f"{cover}" + r''' \text{ mm} \\
    \text{Est. Effective Depth } (d) &\approx ''' + f"{d_est:.1f}" + r''' \text{ mm}
    \end{aligned}
    ''')

    # =========================================================================
    # Section 2: Flexural Design (Positive Moment)
    # =========================================================================
    st.markdown("#### 2. Flexural Design: Positive Moment (Mid-Span)")
    st.info(f"**Design Moment:** $M_u^+ = {Mu_pos:.2f}$ kNm")

    st.markdown("**2.1 Provided Reinforcement**")
    st.markdown(f"- **Bottom Bars:** {int(bot_n)}-DB{int(bot_db)}")
    st.latex(r"A_{s,prov} = " + f"{bot_n} \\times \\frac{{\\pi ({bot_db})^2}}{{4}} = {As_bot:.2f} \\text{{ mm}}^2")

    st.markdown("**2.2 Moment Capacity Check ($\phi M_n$)**")
    
    # Calculate Whitney Stress Block (a)
    a_depth = (As_bot * fy) / (0.85 * fc * b_mm)
    st.markdown("Calculate depth of equivalent stress block ($a$):")
    st.latex(r"a = \frac{A_s f_y}{0.85 f'_c b} = " + f"\\frac{{{As_bot:.2f} \\cdot {fy}}}{{0.85 \\cdot {fc} \\cdot {b_mm}}} = {a_depth:.2f} \\text{{ mm}}")

    # Calculate Nominal Moment (Mn)
    d_actual = h_mm - cover - stir_db - (bot_db/2)
    Mn = As_bot * fy * (d_actual - (a_depth/2)) / 1e6 # Convert N-mm to kNm
    phi = 0.90 # Reduction factor for tension-controlled
    phi_Mn = phi * Mn

    st.markdown("Calculate Design Moment Capacity ($\phi M_n$):")
    st.latex(r"\phi M_n = \phi A_s f_y \left(d - \frac{a}{2}\right)")
    st.latex(f"= 0.90 \\cdot {As_bot:.2f} \\cdot {fy} \\cdot ({d_actual:.2f} - \\frac{{{a_depth:.2f}}}{{2}}) \\cdot 10^{{-6}}")
    st.latex(f"= \\mathbf{{{phi_Mn:.2f}}} \\text{{ kNm}}")

    # Conclusion
    if phi_Mn >= Mu_pos:
        st.success(f"✅ **OK**: Capacity ({phi_Mn:.2f} kNm) > Demand ({Mu_pos:.2f} kNm)")
    else:
        st.error(f"❌ **FAIL**: Insufficient Capacity (Req: {Mu_pos:.2f} kNm)")

    st.markdown("---")

    # =========================================================================
    # Section 3: Flexural Design (Negative Moment)
    # =========================================================================
    st.markdown("#### 3. Flexural Design: Negative Moment (Supports)")
    st.info(f"**Design Moment:** $M_u^- = {abs(Mu_neg):.2f}$ kNm")

    if abs(Mu_neg) < 1.0:
        st.caption("Moment is negligible. Minimum reinforcement governs.")
    else:
        st.markdown("**3.1 Provided Reinforcement**")
        st.markdown(f"- **Top Bars:** {int(top_n)}-DB{int(top_db)}")
        st.latex(r"A_{s,top} = " + f"{As_top:.2f} \\text{{ mm}}^2")

        # Calculate a
        a_depth_neg = (As_top * fy) / (0.85 * fc * b_mm)
        
        # Calculate Mn
        d_actual_top = h_mm - cover - stir_db - (top_db/2)
        Mn_neg = As_top * fy * (d_actual_top - (a_depth_neg/2)) / 1e6
        phi_Mn_neg = 0.90 * Mn_neg
        
        st.markdown("**3.2 Capacity Check**")
        st.latex(f"\\phi M_n = \\mathbf{{{phi_Mn_neg:.2f}}} \\text{{ kNm}}")

        if phi_Mn_neg >= abs(Mu_neg):
            st.success(f"✅ **OK**: Capacity ({phi_Mn_neg:.2f} kNm) > Demand ({abs(Mu_neg):.2f} kNm)")
        else:
            st.error("❌ **FAIL**: Insufficient Capacity")

    st.markdown("---")

    # =========================================================================
    # Section 4: Shear Design
    # =========================================================================
    st.markdown("#### 4. Shear Design")
    
    Vu_design = abs(Vu)
    st.info(f"**Design Shear Force:** $V_u = {Vu_design:.2f}$ kN")

    # 4.1 Concrete Capacity (Vc)
    st.markdown("**4.1 Concrete Shear Capacity ($\phi V_c$)**")
    # Simplified Vc formula
    Vc = 0.17 * np.sqrt(fc) * b_mm * d_est / 1000.0 # kN
    phi_v = 0.85 
    phi_Vc = phi_v * Vc

    st.latex(r"V_c = 0.17 \sqrt{f'_c} b d")
    st.latex(f"= 0.17 \\cdot \\sqrt{{{fc}}} \\cdot {b_mm} \\cdot {d_est:.1f} \\cdot 10^{{-3}} = {Vc:.2f} \\text{{ kN}}")
    st.latex(f"\\phi V_c = 0.85 \\cdot {Vc:.2f} = \\mathbf{{{phi_Vc:.2f}}} \\text{{ kN}}")

    # 4.2 Stirrup Requirement
    st.markdown("**4.2 Stirrup Design**")
    
    if phi_Vc >= Vu_design:
        st.success(f"✅ **OK**: Concrete alone is sufficient ($\phi V_c > V_u$). Minimum stirrups provided.")
        st.latex(f"\\text{{Use }} \\mathbf{{RB{int(stir_db)} @ {int(stir_s)} mm}} \\text{{ (Minimum)}}")
    else:
        st.warning(f"⚠️ **Reinforcement Needed**: $\phi V_c < V_u$. Stirrups must carry excess shear.")
        Vs_req = (Vu_design - phi_Vc) / phi_v
        st.latex(f"V_s,req = \\frac{{V_u - \phi V_c}}{{\phi}} = \\frac{{{Vu_design:.2f} - {phi_Vc:.2f}}}{{0.85}} = {Vs_req:.2f} \\text{{ kN}}")
        
        # Check Capacity of Provided Stirrups
        # Vs = Av * fy * d / s
        Vs_prov = (Av * fy * d_est) / stir_s / 1000.0 # kN
        
        st.markdown("**Check Provided Stirrups:**")
        st.latex(r"V_s = \frac{A_v f_y d}{s} = " + f"\\frac{{{Av:.2f} \\cdot {fy} \\cdot {d_est:.1f}}}{{{stir_s}}} \\cdot 10^{{-3}} = {Vs_prov:.2f} \\text{{ kN}}")
        
        phi_Vn = phi_Vc + (phi_v * Vs_prov)
        
        st.markdown(f"**Total Capacity ($\phi V_n$):** {phi_Vn:.2f} kN")
        
        if phi_Vn >= Vu_design:
            st.success(f"✅ **OK**: Total Capacity ({phi_Vn:.2f} kN) > Demand ({Vu_design:.2f} kN)")
            st.latex(f"\\text{{Use }} \\mathbf{{RB{int(stir_db)} @ {int(stir_s)} mm}}")
        else:
            st.error("❌ **FAIL**: Stirrup spacing is too wide or diameter too small.")
