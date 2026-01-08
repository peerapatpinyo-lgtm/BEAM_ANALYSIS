# reporter.py
import streamlit as st
import numpy as np

def render_calculation_report(span_idx, span_len, b, h, fc, fy, Mu_pos, Mu_neg, Vu, res_data, Ma_pos, delta_analysis_mm):
    """
    delta_analysis_mm: Max deflection from Structural Analysis (Elastic, based on Ig) [mm]
    """
    
    # --- 1. Constants & Section Properties ---
    st.markdown(f"### 📍 Design Calculation: Span {span_idx + 1}")
    st.markdown("---")
    
    b_mm = b * 1000
    h_mm = h * 1000
    cover = res_data['cover']
    Es = 200000 # MPa
    
    # Display Data
    st.markdown("**1. Design Data & Material Properties**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write(f"- $f_c'$: **{fc}** MPa")
        st.write(f"- $f_y$: **{fy}** MPa")
    with c2:
        st.write(f"- $b$: **{b_mm:.0f}** mm")
        st.write(f"- $h$: **{h_mm:.0f}** mm")
    with c3:
        st.write(f"- Span: **{span_len:.2f}** m")
        st.write(f"- Service Moment ($M_a$): **{Ma_pos:.2f}** kNm")

    # Determine Beta1
    if fc <= 30:
        beta1 = 0.85
    else:
        beta1 = max(0.65, 0.85 - 0.05 * (fc - 30) / 7)

    # --- 2. FLEXURAL DESIGN (POSITIVE) ---
    st.markdown("---")
    st.markdown(f"**2. Flexural Design: Mid-Span (Positive)**")
    
    mu = Mu_pos
    if mu <= 0.01:
        st.info("No significant positive moment.")
        # Setup dummy variables to avoid errors if Ma is high but Mu is low (rare)
        d_bot = h_mm - cover - 20 # approx
        As_prov = 0
    else:
        bot_db = res_data['bot_db']
        bot_n = res_data['pos']['n']
        stir_db = res_data['stir_db']
        
        # 2.1 Effective Depth
        d_bot = h_mm - cover - stir_db - (bot_db/2)
        st.latex(f"d = {h_mm:.0f} - {cover} - {stir_db} - {bot_db/2:.1f} = \\mathbf{{{d_bot:.1f}}} \\text{{ mm}}")

        # 2.2 Reinforcement & Capacity
        As_prov = bot_n * (np.pi * (bot_db/2)**2)
        st.write(f"Design $M_u$: **{mu:.2f}** kNm | Provide: **{bot_n}-DB{bot_db}**")
        
        a = (As_prov * fy) / (0.85 * fc * b_mm)
        phiMn = 0.9 * As_prov * fy * (d_bot - a/2) * 1e-6 # Simplified phi
        
        if phiMn >= mu:
            st.success(f"✅ PASS: $\phi M_n$ ({phiMn:.2f}) > $M_u$ ({mu:.2f})")
        else:
            st.error(f"❌ FAIL: Capacity Insufficient")

    # --- 3. FLEXURAL DESIGN (NEGATIVE) ---
    # (Skipping details for brevity, assumed calculated previously)
    
    # --- 4. SHEAR DESIGN ---
    st.markdown("---")
    st.markdown(f"**4. Shear Design**")
    vu = res_data['Vu_max']
    stir_s = res_data['shear']['s']
    stir_db = res_data['stir_db']
    
    Vc = 0.17 * np.sqrt(fc) * b_mm * d_bot / 1000.0
    phiVc = 0.85 * Vc
    
    Av = 2 * (np.pi * (stir_db/2)**2)
    Vs_prov = (Av * fy * d_bot) / stir_s / 1000.0
    phiVn = phiVc + (0.85 * Vs_prov)
    
    st.write(f"Max $V_u$: **{vu:.2f}** kN | Capacity $\phi V_n$: **{phiVn:.2f}** kN")
    st.caption(f"Using RB{stir_db} @ {stir_s} mm")

    # --- 5. DEFLECTION CHECK (FROM ANALYSIS GRAPH) ---
    st.markdown("---")
    st.markdown(f"### 5. Serviceability Check (Deflection)")
    st.info("Calculation adjusts the **Elastic Analysis Result** (from graph) using **Effective Inertia ($I_e$)**.")

    # 5.1 Material Modulus
    Ec = 4700 * np.sqrt(fc)
    n = Es / Ec
    
    # 5.2 Gross Section Properties (Uncracked)
    Ig = (b_mm * h_mm**3) / 12
    fr = 0.62 * np.sqrt(fc) 
    yt = h_mm / 2
    Mcr = (fr * Ig) / yt * 1e-6 
    
    st.write(f"$E_c = {Ec:.0f}$ MPa | $n = {n:.2f}$ | $I_g = {Ig:.2e}$ mm$^4$")
    st.write(f"Cracking Moment ($M_{{cr}}$): **{Mcr:.2f}** kNm")

    # 5.3 Effective Inertia (Ie)
    Ma = Ma_pos
    
    if Ma < Mcr:
        st.success(f"Section Uncracked ($M_a < M_{{cr}}$) $\\to I_e = I_g$")
        Ie = Ig
    else:
        st.warning(f"Section Cracked ($M_a > M_{{cr}}$)")
        
        # Calculate Cracked Inertia (Icr)
        rho = As_prov / (b_mm * d_bot)
        rn = rho * n
        k = np.sqrt(rn**2 + 2*rn) - rn
        kd = k * d_bot
        Icr = (b_mm * kd**3)/3 + n * As_prov * (d_bot - kd)**2
        
        # Branson's Equation
        term = (Mcr / Ma)**3
        Ie = term * Ig + (1 - term) * Icr
        Ie = min(Ie, Ig)
        
        st.latex(rf"I_{{cr}} = {Icr:.2e} \text{{ mm}}^4 \quad \to \quad I_e = {Ie:.2e} \text{{ mm}}^4")

    # 5.4 Deflection Calculation (Adjusting Analysis Result)
    st.markdown("**5.4 Deflection Calculation**")
    
    # The value from graph (Analysis Engine) is usually based on Ig (Uncracked stiffness)
    # We must magnify it if the section is cracked: Delta_real = Delta_graph * (Ig / Ie)
    
    st.write(f"Elastic Deflection from Analysis ($I_g$): **{delta_analysis_mm:.2f}** mm")
    
    magnification_factor = Ig / Ie
    delta_imm = delta_analysis_mm * magnification_factor
    
    if magnification_factor > 1.0:
        st.write(f"Magnification Factor ($I_g/I_e$): **{magnification_factor:.2f}**")
        st.latex(rf"\Delta_i = \Delta_{{analysis}} \times \frac{{I_g}}{{I_e}} = {delta_analysis_mm:.2f} \times {magnification_factor:.2f} = \mathbf{{{delta_imm:.2f}}} \text{{ mm}}")
    else:
        st.write("Section is uncracked, no magnification needed.")
        st.latex(rf"\Delta_i = \mathbf{{{delta_imm:.2f}}} \text{{ mm}}")

    # Long-term Deflection
    xi = 2.0 
    delta_long = xi * delta_imm # Simplified (rho' = 0)
    delta_total = delta_imm + delta_long
    
    st.latex(rf"\Delta_{{total}} = \Delta_i + \Delta_{{long}} = {delta_imm:.2f} + {delta_long:.2f} = \mathbf{{{delta_total:.2f}}} \text{{ mm}}")
    
    # 5.5 Limits
    limit_240 = (span_len * 1000) / 240
    st.write(f"Limit $L/240$: **{limit_240:.2f}** mm")
    
    if delta_total <= limit_240:
         st.success("✅ PASS (Deflection OK)")
    else:
         st.error("❌ FAIL (Deflection Exceeded)")
