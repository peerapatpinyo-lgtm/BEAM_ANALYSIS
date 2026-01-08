# reporter.py
import streamlit as st
import numpy as np

def render_calculation_report(span_idx, span_len, b, h, fc, fy, Mu_pos, Mu_neg, Vu, res_data, Ma_pos, delta_analysis_mm):
    """
    Generate a highly detailed Step-by-Step calculation report.
    Includes: 
    1. Design Data
    2. Flexural Strength (Pos & Neg)
    3. Shear Strength
    4. Serviceability (Deflection & Cracking) with Transformed Section Analysis
    """
    
    # --- 0. HEADER & PREPARE DATA ---
    st.markdown(f"## 📄 Detailed Calculation: Span {span_idx + 1}")
    st.markdown("---")
    
    # Unit Conversions
    b_mm = b * 1000
    h_mm = h * 1000
    L_mm = span_len * 1000
    cover = res_data.get('cover', 25)
    Es = 200000 # Steel Modulus (MPa)
    
    # --- [FIX] EXTRACT REINFORCEMENT DATA EARLY ---
    # ดึงค่าออกมาไว้ตรงนี้ เพื่อให้ทุก Section เรียกใช้ได้ ไม่ว่าจะเข้าเงื่อนไข if/else ไหน
    stir_db = res_data['stir_db']
    bot_db = res_data['bot_db']
    top_db = res_data['top_db']
    
    # --- 1. DESIGN DATA & PROPERTIES ---
    st.markdown("### 1. Design Data & Material Properties")
    
    # Determine Beta1 (ACI 318)
    if fc <= 30:
        beta1 = 0.85
    else:
        beta1 = max(0.65, 0.85 - 0.05 * (fc - 30) / 7)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**Material:**")
        st.write(f"- Concrete ($f_c'$): `{fc}` MPa")
        st.write(f"- Steel ($f_y$): `{fy}` MPa")
        st.write(f"- Modulus ($E_s$): `{Es}` MPa")
        st.write(f"- $\\beta_1$: `{beta1:.2f}`")
    with c2:
        st.write("**Section:**")
        st.write(f"- Width ($b$): `{b_mm:.0f}` mm")
        st.write(f"- Depth ($h$): `{h_mm:.0f}` mm")
        st.write(f"- Covering: `{cover}` mm")
    with c3:
        st.write("**Loads (Analysis):**")
        st.write(f"- Span Length: `{span_len:.2f}` m")
        st.write(f"- Ult. Moment ($M_u^+$): `{Mu_pos:.2f}` kNm")
        st.write(f"- Svc. Moment ($M_a$): `{Ma_pos:.2f}` kNm")

    # --- 2. FLEXURAL DESIGN (POSITIVE MOMENT) ---
    st.markdown("---")
    st.markdown("### 2. Flexural Design: Mid-Span (Positive Moment)")
    st.info("Check capacity against Maximum Positive Moment ($M_u^+$)")
    
    mu = Mu_pos
    if mu <= 0.01:
        st.write("No significant positive moment in this span.")
    else:
        bot_n = res_data['pos']['n']
        
        # 2.1 Effective Depth
        d_bot = h_mm - cover - stir_db - (bot_db/2)
        st.markdown("**2.1 Effective Depth ($d$)**")
        st.latex(rf"d = h - C_{{cov}} - d_{{stir}} - \frac{{d_b}}{{2}}")
        st.latex(rf"d = {h_mm:.0f} - {cover} - {stir_db} - \frac{{{bot_db}}}{{2}} = \mathbf{{{d_bot:.1f}}} \text{{ mm}}")

        # 2.2 Reinforcement Provided
        As_prov = bot_n * (np.pi * (bot_db/2)**2)
        rho_prov = As_prov / (b_mm * d_bot)
        st.markdown(f"**2.2 Reinforcement Provided: {bot_n}-DB{bot_db}**")
        st.latex(rf"A_{{s,prov}} = {bot_n} \times \pi \times \left(\frac{{{bot_db}}}{{2}}\right)^2 = \mathbf{{{As_prov:.0f}}} \text{{ mm}}^2")
        st.write(f"Reinforcement Ratio ($\\rho$): {rho_prov:.5f}")

        # 2.3 Moment Capacity
        st.markdown("**2.3 Moment Capacity Calculation ($\phi M_n$)**")
        
        # Depth of Stress Block (a)
        a = (As_prov * fy) / (0.85 * fc * b_mm)
        st.latex(rf"a = \frac{{A_s f_y}}{{0.85 f_c' b}} = \frac{{{As_prov:.0f} \cdot {fy}}}{{0.85 \cdot {fc} \cdot {b_mm:.0f}}} = {a:.2f} \text{{ mm}}")
        
        # Neutral Axis (c) & Strain
        c = a / beta1
        epsilon_t = 0.003 * (d_bot - c) / c
        st.write(f"Neutral Axis ($c$): {c:.2f} mm | Net Tensile Strain ($\epsilon_t$): {epsilon_t:.4f}")
        
        # Phi Factor
        if epsilon_t >= 0.005:
            phi = 0.9
            st.write("$\epsilon_t \ge 0.005 \Rightarrow$ Tension Controlled, $\phi = 0.9$")
        else:
            phi = 0.65 + 0.25*((epsilon_t - 0.002)/0.003)
            st.write(f"Transition Region, $\phi = {phi:.3f}$")

        # Nominal Moment
        Mn = As_prov * fy * (d_bot - a/2) * 1e-6 # kNm
        phiMn = phi * Mn
        
        st.latex(rf"\phi M_n = \phi \cdot A_s f_y (d - \frac{{a}}{{2}})")
        st.latex(rf"\phi M_n = {phi} \cdot {As_prov:.0f} \cdot {fy} \cdot ({d_bot:.1f} - \frac{{{a:.2f}}}{{2}}) \cdot 10^{{-6}} = \mathbf{{{phiMn:.2f}}} \text{{ kNm}}")
        
        # Check
        if phiMn >= mu:
            st.success(f"✅ PASS: Capacity ({phiMn:.2f} kNm) > Demand ({mu:.2f} kNm)")
        else:
            st.error(f"❌ FAIL: Capacity ({phiMn:.2f} kNm) < Demand ({mu:.2f} kNm)")

    # --- 3. FLEXURAL DESIGN (NEGATIVE MOMENT) ---
    st.markdown("---")
    st.markdown("### 3. Flexural Design: Support (Negative Moment)")
    
    mu_neg_val = Mu_neg
    if mu_neg_val <= 0.01:
        st.info("No significant negative moment.")
    else:
        top_n = res_data['neg']['n']
        d_top = h_mm - cover - stir_db - (top_db/2)
        As_top = top_n * (np.pi * (top_db/2)**2)
        
        st.write(f"**Demand:** $M_u^- =$ **{mu_neg_val:.2f}** kNm")
        st.write(f"**Provide:** {top_n}-DB{top_db} ($A_s = {As_top:.0f}$ mm$^2$)")
        
        a_top = (As_top * fy) / (0.85 * fc * b_mm)
        phiMn_top = 0.9 * As_top * fy * (d_top - a_top/2) * 1e-6
        
        st.latex(rf"\phi M_n = \mathbf{{{phiMn_top:.2f}}} \text{{ kNm}}")
        
        if phiMn_top >= mu_neg_val:
            st.success(f"✅ PASS: {phiMn_top:.2f} > {mu_neg_val:.2f}")
        else:
            st.error(f"❌ FAIL: Insufficient Top Steel")

    # --- 4. SHEAR DESIGN ---
    st.markdown("---")
    st.markdown("### 4. Shear Design Checks")
    
    vu = res_data['Vu_max']
    stir_s = res_data['shear']['s']
    
    # Use d_bot for shear calc (conservative/standard)
    # If d_bot is not calculated (e.g. no pos moment), calculate it now
    if 'd_bot' not in locals():
        d_bot = h_mm - cover - stir_db - (bot_db/2)

    Av = 2 * (np.pi * (stir_db/2)**2)
    
    st.markdown("**4.1 Concrete Capacity ($\phi V_c$)**")
    Vc = 0.17 * np.sqrt(fc) * b_mm * d_bot / 1000.0 # kN
    phiVc = 0.85 * Vc
    st.latex(rf"\phi V_c = 0.85 \cdot 0.17\sqrt{{f_c'}} b d = \mathbf{{{phiVc:.2f}}} \text{{ kN}}")
    
    st.markdown("**4.2 Steel Required ($\phi V_s$)**")
    st.write(f"Max Shear $V_u$ = **{vu:.2f}** kN")
    
    if vu > phiVc:
        req_Vs_phi = vu - phiVc
        st.write(f"Shear reinforcement required for remaining: {req_Vs_phi:.2f} kN")
    else:
        st.write("Concrete alone is theoretically sufficient, but min stirrups provided.")
    
    st.markdown(f"**4.3 Provided Stirrups: RB{stir_db} @ {stir_s} mm**")
    Vs_prov = (Av * fy * d_bot) / stir_s / 1000.0
    phiVn = phiVc + (0.85 * Vs_prov)
    
    st.latex(rf"\phi V_n = \phi V_c + \phi \frac{{A_v f_y d}}{{s}} = {phiVc:.2f} + 0.85\frac{{{Av:.1f} \cdot {fy} \cdot {d_bot:.1f}}}{{{stir_s}}} = \mathbf{{{phiVn:.2f}}} \text{{ kN}}")
    
    if phiVn >= vu:
        st.success("✅ PASS: Shear Capacity OK")
    else:
        st.error("❌ FAIL: Shear Capacity Insufficient")

    # --- 5. SERVICEABILITY (DEFLECTION) ---
    st.markdown("---")
    st.markdown("### 5. Serviceability & Deflection Check (Detailed)")
    st.info("Methodology: ACI 318 / EIT (Effective Moment of Inertia - Branson's Formula)")

    # 5.1 Material Modulus
    Ec = 4700 * np.sqrt(fc)
    n = Es / Ec
    st.markdown("**5.1 Modulus of Elasticity & Modular Ratio**")
    st.latex(rf"E_c = 4700\sqrt{{f_c'}} = 4700\sqrt{{{fc}}} = {Ec:.0f} \text{{ MPa}}")
    st.latex(rf"n = \frac{{E_s}}{{E_c}} = \frac{{{Es}}}{{{Ec:.0f}}} = {n:.2f}")

    # 5.2 Gross Section Properties (Uncracked)
    Ig = (b_mm * h_mm**3) / 12
    fr = 0.62 * np.sqrt(fc) 
    yt = h_mm / 2
    Mcr = (fr * Ig) / yt * 1e-6 
    
    st.markdown("**5.2 Uncracked Properties & Cracking Moment**")
    st.latex(rf"I_g = \frac{{bh^3}}{{12}} = {Ig:.2e} \text{{ mm}}^4")
    st.latex(rf"M_{{cr}} = \frac{{f_r I_g}}{{y_t}} = \frac{{{fr:.2f} \cdot {Ig:.2e}}}{{{yt:.0f}}} = \mathbf{{{Mcr:.2f}}} \text{{ kNm}}")

    # 5.3 Effective Inertia (Ie)
    st.markdown("**5.3 Effective Moment of Inertia ($I_e$)**")
    st.write(f"Service Moment ($M_a$): **{Ma_pos:.2f}** kNm")
    
    # Calculate As_prov again if not in locals (case where Mu_pos was 0)
    if 'As_prov' not in locals():
         # Default to min steel or actual user input for calculation purposes
         bot_n = res_data['pos']['n']
         As_prov = bot_n * (np.pi * (bot_db/2)**2)

    if Ma_pos < Mcr:
        st.success(f"Condition: $M_a < M_{{cr}}$ $\\to$ Section is **Uncracked**")
        Ie = Ig
        st.latex(r"I_e = I_g")
    else:
        st.warning(f"Condition: $M_a > M_{{cr}}$ $\\to$ Section is **Cracked**")
        
        # Transformed Section Analysis
        rho = As_prov / (b_mm * d_bot)
        rn = rho * n
        # k = sqrt((rn)^2 + 2rn) - rn
        k = np.sqrt(rn**2 + 2*rn) - rn
        kd = k * d_bot
        
        st.markdown("*Transformed Section Analysis:*")
        st.write(f"- Neutral Axis depth ($kd$): {kd:.2f} mm")
        
        # Icr Calculation
        Icr = (b_mm * kd**3)/3 + n * As_prov * (d_bot - kd)**2
        st.latex(rf"I_{{cr}} = \frac{{b(kd)^3}}{{3}} + n A_s (d-kd)^2 = {Icr:.2e} \text{{ mm}}^4")
        
        # Branson's Equation
        term = (Mcr / Ma_pos)**3
        Ie = term * Ig + (1 - term) * Icr
        Ie = min(Ie, Ig) # Can't be greater than Ig
        
        st.markdown("*Branson's Formula:*")
        st.latex(r"I_e = \left(\frac{M_{cr}}{M_a}\right)^3 I_g + \left[1 - \left(\frac{M_{cr}}{M_a}\right)^3\right] I_{cr}")
        st.latex(rf"I_e = {term:.3f} I_g + {1-term:.3f} I_{{cr}} = \mathbf{{{Ie:.2e}}} \text{{ mm}}^4")
        st.write(f"Reduction Factor ($I_e/I_g$): {Ie/Ig:.3f}")

    # 5.4 Deflection Calculation
    st.markdown("**5.4 Deflection Calculation**")
    st.write("Using elastic deflection from analysis ($\Delta_{elastic}$) adjusted by stiffness ratio.")
    
    st.write(f"Elastic Analysis Deflection (based on $I_g$): **{delta_analysis_mm:.2f}** mm")
    
    # Magnification
    magnification = Ig / Ie
    delta_imm = delta_analysis_mm * magnification
    
    st.latex(rf"\Delta_{{immediate}} = \Delta_{{elastic}} \times \frac{{I_g}}{{I_e}} = {delta_analysis_mm:.2f} \times {magnification:.2f} = \mathbf{{{delta_imm:.2f}}} \text{{ mm}}")
    
    # Long-term Deflection
    st.markdown("**5.5 Long-term Deflection**")
    xi = 2.0 # Duration > 5 years
    rho_prime = 0 # Conservative (assume no compression steel for deflection)
    lambda_delta = xi / (1 + 50*rho_prime)
    
    delta_long = lambda_delta * delta_imm
    delta_total = delta_imm + delta_long
    
    st.latex(rf"\lambda_\Delta = \frac{{\xi}}{{1+50\rho'}} = {lambda_delta} \quad (\text{{Assume }} \rho'=0)")
    st.latex(rf"\Delta_{{total}} = \Delta_i + \lambda \Delta_i = {delta_imm:.2f} + {delta_long:.2f} = \mathbf{{{delta_total:.2f}}} \text{{ mm}}")
    
    # 5.6 Limits Check
    st.markdown("**5.6 Compliance Check**")
    
    limit_180 = L_mm / 180
    limit_240 = L_mm / 240
    limit_360 = L_mm / 360
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.metric("Total Deflection", f"{delta_total:.2f} mm")
    with col_b:
        st.metric("Limit L/240", f"{limit_240:.2f} mm")
        if delta_total <= limit_240:
             st.success("✅ PASS (L/240)")
        else:
             st.error("❌ FAIL (L/240)")
    with col_c:
        st.metric("Limit L/180", f"{limit_180:.2f} mm")
        if delta_total <= limit_180:
             st.success("✅ PASS (L/180)")
        else:
             st.error("❌ FAIL (L/180)")
             
    st.caption("Note: L/240 is typical limit for total load. L/360 is for live load only.")
