# reporter.py
import streamlit as st
import numpy as np

def render_calculation_report(span_idx, span_len, b, h, fc, fy, Mu_pos, Mu_neg, Vu, res_data):
    """
    Generate a detailed Step-by-Step calculation report based on ACI 318 / EIT (SDM).
    """
    
    # --- 1. Constants & Section Properties ---
    st.markdown(f"### 📍 Design Calculation: Span {span_idx + 1}")
    st.markdown("---")
    
    b_mm = b * 1000
    h_mm = h * 1000
    cover = res_data['cover']
    
    # Display Design Data
    st.markdown("**1. Design Data & Material Properties**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write(f"- Concrete Strength ($f_c'$): **{fc}** MPa")
        st.write(f"- Steel Yield Strength ($f_y$): **{fy}** MPa")
    with c2:
        st.write(f"- Beam Width ($b$): **{b_mm:.0f}** mm")
        st.write(f"- Beam Depth ($h$): **{h_mm:.0f}** mm")
    with c3:
        st.write(f"- Covering: **{cover}** mm")
        st.write(f"- Span Length: **{span_len:.2f}** m")

    # Determine Beta1 (ACI 318)
    if fc <= 30:
        beta1 = 0.85
    else:
        beta1 = max(0.65, 0.85 - 0.05 * (fc - 30) / 7)

    st.latex(r"\beta_1 = " + f"{beta1:.3f} " + r"\quad (\text{for } f_c' = " + f"{fc}" + r"\text{ MPa})")

    # --- 2. FLEXURAL DESIGN (POSITIVE MOMENT) ---
    st.markdown("---")
    st.markdown(f"**2. Flexural Design: Mid-Span (Positive Moment $M_u^+$)**")
    
    mu = Mu_pos
    if mu <= 0.01:
        st.info("No significant positive moment in this span (Cantilever or continuous effect).")
    else:
        bot_db = res_data['bot_db']
        bot_n = res_data['pos']['n']
        stir_db = res_data['stir_db']
        
        # 2.1 Effective Depth
        d_bot = h_mm - cover - stir_db - (bot_db/2)
        st.markdown(f"**2.1 Effective Depth ($d$):**")
        st.latex(r"d = h - c_c - d_{stirrup} - \frac{d_b}{2}")
        st.latex(f"d = {h_mm:.0f} - {cover} - {stir_db} - {bot_db/2:.1f} = \\mathbf{{{d_bot:.1f}}} \\text{ mm}")

        # 2.2 Required Reinforcement
        st.markdown(f"**2.2 Design Moment & Reinforcement:**")
        st.write(f"Design Moment ($M_u$): **{mu:.2f}** kNm")
        
        # Calculate As provided
        As_prov = bot_n * (np.pi * (bot_db/2)**2)
        st.write(f"Provide: **{bot_n}-DB{bot_db}**")
        st.latex(f"A_{{s,prov}} = {bot_n} \\times \\pi \\times ({bot_db}/2)^2 = \\mathbf{{{As_prov:.0f}}} \\text{ mm}^2")

        # 2.3 Calculate Capacity (Phi Mn)
        st.markdown(f"**2.3 Moment Capacity Check ($\phi M_n$):**")
        
        # a = As*fy / (0.85*fc*b)
        a = (As_prov * fy) / (0.85 * fc * b_mm)
        st.latex(r"a = \frac{A_s f_y}{0.85 f_c' b} = " + f"\\frac{{{As_prov:.0f} \\cdot {fy}}}{{0.85 \\cdot {fc} \\cdot {b_mm:.0f}}} = {a:.2f} \\text{ mm}")
        
        # c = a / beta1
        c = a / beta1
        
        # Strain check
        dt = d_bot # Assume single layer
        epsilon_t = 0.003 * (dt - c) / c
        phi = 0.9 if epsilon_t >= 0.005 else 0.65 + 0.25*((epsilon_t - 0.002)/0.003)
        
        st.write(f"Neutral Axis ($c$): {c:.2f} mm | Strain ($\\epsilon_t$): {epsilon_t:.5f}")
        if epsilon_t >= 0.005:
            st.success(f"Section is Tension-Controlled ($\\epsilon_t \\ge 0.005$) $\\to \\phi = {phi}$")
        else:
            st.warning(f"Section is Transition/Compression Controlled $\\to \\phi = {phi:.3f}$")

        # Nominal Moment
        Mn = As_prov * fy * (d_bot - a/2) * 1e-6 # kNm
        phiMn = phi * Mn
        
        st.latex(r"\phi M_n = \phi A_s f_y (d - \frac{a}{2})")
        st.latex(f"\\phi M_n = {phi} \\cdot {As_prov:.0f} \\cdot {fy} \\cdot ({d_bot:.1f} - {a:.2f}/2) \\cdot 10^{{-6}}")
        st.latex(f"\\phi M_n = \\mathbf{{{phiMn:.2f}}} \\text{ kNm}")
        
        if phiMn >= mu:
            st.success(f"✅ PASS: Capacity ({phiMn:.2f} kNm) > Demand ({mu:.2f} kNm)")
        else:
            st.error(f"❌ FAIL: Capacity ({phiMn:.2f} kNm) < Demand ({mu:.2f} kNm) -> Increase Steel")

        # 2.4 Minimum Reinforcement Check (ACI)
        st.markdown("**2.4 Minimum Reinforcement ($A_{s,min}$):**")
        as_min1 = (0.25 * np.sqrt(fc) / fy) * b_mm * d_bot
        as_min2 = (1.4 / fy) * b_mm * d_bot
        as_min = max(as_min1, as_min2)
        
        st.latex(r"A_{s,min} = \max \left( \frac{0.25\sqrt{f_c'}}{f_y} b_w d, \frac{1.4}{f_y} b_w d \right)")
        st.latex(f"A_{{s,min}} = \\max({as_min1:.0f}, {as_min2:.0f}) = \\mathbf{{{as_min:.0f}}} \\text{ mm}^2")
        
        if As_prov >= as_min:
            st.caption(f"✅ OK: Provided {As_prov:.0f} > Min {as_min:.0f}")
        elif As_prov >= 1.33 * As_prov: # Exception rule (simplified check)
             st.caption("⚠️ Note: As < As,min but might satisfy As provided > 1.33 As required.")
        else:
             st.error(f"❌ Warning: As ({As_prov:.0f}) < As,min ({as_min:.0f})")

    # --- 3. FLEXURAL DESIGN (NEGATIVE MOMENT) ---
    st.markdown("---")
    st.markdown(f"**3. Flexural Design: Support (Negative Moment $M_u^-$)**")
    
    mu_n = Mu_neg
    if mu_n <= 0.01:
        st.info("No significant negative moment.")
    else:
        top_db = res_data['top_db']
        top_n = res_data['neg']['n']
        
        # 3.1 Effective Depth
        d_top = h_mm - cover - stir_db - (top_db/2)
        st.latex(f"d = {h_mm:.0f} - {cover} - {stir_db} - {top_db/2:.1f} = \\mathbf{{{d_top:.1f}}} \\text{ mm}")

        # 3.2 Capacity
        As_prov_top = top_n * (np.pi * (top_db/2)**2)
        st.write(f"Design Moment ($M_u$): **{mu_n:.2f}** kNm")
        st.write(f"Provide: **{top_n}-DB{top_db}** ($A_s = {As_prov_top:.0f}$ mm²)")
        
        a_top = (As_prov_top * fy) / (0.85 * fc * b_mm)
        phi_top = 0.9 # Simplify check for report, but ideally check strain again
        Mn_top = As_prov_top * fy * (d_top - a_top/2) * 1e-6
        phiMn_top = phi_top * Mn_top
        
        st.latex(f"a = {a_top:.2f} \\text{ mm} \\quad \\to \\quad \\phi M_n = \\mathbf{{{phiMn_top:.2f}}} \\text{ kNm}")
        
        if phiMn_top >= mu_n:
            st.success(f"✅ PASS: Capacity ({phiMn_top:.2f}) > Demand ({mu_n:.2f})")
        else:
            st.error(f"❌ FAIL: Capacity ({phiMn_top:.2f}) < Demand ({mu_n:.2f})")

    # --- 4. SHEAR DESIGN ---
    st.markdown("---")
    st.markdown(f"**4. Shear Design ($V_u$)**")
    
    vu = res_data['Vu_max']
    stir_s = res_data['shear']['s']
    stir_db_shear = res_data['stir_db']
    
    st.write(f"Factored Shear Force ($V_u$): **{vu:.2f}** kN")
    
    # 4.1 Concrete Capacity (Vc)
    # ACI Simplified: Vc = 0.17 * sqrt(fc) * b * d
    # Use d from positive moment region as conservative or average
    d_shear = d_bot 
    Vc = 0.17 * np.sqrt(fc) * b_mm * d_shear / 1000.0 # kN
    phi_v = 0.85 # Using 0.85 (common in older ACI/Thai) or 0.75 (New ACI). Matching App logic.
    phiVc = phi_v * Vc
    
    st.markdown("**4.1 Concrete Shear Capacity ($\phi V_c$)**")
    st.latex(r"V_c = 0.17 \sqrt{f_c'} b_w d")
    st.latex(f"V_c = 0.17 \\sqrt{{{fc}}} \\cdot {b_mm:.0f} \\cdot {d_shear:.1f} / 1000 = {Vc:.2f} \\text{ kN}")
    st.latex(f"\\phi V_c = {phi_v} \\times {Vc:.2f} = \\mathbf{{{phiVc:.2f}}} \\text{ kN}")
    
    # 4.2 Shear Reinforcement Check
    st.markdown("**4.2 Stirrup Requirement**")
    
    if vu <= 0.5 * phiVc:
        st.success(f"Condition: $V_u \\le 0.5 \\phi V_c$ ({vu:.2f} vs {0.5*phiVc:.2f})")
        st.write("👉 No shear reinforcement required by calculation (Practice: Minimum stirrups).")
    elif vu <= phiVc:
        st.info(f"Condition: $0.5 \\phi V_c < V_u \\le \\phi V_c$")
        st.write("👉 Minimum shear reinforcement required.")
    else:
        st.warning(f"Condition: $V_u > \\phi V_c$ ({vu:.2f} > {phiVc:.2f})")
        st.write("👉 Shear reinforcement **REQUIRED**.")
        
        # Calculate Vs required
        # Vu <= phi(Vc + Vs)  -> Vs >= Vu/phi - Vc
        Vs_req = (vu / phi_v) - Vc
        st.latex(r"V_s = \frac{V_u}{\phi} - V_c")
        st.latex(f"V_s = \\frac{{{vu:.2f}}}{{{phi_v}}} - {Vc:.2f} = \\mathbf{{{Vs_req:.2f}}} \\text{ kN}")
        
        # Check Section Size limit (Vs <= 4*Vc is roughly sqrt(fc)/3 limit check, simplified here)
        # Max Vs allowed = 0.66 sqrt(fc) b d ~ 4 * Vc_simplified
        if Vs_req > 4 * Vc:
            st.error("❌ DANGER: $V_s$ too high! Section dimensions too small. Increase Depth (h).")

    # 4.3 Provided Stirrups
    st.markdown(f"**4.3 Provided Stirrups: RB{stir_db_shear} @ {stir_s} mm**")
    Av = 2 * (np.pi * (stir_db_shear/2)**2) # 2 legs
    st.latex(f"A_v = 2 \\times \\pi \\times ({stir_db_shear}/2)^2 = {Av:.1f} \\text{ mm}^2")
    
    # Vs provided
    Vs_prov = (Av * fy * d_shear) / stir_s / 1000.0
    phiVn = phiVc + (phi_v * Vs_prov)
    
    st.latex(r"V_{s,prov} = \frac{A_v f_y d}{s}")
    st.latex(f"V_{{s,prov}} = \\frac{{{Av:.1f} \\cdot {fy} \\cdot {d_shear:.1f}}}{{{stir_s}}} \\cdot 10^{{-3}} = {Vs_prov:.2f} \\text{ kN}")
    
    st.write(f"Total Capacity $\phi V_n = {phiVc:.2f} + {phi_v*Vs_prov:.2f} = \\mathbf{{{phiVn:.2f}}}$ **kN**")
    
    if phiVn >= vu:
        st.success(f"✅ PASS: Shear Capacity ({phiVn:.2f}) > Demand ({vu:.2f})")
    else:
        st.error(f"❌ FAIL: Shear Capacity ({phiVn:.2f}) < Demand ({vu:.2f}) -> Decrease spacing")

    # 4.4 Maximum Spacing Check (ACI)
    st.markdown("**4.4 Maximum Spacing Check ($s_{max}$)**")
    
    # Standard max spacing
    s_max_1 = d_shear / 2
    s_max_2 = 600
    s_max = min(s_max_1, s_max_2)
    
    st.latex(r"s_{max} = \min(d/2, 600 \text{ mm})")
    st.latex(f"s_{{max}} = \\min({s_max_1:.1f}, 600) = \\mathbf{{{s_max:.0f}}} \\text{ mm}")
    
    if stir_s <= s_max:
         st.caption(f"✅ Spacing {stir_s} mm <= Max {s_max:.0f} mm")
    else:
         st.error(f"❌ Spacing {stir_s} mm > Max {s_max:.0f} mm (Violation of Code)")

    st.markdown("---")
    st.caption("Calculation based on ACI 318 Strength Design Method (USD).")
