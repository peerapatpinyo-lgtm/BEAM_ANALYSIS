#reporter.py
import streamlit as st
import numpy as np

def render_calculation_report(res):
    """
    Ultra-Detailed ACI 318-19 Compliance Report.
    Includes Clause References, Substitutions, and Limit States.
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

    # --- Constants & ACI Parameters ---
    Es = 200000.0 
    Ec = 4700 * np.sqrt(fc)
    
    # ACI 22.2.2.4.3: Beta1 calculation
    if fc <= 28: beta1 = 0.85
    elif fc >= 55: beta1 = 0.65
    else: beta1 = 0.85 - (0.05 * (fc - 28) / 7)

    st.markdown(f"## 🏛️ Comprehensive ACI 318-19 Design Audit: Span {idx}")
    st.markdown(f"**Structural Element:** Continuous RC Beam | **Span Length:** {L_m:.2f} m")
    st.divider()

    # =========================================================
    # 1. MATERIAL & SECTION PROPERTIES (ACI 19.2 & 20.2)
    # =========================================================
    st.markdown("### 1. Materials & Geometry (Ref: ACI 19.2 & 20.2)")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Concrete Strength Properties:**")
        st.latex(rf"f'_c = {fc} \text{{ MPa}}")
        st.latex(rf"E_c = 4700\sqrt{{f'_c}} = {Ec:.0f} \text{{ MPa}}")
        st.latex(rf"\beta_1 = {beta1:.3f} \quad \text{{(ACI 22.2.2.4.3)}}")
    with c2:
        st.write("**Steel Reinforcement:**")
        st.latex(rf"f_y = {fy} \text{{ MPa}}, \quad E_s = 200,000 \text{{ MPa}}")
        st.latex(rf"\text{{Section: }} {b:.0f} \times {h:.0f} \text{{ mm}}")

    # =========================================================
    # 2. FLEXURAL CAPACITY AUDIT (ACI 22.2)
    # =========================================================
    st.markdown("### 2. Flexural Strength Audit (Ref: ACI 22.2)")
    
    # 2.1 Effective Depth (d)
    d = h - cov - stir_db - (bot_db/2)
    st.markdown("**2.1 Effective Depth Calculation**")
    st.latex(rf"d = h - c_{{clear}} - \text{{db}}_{{stirrup}} - \frac{{\text{{db}}_{{bar}}}}{{2}}")
    st.latex(rf"d = {h} - {cov} - {stir_db} - \frac{{{bot_db}}}{{2}} = \mathbf{{{d:.1f}}}\text{{ mm}}")

    # 2.2 Minimum Reinforcement (ACI 9.6.1.2)
    st.markdown("**2.2 Minimum Steel Check (Ref: ACI 9.6.1.2)**")
    As = bot_n * (np.pi * (bot_db/2)**2)
    As_min_1 = (0.25 * np.sqrt(fc) / fy) * b * d
    As_min_2 = (1.4 / fy) * b * d
    As_min = max(As_min_1, As_min_2)
    
    st.latex(rf"A_{{s,min}} = \max \left( \frac{{0.25\sqrt{{f'_c}}}}{{f_y}} b_w d, \frac{{1.4}}{{f_y}} b_w d \right) = \mathbf{{{As_min:.1f}}}\text{{ mm}}^2")
    if As >= As_min:
        st.caption(f"✅ Provided As ({As:.1f} mm²) > Min As ({As_min:.1f} mm²)")
    else:
        st.error(f"❌ Provided As ({As:.1f} mm²) < Min As ({As_min:.1f} mm²)")

    # 2.3 Tension-Controlled & Ductility (ACI 21.2.2)
    st.markdown("**2.3 Strain Compatibility & Strength Reduction ($\phi$)**")
    a = (As * fy) / (0.85 * fc * b)
    c_neutral = a / beta1
    epsilon_t = 0.003 * (d - c_neutral) / c_neutral # Net tensile strain

    # Determine Phi (Table 21.2.2)
    if epsilon_t >= 0.005:
        phi_f = 0.90
        state = "Tension-Controlled (Ductile)"
    elif epsilon_t <= 0.002:
        phi_f = 0.65
        state = "Compression-Controlled (Brittle - NOT RECOMMENDED)"
    else:
        phi_f = 0.65 + 0.25 * (epsilon_t - 0.002) / 0.003
        state = "Transition Zone"

    st.latex(rf"a = \frac{{A_s f_y}}{{0.85 f'_c b}} = \frac{{{As:.1f} \cdot {fy}}}{{0.85 \cdot {fc} \cdot {b}}} = {a:.2f}\text{{ mm}}")
    st.latex(rf"c = a/\beta_1 = {a:.2f}/{beta1:.3f} = {c_neutral:.2f}\text{{ mm}}")
    st.latex(rf"\epsilon_t = 0.003 \left( \frac{{d - c}}{{c}} \right) = \mathbf{{{epsilon_t:.5f}}}")
    st.info(f"**Result:** {state} | $\phi = {phi_f:.3f}$")

    # 2.4 Nominal vs Factored Moment
    Mn = As * fy * (d - a/2) * 1e-6
    phiMn = phi_f * Mn
    st.markdown("**2.4 Ultimate Strength Verification**")
    st.latex(rf"M_n = A_s f_y (d - a/2) = {As:.0f} \cdot {fy} \cdot ({d:.1f} - {a/2:.1f}) \cdot 10^{{-6}} = {Mn:.2f}\text{{ kNm}}")
    st.latex(rf"\phi M_n = {phi_f:.2f} \cdot {Mn:.2f} = \mathbf{{{phiMn:.2f}}}\text{{ kNm}}")
    
    if phiMn >= Mu:
        st.success(rf"$\phi M_n ({phiMn:.2f}) \ge M_u ({Mu:.2f})$ — Design Capacity is Sufficient.")
    else:
        st.error(rf"$\phi M_n ({phiMn:.2f}) < M_u ({Mu:.2f})$ — REINFORCEMENT INSUFFICIENT.")

    # =========================================================
    # 3. SHEAR CAPACITY AUDIT (ACI 22.5)
    # =========================================================
    st.divider()
    st.markdown("### 3. Shear Strength Audit (Ref: ACI 22.5)")
    st.latex(rf"V_u = {Vu:.2f}\text{{ kN}}")
    
    # Concrete Shear (ACI 22.5.5.1)
    Vc = (0.17 * 1.0 * np.sqrt(fc) * b * d) / 1000
    # Stirrup Shear (ACI 22.5.10.1)
    Av = 2 * (np.pi * (stir_db/2)**2) # Assuming 2 legs
    Vs = (Av * fy * d / stir_s) / 1000
    phiVn = 0.75 * (Vc + Vs)

    st.latex(rf"V_c = 0.17 \lambda \sqrt{{f'_c}} b_w d = 0.17 \cdot 1.0 \cdot \sqrt{{{fc}}} \cdot {b} \cdot {d:.1f} = {Vc:.2f}\text{{ kN}}")
    st.latex(rf"V_s = \frac{{A_v f_{{yt}} d}}{{s}} = \frac{{{Av:.1f} \cdot {fy} \cdot {d:.1f}}}{{{stir_s}}} = {Vs:.2f}\text{{ kN}}")
    st.latex(rf"\phi V_n = 0.75(V_c + V_s) = 0.75({Vc:.2f} + {Vs:.2f}) = \mathbf{{{phiVn:.2f}}}\text{{ kN}}")
    
    # Shear Spacing Requirements (ACI 9.7.6.2.2)
    s_max = min(d/2, 600)
    st.markdown(rf"**ACI Spacing Limit:** $s_{{max}} = \min(d/2, 600) = \mathbf{{{s_max:.0f}}}\text{{ mm}}$")
    if stir_s <= s_max:
        st.caption(f"✅ Provided Spacing ({stir_s} mm) < Allowable ({s_max:.0f} mm)")
    else:
        st.error(f"❌ Spacing ({stir_s} mm) exceeds ACI Limit ({s_max:.0f} mm)")

    # =========================================================
    # 4. SERVICEABILITY AUDIT (ACI 24.2)
    # =========================================================
    st.divider()
    st.markdown("### 4. Serviceability Audit (Ref: ACI 24.2)")
    L_mm = L_m * 1000
    allowable_def = L_mm / 240 # Standard for floors not supporting non-structural elements
    
    st.write(f"**Limit:** Instantaneous Live Load Deflection limit (L/240):")
    st.latex(rf"\Delta_{{allow}} = \frac{{L}}{{240}} = \frac{{{L_mm:.0f}}}{{240}} = \mathbf{{{allowable_def:.2f}}}\text{{ mm}}")
    st.latex(rf"\Delta_{{actual}} = \mathbf{{{abs(delta_svc):.3f}}}\text{{ mm}}")

    if abs(delta_svc) <= allowable_def:
        st.success("Serviceability Check: PASS")
    else:
        st.warning("Serviceability Check: FAIL (Section stiffness should be increased)")
