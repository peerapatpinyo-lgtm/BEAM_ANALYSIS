import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import input_handler
    import solver
    import rc_design
    import design_view
    import section_plotter
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

st.set_page_config(page_title="Pro Beam Design", layout="wide")

st.markdown("""
<style>
    .calc-box { background-color: #ffffff; border: 1px solid #e0e0e0; padding: 30px; border-radius: 8px; font-family: 'Sarabun', sans-serif; }
    .calc-header { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-bottom: 20px; font-weight: bold; font-size: 1.4em;}
    .sub-header { color: #555; font-weight: bold; margin-top: 15px; margin-bottom: 5px; font-size: 1.1em; text-decoration: underline;}
    .pass { color: #27ae60; font-weight: bold; background-color: #eafaf1; padding: 2px 8px; border-radius: 4px; }
    .fail { color: #c0392b; font-weight: bold; background-color: #fdedec; padding: 2px 8px; border-radius: 4px; }
    .formula { color: #333; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# --- Init Session State ---
if 'analyzed' not in st.session_state: st.session_state.analyzed = False
if 'avg_factor' not in st.session_state: st.session_state.avg_factor = 1.6 
if 'full_loads' not in st.session_state: st.session_state.full_loads = [] 
if 'sw_val' not in st.session_state: st.session_state.sw_val = 0.0

# --- 1. SIDEBAR INPUTS ---
with st.sidebar:
    st.title("⚙️ Project Inputs")
    design_std = st.radio("Standard Code", ["ACI 318 (USA)", "EIT 1008 (Thailand)"])
    
    # [FIX] Define distinct factors for ACI vs EIT
    if "EIT" in design_std:
        # EIT: Typically 1.4D + 1.7L
        # Phi: Flexure 0.90, Shear 0.85 (Difference here)
        factors = {
            'DL': 1.4, 'LL': 1.7, 
            'phi_m': 0.90, 'phi_v': 0.85, 
            'name': 'EIT Standard (วสท.)'
        }
        current_avg_factor = 1.7 # Conservative approx for initial sorting
    else:
        # ACI: 1.2D + 1.6L
        # Phi: Flexure 0.90, Shear 0.75
        factors = {
            'DL': 1.2, 'LL': 1.6, 
            'phi_m': 0.90, 'phi_v': 0.75, 
            'name': 'ACI 318'
        }
        current_avg_factor = 1.6
        
    st.info(f"Using: **{factors['name']}**\n\nLoad Factors: {factors['DL']}D + {factors['LL']}L\nPhi Shear ($\phi_v$): {factors['phi_v']}")
    
    params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()
    
    st.divider()
    run_btn = st.button("🚀 RUN ANALYSIS", type="primary")

# --- 2. MAIN AREA ---
st.header(f"🏗️ RC Beam Analysis & Design Report ({factors['name']})")

if run_btn:
    if not stable:
        st.error("🚨 Structure is unstable. Please check supports.")
    else:
        # --- A. PREPARE LOADS (Include SW) ---
        sw_mag = params['b'] * params['h'] * 24.0 * 1000 # N/m
        
        loads_combined = []
        if not loads_df.empty:
            loads_combined = loads_df.to_dict('records')
            
        for i in range(n_spans):
            loads_combined.append({
                "id": f"sw_{i}", "type": "U", "span_index": i, "x": 0.0,
                "mag": sw_mag, "dist": spans[i], "case": "DL (SW)"
            })

        # --- B. RUN ANALYSIS ---
        solver_service = solver.BeamSolver(spans, sup_df.to_dict('records'), loads_combined, params['E'], params['b'], params['h'], params['I'])
        res_service, reac_service, status = solver_service.solve()
        
        if "error" in status:
            st.error(status['error'])
            st.stop()

        # --- C. SAVE STATE ---
        st.session_state.res_service = res_service
        st.session_state.reac_service = reac_service
        st.session_state.loads_df = loads_df
        st.session_state.full_loads = loads_combined
        st.session_state.params = params
        st.session_state.factors = factors
        st.session_state.avg_factor = current_avg_factor
        st.session_state.sw_val = sw_mag / 1000.0
        st.session_state.analyzed = True

# --- DISPLAY RESULTS ---
if st.session_state.analyzed:
    res = st.session_state.res_service
    reac = st.session_state.reac_service
    p = st.session_state.params
    f = st.session_state.factors
    saf_factor = st.session_state.get('avg_factor', 1.6)
    full_loads = st.session_state.get('full_loads', [])
    sw_val = st.session_state.get('sw_val', 0.0)
    
    tab1, tab2, tab3 = st.tabs(["📊 1. Analysis Diagrams", "📝 2. Detailed Calculation", "💡 3. Improvements"])
    
    # ================= TAB 1: DIAGRAMS =================
    with tab1:
        st.markdown(f"### 🔹 Serviceability Diagrams")
        
        # Summary Metrics
        c1, c2, c3, c4 = st.columns(4)
        max_def = res['deflection'].abs().max()
        max_M = res['moment'].abs().max()/1000
        max_V = res['shear'].abs().max()/1000
        c1.metric("Max Deflection", f"{max_def:.2f} mm")
        c2.metric("Max Moment", f"{max_M:.2f} kNm")
        c3.metric("Max Shear", f"{max_V:.2f} kN")
        c4.metric("Self-Weight", f"{sw_val:.2f} kN/m")
        
        fig = design_view.plot_analysis_results(res, spans, sup_df, full_loads)
        st.plotly_chart(fig, use_container_width=True)

    # ================= TAB 2: DETAILED CALCULATION =================
    with tab2:
        # Prepare Design Data
        cum_dist = [0] + list(np.cumsum(spans))
        design_data = []
        design_res_for_plot = [] 
        
        for i in range(len(spans)):
            x0, x1 = cum_dist[i], cum_dist[i+1]
            span_df = res[(res['x'] >= x0) & (res['x'] <= x1)]
            
            M_pos_serv = max(0, span_df['moment'].max()) / 1000
            M_neg_serv = abs(min(0, span_df['moment'].min())) / 1000
            V_serv = span_df['shear'].abs().max() / 1000
            
            # Factored Loads
            design_data.append({
                "span": i+1,
                "M_serv_pos": M_pos_serv, 
                "M_serv_neg": M_neg_serv, 
                "V_serv": V_serv,
                "Mu_pos": M_pos_serv * saf_factor,
                "Mu_neg": M_neg_serv * saf_factor,
                "Vu": V_serv * saf_factor,
                "def_act": span_df['deflection'].abs().max(),
                "L": spans[i]
            })
            design_res_for_plot.append({'pos': {'n': 3}, 'neg': {'n': 2}, 'db': 16})

        sel_span = st.selectbox("Select Span to View Calculation", design_data, format_func=lambda x: f"Span {x['span']}")
        
        # Controls
        with st.expander("🛠️ Modify Reinforcement for Calculation", expanded=True):
            with st.form("rebar_form"):
                cc1, cc2, cc3, cc4, cc5 = st.columns(5)
                n_top = cc1.number_input("Top Bars", 1, 10, 2)
                n_bot = cc2.number_input("Bot Bars", 1, 10, 3)
                db_main = cc3.selectbox("Main DB (mm)", [12, 16, 20, 25], index=1)
                s_stir = cc4.number_input("Stirrup RB6 @ (cm)", 5, 30, 15, 5)
                cover = cc5.number_input("Cover (mm)", 20, 50, 40)
                st.form_submit_button("Update Calculation")

        # Layout: Calc Sheet
        st.markdown('<div class="calc-box">', unsafe_allow_html=True)
        st.markdown(f'<div class="calc-header">📝 ENGINEER CALCULATION SHEET: Span {sel_span["span"]} ({f["name"]})</div>', unsafe_allow_html=True)

        # --- PART 0: SYSTEM CHECK ---
        st.markdown('<div class="sub-header">0. System Equilibrium Check</div>', unsafe_allow_html=True)
        total_load_y = 0
        for l in full_loads:
            if l['type'] == 'P': total_load_y += l['mag']
            elif l['type'] == 'U': total_load_y += l['mag'] * l['dist']
        
        sum_reac = sum([abs(r['fy']) for r in reac])
        
        st.latex(rf"\sum F_{{load,y}} = {total_load_y/1000:.2f}\ kN")
        st.latex(rf"\sum R_y = {sum_reac/1000:.2f}\ kN")
        
        if abs(total_load_y - sum_reac) < 1.0: 
             st.markdown(f'<span class="pass">✅ EQUILIBRIUM OK (Diff < 1N)</span>', unsafe_allow_html=True)
        else:
             st.markdown(f'<span class="fail">❌ EQUILIBRIUM ERROR</span>', unsafe_allow_html=True)

        # --- PART 1: DEFLECTION ---
        st.markdown('<div class="sub-header">1. Deflection Check (Serviceability)</div>', unsafe_allow_html=True)
        L_mm = sel_span['L'] * 1000
        delta_allow = L_mm / 240.0
        delta_act = sel_span['def_act']
        
        st.latex(rf"\Delta_{{allow}} = \frac{{L}}{{240}} = \frac{{{L_mm:.0f}}}{{240}} = {delta_allow:.2f}\ mm")
        st.latex(rf"\Delta_{{actual}} = \mathbf{{{delta_act:.2f}}}\ mm")
        
        if delta_act <= delta_allow:
            st.markdown(f'<span class="pass">✅ PASS</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="fail">❌ FAIL (Stiffness insufficient)</span>', unsafe_allow_html=True)

        # --- PART 2: FLEXURE ---
        st.markdown('<div class="sub-header">2. Flexural Strength Design (USD)</div>', unsafe_allow_html=True)
        
        # Variables
        b_mm = p['b'] * 1000
        d_mm = p['h'] * 1000 - cover - 6 - db_main/2
        As_prov = n_bot * (3.1416 * (db_main/2)**2)
        
        st.markdown(f"**Step 2.1: Factored Moment ($M_u$)** - *Using factors from {f['name']}*")
        st.latex(rf"M_u \approx M_{{serv}} \times \text{{AvgFactor}} = {sel_span['M_serv_pos']:.2f} \times {saf_factor} = \mathbf{{{sel_span['Mu_pos']:.2f}}}\ kNm")

        st.markdown("**Step 2.2: Steel Area & Effective Depth**")
        st.latex(rf"d = h - cover - d_{{stir}} - d_b/2 = {p['h']*1000:.0f} - {cover} - 6 - {db_main/2} = {d_mm:.1f}\ mm")
        st.latex(rf"A_{{s,prov}} = {n_bot} \times \frac{{\pi \cdot {db_main}^2}}{{4}} = \mathbf{{{As_prov:.0f}}}\ mm^2")

        st.markdown("**Step 2.3: Minimum Steel Check ($A_{s,min}$)**")
        st.caption("Note: ACI and EIT share similar min steel requirements for beams.")
        As_min1 = (0.25 * np.sqrt(p['fc']) / p['fy']) * b_mm * d_mm
        As_min2 = (1.4 / p['fy']) * b_mm * d_mm
        As_min = max(As_min1, As_min2)
        
        st.latex(rf"A_{{s,min}} = \max\left(\frac{{0.25\sqrt{{f_c'}}}}{{f_y}} b_w d, \frac{{1.4}}{{f_y}} b_w d\right) = {As_min:.0f}\ mm^2")
        if As_prov >= As_min: st.markdown(f'<span class="pass">✅ OK</span>', unsafe_allow_html=True)
        else: st.markdown(f'<span class="fail">❌ FAIL ($A_s < A_{{s,min}}$)</span>', unsafe_allow_html=True)

        st.markdown("**Step 2.4: Moment Capacity Calculation ($\phi M_n$)**")
        a_depth = (As_prov * p['fy']) / (0.85 * p['fc'] * b_mm)
        st.latex(rf"a = \frac{{A_s f_y}}{{0.85 f_c' b}} = \frac{{{As_prov:.0f} \cdot {p['fy']}}}{{0.85 \cdot {p['fc']} \cdot {b_mm:.0f}}} = {a_depth:.2f}\ mm")
        
        Mn_kNm = As_prov * p['fy'] * (d_mm - a_depth/2) * 1e-6
        phi_Mn = f['phi_m'] * Mn_kNm
        
        st.latex(rf"M_n = A_s f_y (d - a/2) = {As_prov:.0f} \cdot {p['fy']} ({d_mm:.1f} - {a_depth/2:.1f}) \cdot 10^{{-6}} = {Mn_kNm:.2f}\ kNm")
        st.latex(rf"\phi M_n = {f['phi_m']} \cdot {Mn_kNm:.2f} = \mathbf{{{phi_Mn:.2f}}}\ kNm")
        
        if phi_Mn >= sel_span['Mu_pos']:
            st.markdown(f'<span class="pass">✅ SAFE (Ratio: {sel_span["Mu_pos"]/phi_Mn:.2f})</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="fail">❌ UNSAFE (Capacity Insufficient)</span>', unsafe_allow_html=True)

        # --- PART 3: SHEAR ---
        st.markdown('<div class="sub-header">3. Shear Strength Design</div>', unsafe_allow_html=True)
        
        st.markdown(f"**Step 3.1: Factored Shear ($V_u$) & Capacity Factors**")
        st.markdown(f"*Using $\phi_v = {f['phi_v']}$ according to {f['name']}*")
        st.latex(rf"V_u = \mathbf{{{sel_span['Vu']:.2f}}}\ kN")

        st.markdown("**Step 3.2: Concrete Shear Capacity ($V_c$)**")
        Vc_val = 0.17 * np.sqrt(p['fc']) * b_mm * d_mm / 1000.0
        phi_Vc = f['phi_v'] * Vc_val
        st.latex(rf"V_c = 0.17\sqrt{{f_c'}} b_w d = 0.17\sqrt{{{p['fc']}}} \cdot {b_mm:.0f} \cdot {d_mm:.0f} = {Vc_val:.2f}\ kN")
        st.latex(rf"\phi V_c = {f['phi_v']} \times {Vc_val:.2f} = {phi_Vc:.2f}\ kN")

        st.markdown("**Step 3.3: Steel Stirrup Capacity ($V_s$)**")
        Av = 2 * (3.1416 * 3**2) # RB6 2 legs
        s_mm = s_stir * 10
        Vs_val = (Av * p['fy'] * d_mm) / s_mm / 1000.0
        phi_Vs = f['phi_v'] * Vs_val
        
        st.latex(rf"V_s = \frac{{A_v f_y d}}{{s}} = \frac{{{Av:.1f} \cdot {p['fy']} \cdot {d_mm:.0f}}}{{{s_mm}}} = {Vs_val:.2f}\ kN")
        st.latex(rf"\phi V_s = {f['phi_v']} \times {Vs_val:.2f} = {phi_Vs:.2f}\ kN")

        st.markdown("**Step 3.4: Total Shear Capacity check**")
        phi_Vn = phi_Vc + phi_Vs
        st.latex(rf"\phi V_n = \phi V_c + \phi V_s = {phi_Vc:.2f} + {phi_Vs:.2f} = \mathbf{{{phi_Vn:.2f}}}\ kN")
        
        if phi_Vn >= sel_span['Vu']:
            st.markdown(f'<span class="pass">✅ SAFE</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="fail">❌ UNSAFE (Reduce stirrup spacing)</span>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        
        # --- PLOTS AT BOTTOM ---
        st.markdown("---")
        c_plot1, c_plot2 = st.columns([1, 2])
        with c_plot1:
            st.markdown("**Section View**")
            fig_sec = section_plotter.plot_section(p['b'], p['h'], cover, db_main, n_top, n_bot, f"RB6@{s_stir}", p['fc'], p['fy'])
            st.pyplot(fig_sec, use_container_width=False)
        with c_plot2:
            st.markdown("**Longitudinal Profile**")
            design_res_for_plot[sel_span['span']-1] = {'pos': {'n': n_bot}, 'neg': {'n': n_top}, 'db': db_main}
            fig_long = section_plotter.plot_longitudinal_section_detailed(spans, sup_df, design_res_for_plot, p['h'], cover)
            st.pyplot(fig_long, use_container_width=True)

    # ================= TAB 3: RECOMMENDATIONS =================
    with tab3:
        st.subheader("💡 5 Professional Design Recommendations")
        
        # Logic for recommendations
        recs = []
        
        # 1. Deflection logic
        if sel_span['def_act'] > sel_span['L']*1000/240:
            recs.append(f"**Critical:** Deflection ({sel_span['def_act']:.2f} mm) exceeds limit. **Increase Beam Depth (h)** immediately. Increasing steel has minimal effect on stiffness compared to depth.")
        else:
            recs.append(f"**Optimization:** Deflection is within limits. You might optimize by slightly reducing depth if shear allows.")

        # 2. Section size logic
        if p['b'] < p['h']*0.4:
            recs.append(f"**Geometry:** The beam is quite slender (b/h ratio). Ensure lateral stability or increase 'b' to at least {p['h']*0.5:.2f}m for better concrete placement.")
        
        # 3. Steel Percentage
        rho = As_prov / (b_mm * d_mm)
        if rho > 0.025: # High steel
            recs.append(f"**Congestion:** Reinforcement ratio is high (> 2.5%). Consider increasing section size to avoid honeycombing and ensure proper vibration during pouring.")
        elif rho < 0.0033: # Very low
            recs.append(f"**Minimums:** Reinforcement is low. Ensure you meet $A_{{s,min}}$ to prevent sudden brittle failure.")

        # 4. Shear logic
        if sel_span['Vu'] > phi_Vc:
            recs.append("**Shear:** The concrete capacity alone is insufficient. Stirrups are structurally critical. Ensure strict supervision on stirrup spacing during construction.")
        else:
            recs.append("**Shear:** Concrete takes most shear force. Stirrups are likely governed by maximum spacing rules ($d/2$).")

        # 5. General detailing
        recs.append(f"**Detailing:** For **{f['name']}** compliance, ensure top bars extend past the inflection point by at least $d$ or $12d_b$.")

        # Display
        for i, r in enumerate(recs):
            st.markdown(f"{i+1}. {r}")
