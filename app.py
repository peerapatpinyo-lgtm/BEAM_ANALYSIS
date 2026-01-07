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
    .calc-box { background-color: white; border: 1px solid #ddd; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .calc-header { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px; margin-bottom: 15px; font-weight: bold; }
    .pass { color: green; font-weight: bold; }
    .fail { color: red; font-weight: bold; }
    .sub-calc { color: #555; font-family: 'Courier New', monospace; font-size: 0.9em; margin-bottom: 5px; }
</style>
""", unsafe_allow_html=True)

# --- Init Session State ---
if 'analyzed' not in st.session_state: st.session_state.analyzed = False
if 'avg_factor' not in st.session_state: st.session_state.avg_factor = 1.6 
if 'full_loads' not in st.session_state: st.session_state.full_loads = [] # Store loads with SW

# --- 1. SIDEBAR INPUTS ---
with st.sidebar:
    st.title("⚙️ Project Inputs")
    design_std = st.radio("Standard", ["ACI 318", "EIT (Thailand)"])
    
    if "EIT" in design_std:
        factors = {'DL': 1.4, 'LL': 1.7, 'phi_m': 0.90, 'phi_v': 0.85, 'label': '1.70'}
        current_avg_factor = 1.7
    else:
        factors = {'DL': 1.2, 'LL': 1.6, 'phi_m': 0.90, 'phi_v': 0.75, 'label': '1.60'}
        current_avg_factor = 1.6
        
    st.info(f"Design Factor (Approx) ≈ {factors['label']}")
    
    params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()
    
    st.divider()
    run_btn = st.button("🚀 RUN ANALYSIS", type="primary")

# --- 2. MAIN AREA ---
st.header("🏗️ RC Beam Analysis & Design Report")

if run_btn:
    if not stable:
        st.error("🚨 Structure is unstable. Please check supports.")
    else:
        # --- A. PREPARE LOADS (Include SW) ---
        sw_mag = params['b'] * params['h'] * 24.0 * 1000 # N/m
        
        # Create a combined list for Solver AND Plotting
        loads_combined = []
        if not loads_df.empty:
            loads_combined = loads_df.to_dict('records')
            
        # Add Self Weight explicitly
        for i in range(n_spans):
            loads_combined.append({
                "id": f"sw_{i}", "type": "U", "span_index": i, "x": 0.0,
                "mag": sw_mag, "dist": spans[i], "case": "DL (SW)"
            })

        # --- B. RUN ANALYSIS ---
        # Pass loads_combined (User + SW) to solver
        solver_service = solver.BeamSolver(spans, sup_df.to_dict('records'), loads_combined, params['E'], params['b'], params['h'], params['I'])
        res_service, reac_service, status = solver_service.solve()
        
        if "error" in status:
            st.error(status['error'])
            st.stop()

        # --- C. SAVE STATE ---
        st.session_state.res_service = res_service
        st.session_state.reac_service = reac_service
        st.session_state.loads_df = loads_df
        st.session_state.full_loads = loads_combined # [FIX] Save user+sw loads for plotting
        st.session_state.params = params
        st.session_state.factors = factors
        st.session_state.avg_factor = current_avg_factor
        st.session_state.sw_val = sw_mag / 1000.0 # kN/m
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
    
    tab1, tab2 = st.tabs(["📊 1. Analysis (Service Load)", "📝 2. Design (Ultimate Load)"])
    
    # ================= TAB 1: SERVICE ANALYSIS =================
    with tab1:
        st.markdown(f"### 🔹 Serviceability State (DL + LL)")
        st.caption(f"**Note:** Analysis includes Self-Weight of Beam = **{sw_val:.2f} kN/m**")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Max Deflection", f"{res['deflection'].abs().max():.2f} mm")
        col2.metric("Max M (+)", f"{res['moment'].max()/1000:.2f} kNm")
        col3.metric("Max M (-)", f"{res['moment'].min()/1000:.2f} kNm")
        col4.metric("Max V", f"{res['shear'].abs().max()/1000:.2f} kN")
        
        # [FIX] Plot using full_loads (Includes SW now)
        fig = design_view.plot_analysis_results(res, spans, sup_df, full_loads)
        st.plotly_chart(fig, use_container_width=True)

    # ================= TAB 2: ULTIMATE DESIGN =================
    with tab2:
        st.markdown("### 🔹 Ultimate Limit State (Factored)")
        
        cum_dist = [0] + list(np.cumsum(spans))
        design_data = []
        for i in range(len(spans)):
            x0, x1 = cum_dist[i], cum_dist[i+1]
            span_df = res[(res['x'] >= x0) & (res['x'] <= x1)]
            
            M_pos_serv = max(0, span_df['moment'].max()) / 1000
            M_neg_serv = abs(min(0, span_df['moment'].min())) / 1000
            V_serv = span_df['shear'].abs().max() / 1000
            
            design_data.append({
                "span": i+1,
                "M_serv_pos": M_pos_serv, "M_serv_neg": M_neg_serv, "V_serv": V_serv,
                "Mu_pos": M_pos_serv * saf_factor,
                "Mu_neg": M_neg_serv * saf_factor,
                "Vu": V_serv * saf_factor,
                "def_act": span_df['deflection'].abs().max(),
                "L": spans[i]
            })

        sel_span = st.selectbox("Select Span", design_data, format_func=lambda x: f"Span {x['span']}")
        
        c_left, c_right = st.columns([1, 1.5]) # ให้พื้นที่คำนวณกว้างขึ้น
        
        with c_left:
            st.markdown(f"#### 🛠️ Reinforcement")
            with st.form("rebar_form"):
                cc1, cc2 = st.columns(2)
                n_top = cc1.number_input("Top Bars", 1, 10, 2)
                n_bot = cc2.number_input("Bot Bars", 1, 10, 3)
                cc3, cc4 = st.columns(2)
                db_main = cc3.selectbox("DB (mm)", [12, 16, 20, 25], index=1)
                s_stir = cc4.number_input("Stirrup @ (cm)", 5, 30, 15, 5)
                cover = st.number_input("Cover (mm)", 20, 50, 40)
                st.form_submit_button("Update")
            
            fig_sec = section_plotter.plot_section(p['b'], p['h'], cover, db_main, n_top, n_bot, f"RB6@{s_stir}", p['fc'], p['fy'])
            st.pyplot(fig_sec)

        with c_right:
            st.markdown('<div class="calc-box">', unsafe_allow_html=True)
            st.markdown(f'<div class="calc-header">📝 DETAILED CALCULATION: Span {sel_span["span"]}</div>', unsafe_allow_html=True)
            
            # --- 1. Load Combination ---
            st.markdown("**1. Design Forces (Factored)**")
            st.latex(rf"M_u^+ = M_{{serv}} \times {saf_factor} = {sel_span['M_serv_pos']:.2f} \times {saf_factor} = \mathbf{{{sel_span['Mu_pos']:.2f}}}\ kNm")
            st.latex(rf"V_u = V_{{serv}} \times {saf_factor} = {sel_span['V_serv']:.2f} \times {saf_factor} = \mathbf{{{sel_span['Vu']:.2f}}}\ kN")

            # --- 2. Flexure Detailed ---
            st.markdown("---")
            st.markdown("**2. Flexural Strength Check (+M)**")
            
            # Variables
            b_mm = p['b'] * 1000
            d_mm = p['h'] * 1000 - cover - 6 - db_main/2
            As_prov = n_bot * (3.1416 * (db_main/2)**2)
            
            st.markdown("*(a) Determine Effective Depth (d) & Steel Area (As)*")
            st.latex(rf"d = h - c_c - d_v - d_b/2 = {p['h']*1000:.0f} - {cover} - 6 - {db_main/2} = {d_mm:.1f}\ mm")
            st.latex(rf"A_{{s,prov}} = {n_bot} \times \pi ({db_main}/2)^2 = \mathbf{{{As_prov:.0f}}}\ mm^2")

            st.markdown("*(b) Calculate Whitney Stress Block (a)*")
            a_depth = (As_prov * p['fy']) / (0.85 * p['fc'] * b_mm)
            st.latex(rf"a = \frac{{A_s f_y}}{{0.85 f_c' b}} = \frac{{{As_prov:.0f} \cdot {p['fy']}}}{{0.85 \cdot {p['fc']} \cdot {b_mm:.0f}}} = \mathbf{{{a_depth:.2f}}}\ mm")

            st.markdown("*(c) Calculate Moment Capacity*")
            Mn_kNm = As_prov * p['fy'] * (d_mm - a_depth/2) * 1e-6
            phi_Mn = f['phi_m'] * Mn_kNm
            
            st.latex(rf"M_n = A_s f_y (d - a/2) = {As_prov:.0f} \cdot {p['fy']} ({d_mm:.1f} - {a_depth/2:.1f}) \cdot 10^{{-6}} = {Mn_kNm:.2f}\ kNm")
            st.latex(rf"\phi M_n = {f['phi_m']} \cdot {Mn_kNm:.2f} = \mathbf{{{phi_Mn:.2f}}}\ kNm")
            
            if phi_Mn >= sel_span['Mu_pos']:
                st.markdown(f'<span class="pass">✅ OK (Ratio: {sel_span["Mu_pos"]/phi_Mn:.2f})</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="fail">❌ FAIL (Increase steel or depth)</span>', unsafe_allow_html=True)

            # --- 3. Shear Detailed ---
            st.markdown("---")
            st.markdown("**3. Shear Strength Check**")
            
            st.markdown("*(a) Concrete Capacity (Vc)*")
            Vc_val = 0.17 * np.sqrt(p['fc']) * b_mm * d_mm / 1000.0
            st.latex(rf"V_c = 0.17\sqrt{{f_c'}} b_w d = 0.17\sqrt{{{p['fc']}}} \cdot {b_mm:.0f} \cdot {d_mm:.0f} = \mathbf{{{Vc_val:.2f}}}\ kN")

            st.markdown("*(b) Steel Capacity (Vs) - RB6*")
            Av = 2 * (3.1416 * 3**2) # 2 legs of 6mm
            s_mm = s_stir * 10
            Vs_val = (Av * p['fy'] * d_mm) / s_mm / 1000.0
            
            st.latex(rf"A_v = 2 \cdot A_{{bar}} = 2 \cdot 28.3 = {Av:.1f}\ mm^2")
            st.latex(rf"V_s = \frac{{A_v f_y d}}{{s}} = \frac{{{Av:.1f} \cdot {p['fy']} \cdot {d_mm:.0f}}}{{{s_mm}}} = \mathbf{{{Vs_val:.2f}}}\ kN")
            
            st.markdown("*(c) Total Capacity*")
            phi_Vn = f['phi_v'] * (Vc_val + Vs_val)
            st.latex(rf"\phi V_n = \phi (V_c + V_s) = {f['phi_v']} ({Vc_val:.2f} + {Vs_val:.2f}) = \mathbf{{{phi_Vn:.2f}}}\ kN")
             
            if phi_Vn >= sel_span['Vu']:
                 st.markdown(f'<span class="pass">✅ OK</span>', unsafe_allow_html=True)
            else:
                 st.markdown(f'<span class="fail">❌ FAIL (Reduce stirrup spacing)</span>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
