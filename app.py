import streamlit as st
import pandas as pd
import numpy as np

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

# --- CSS: Engineering Report Style ---
st.markdown("""
<style>
    .report-frame { border: 1px solid #ddd; padding: 20px; background-color: #fff; border-radius: 5px; }
    .calc-header { font-size: 18px; font-weight: bold; color: #333; border-bottom: 2px solid #0056b3; margin-bottom: 15px; padding-bottom: 5px; }
    .sub-head { font-weight: bold; color: #555; margin-top: 10px; text-decoration: underline; }
    .status-pass { color: green; font-weight: bold; background-color: #e6fffa; padding: 2px 5px; border-radius: 3px; }
    .status-fail { color: red; font-weight: bold; background-color: #fff5f5; padding: 2px 5px; border-radius: 3px; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0 0; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-top: 2px solid #0056b3; }
</style>
""", unsafe_allow_html=True)

# --- Init Session State ---
if 'analyzed' not in st.session_state: st.session_state.analyzed = False
if 'avg_factor' not in st.session_state: st.session_state.avg_factor = 1.6 
if 'full_loads' not in st.session_state: st.session_state.full_loads = [] 
if 'sw_val' not in st.session_state: st.session_state.sw_val = 0.0

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Project Inputs")
    design_std = st.radio("Standard", ["ACI 318", "EIT 1008"])
    
    if "EIT" in design_std:
        factors = {'DL': 1.4, 'LL': 1.7, 'phi_m': 0.90, 'phi_v': 0.85, 'name': 'EIT 1008'}
        current_avg_factor = 1.7 
    else:
        factors = {'DL': 1.2, 'LL': 1.6, 'phi_m': 0.90, 'phi_v': 0.75, 'name': 'ACI 318'}
        current_avg_factor = 1.6
        
    params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()
    st.divider()
    run_btn = st.button("🚀 RUN ANALYSIS", type="primary", use_container_width=True)

# --- MAIN ---
st.title("🏗️ RC Beam Analysis & Design")

if run_btn:
    if not stable:
        st.error("🚨 Structure Unstable")
    else:
        # Load Prep
        sw_mag = params['b'] * params['h'] * 24.0 * 1000 
        loads_combined = []
        if not loads_df.empty: loads_combined = loads_df.to_dict('records')
        for i in range(n_spans):
            loads_combined.append({"id": f"sw_{i}", "type": "U", "span_index": i, "x": 0.0, "mag": sw_mag, "dist": spans[i]})

        # Solve
        solver_service = solver.BeamSolver(spans, sup_df.to_dict('records'), loads_combined, params['E'], params['b'], params['h'], params['I'])
        res_service, reac_raw, status = solver_service.solve()
        
        if "error" in status: st.error(status['error']); st.stop()

        # Normalize Reac
        final_reac_list = []
        if isinstance(reac_raw, pd.DataFrame):
            if 'fy' in reac_raw.columns: final_reac_list = reac_raw.to_dict('records')
            else: final_reac_list = [{'node_id': i, 'fy': r.iloc[0]} for i, r in reac_raw.iterrows()]
        elif isinstance(reac_raw, dict):
            final_reac_list = [{'node_id': int(k), 'fy': v} for k, v in reac_raw.items()]
        elif isinstance(reac_raw, list):
             if len(reac_raw) > 0 and isinstance(reac_raw[0], (float, int)):
                 final_reac_list = [{'node_id': i, 'fy': v} for i, v in enumerate(reac_raw)]
             else:
                 final_reac_list = reac_raw

        clean_reac = [{'node_id': r.get('node_id', r.get('id', i)), 'fy': r.get('fy', 0)} for i, r in enumerate(final_reac_list)]

        # Save State
        st.session_state.res_service = res_service
        st.session_state.reac_service = clean_reac
        st.session_state.loads_df = loads_df
        st.session_state.full_loads = loads_combined
        st.session_state.params = params
        st.session_state.factors = factors
        st.session_state.avg_factor = current_avg_factor
        st.session_state.sw_val = sw_mag / 1000.0
        st.session_state.analyzed = True

if st.session_state.analyzed:
    res = st.session_state.res_service
    reac = st.session_state.reac_service
    p = st.session_state.params
    f = st.session_state.factors
    saf_factor = st.session_state.get('avg_factor', 1.6)
    full_loads = st.session_state.get('full_loads', [])
    
    # === TAB LAYOUT ===
    tab1, tab2 = st.tabs(["📊 1. Structural Analysis", "📝 2. Design Calculation"])
    
    # --- TAB 1: GRAPHS ---
    with tab1:
        st.subheader("Analysis Diagrams")
        fig = design_view.plot_analysis_results(res, spans, sup_df, full_loads, reac)
        st.plotly_chart(fig, use_container_width=True)
    
    # --- TAB 2: DESIGN & CALCULATION ---
    with tab2:
        # Prepare Data
        cum_dist = [0] + list(np.cumsum(spans))
        design_data = []
        for i in range(len(spans)):
            x0, x1 = cum_dist[i], cum_dist[i+1]
            span_df = res[(res['x'] >= x0) & (res['x'] <= x1)]
            M_max = max(span_df['moment'].abs().max(), 0) / 1000
            V_max = span_df['shear'].abs().max() / 1000
            design_data.append({
                "span": i+1, "Mu": M_max * saf_factor, "Vu": V_max * saf_factor,
                "def_act": span_df['deflection'].abs().max(), "L": spans[i]
            })
            
        # 2.1 INPUT BAR
        with st.container(border=True):
            st.markdown("#### 🛠️ Reinforcement Configuration")
            col_sel, col_in = st.columns([1, 4])
            with col_sel:
                sel_span = st.selectbox("Select Span", design_data, format_func=lambda x: f"Span {x['span']} (Mu={x['Mu']:.1f})")
            with col_in:
                with st.form("rebar_form"):
                    c1, c2, c3, c4, c5, c6 = st.columns([1,1,1,1.5,1,1])
                    n_top = c1.number_input("Top", 2, 10, 2)
                    n_bot = c2.number_input("Bot", 2, 10, 3)
                    db_main = c3.selectbox("DB", [12, 16, 20, 25, 28], index=1)
                    s_stir = c4.number_input("Stirrup@(cm)", 5, 50, 15, 5)
                    cover = c5.number_input("Cov(mm)", 20, 50, 40)
                    c6.write("")
                    c6.form_submit_button("Update")
        
        st.write("") # Spacer

        # 2.2 CALCULATION SHEET + CROSS SECTION (SIDE BY SIDE)
        col_calc, col_img = st.columns([1.5, 1])
        
        # --- Left: Detailed Calculation (Restored Good Data) ---
        with col_calc:
            st.markdown('<div class="report-frame">', unsafe_allow_html=True)
            st.markdown(f'<div class="calc-header">📝 CALCULATION SHEET: SPAN {sel_span["span"]}</div>', unsafe_allow_html=True)
            
            # Variables
            b_mm, h_mm = p['b']*1000, p['h']*1000
            d_mm = h_mm - cover - 9 - db_main/2 # 9mm stirrup est
            As_prov = n_bot * 3.1416 * (db_main/2)**2
            
            # FLEXURE
            st.markdown('<div class="sub-head">1. FLEXURAL DESIGN</div>', unsafe_allow_html=True)
            st.write(f"• **Moment Demand:** $M_u = {sel_span['Mu']:.2f}$ kNm")
            st.write(f"• **Section:** $b={b_mm:.0f}, h={h_mm:.0f}, d={d_mm:.1f}$ mm")
            st.write(f"• **Reinforcement:** {n_bot}-DB{db_main} ($A_s = {As_prov:.0f} mm^2$)")
            
            # Calc
            a = (As_prov * p['fy']) / (0.85 * p['fc'] * b_mm)
            Mn = As_prov * p['fy'] * (d_mm - a/2) * 1e-6
            phi_Mn = f['phi_m'] * Mn
            
            st.latex(rf"a = \frac{{A_s f_y}}{{0.85 f'_c b}} = {a:.2f} mm")
            st.latex(rf"\phi M_n = \phi A_s f_y (d - a/2) = {phi_Mn:.2f} kNm")
            
            if phi_Mn >= sel_span['Mu']: 
                st.markdown(f'<div class="status-pass">✅ OK (Ratio: {sel_span["Mu"]/phi_Mn:.2f})</div>', unsafe_allow_html=True)
            else: 
                st.markdown(f'<div class="status-fail">❌ FAIL (Ratio: {sel_span["Mu"]/phi_Mn:.2f})</div>', unsafe_allow_html=True)

            # SHEAR
            st.markdown('<div class="sub-head">2. SHEAR DESIGN</div>', unsafe_allow_html=True)
            Vc = 0.17 * np.sqrt(p['fc']) * b_mm * d_mm / 1000
            phi_Vc = f['phi_v'] * Vc
            Av = 2 * 28.27 # RB6
            Vs = (Av * p['fy'] * d_mm) / (s_stir*10) / 1000
            phi_Vs = f['phi_v'] * Vs
            phi_Vn = phi_Vc + phi_Vs
            
            st.write(f"• **Demand:** $V_u = {sel_span['Vu']:.2f}$ kN")
            st.latex(rf"\phi V_c = {phi_Vc:.2f} kN, \quad \phi V_s = {phi_Vs:.2f} kN")
            st.latex(rf"\phi V_n = {phi_Vn:.2f} kN")
            
            if phi_Vn >= sel_span['Vu']: st.markdown('<div class="status-pass">✅ SHEAR OK</div>', unsafe_allow_html=True)
            else: st.markdown('<div class="status-fail">❌ SHEAR FAIL</div>', unsafe_allow_html=True)
            
            # DEFLECTION
            st.markdown('<div class="sub-head">3. SERVICEABILITY</div>', unsafe_allow_html=True)
            d_all = (sel_span['L']*1000)/240
            if sel_span['def_act'] <= d_all: st.write(f"✅ Deflection: {sel_span['def_act']:.2f} mm < {d_all:.1f} mm")
            else: st.write(f"❌ Deflection: {sel_span['def_act']:.2f} mm > {d_all:.1f} mm")
            
            st.markdown('</div>', unsafe_allow_html=True) # End Frame

        # --- Right: Cross Section Image ---
        with col_img:
            st.write("### Cross Section")
            fig_sec = section_plotter.plot_section(p['b'], p['h'], cover, db_main, n_top, n_bot, f"RB6@{s_stir}cm", p['fc'], p['fy'])
            st.pyplot(fig_sec, use_container_width=True)

        # 2.3 LONGITUDINAL PROFILE (Full Width Below)
        st.markdown("---")
        st.subheader(f"Longitudinal Profile: Span {sel_span['span']}")
        fig_long = section_plotter.plot_longitudinal_detailed(sel_span['L'], p['h'], cover, n_top, n_bot, db_main, s_stir, sel_span['span'])
        st.pyplot(fig_long, use_container_width=True)
