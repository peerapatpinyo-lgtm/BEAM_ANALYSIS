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

# --- CSS: Clean Engineering Look ---
st.markdown("""
<style>
    .report-box { border: 1px solid #ccc; padding: 25px; background-color: #ffffff; margin-bottom: 20px; }
    .header-box { border-bottom: 2px solid #000; padding-bottom: 10px; margin-bottom: 20px; }
    .header-text { font-size: 20px; font-weight: bold; color: #000; }
    .sub-header { font-size: 16px; font-weight: bold; color: #444; margin-top: 15px; border-left: 4px solid #007bff; padding-left: 10px; }
    .pass { color: green; font-weight: bold; }
    .fail { color: red; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- Init Session State ---
if 'analyzed' not in st.session_state: st.session_state.analyzed = False
if 'avg_factor' not in st.session_state: st.session_state.avg_factor = 1.6 
if 'full_loads' not in st.session_state: st.session_state.full_loads = [] 
if 'sw_val' not in st.session_state: st.session_state.sw_val = 0.0

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Design Parameters")
    design_std = st.radio("Design Code", ["ACI 318", "EIT 1008"])
    
    if "EIT" in design_std:
        factors = {'DL': 1.4, 'LL': 1.7, 'phi_m': 0.90, 'phi_v': 0.85, 'name': 'EIT 1008'}
        current_avg_factor = 1.7 
    else:
        factors = {'DL': 1.2, 'LL': 1.6, 'phi_m': 0.90, 'phi_v': 0.75, 'name': 'ACI 318'}
        current_avg_factor = 1.6
        
    params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()
    st.divider()
    run_btn = st.button("RUN ANALYSIS", type="primary", use_container_width=True)

# --- MAIN APP ---
st.title(f"🏗️ RC BEAM ANALYSIS & DESIGN")

if run_btn:
    if not stable:
        st.error("Structure Unstable!")
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
    sw_val = st.session_state.get('sw_val', 0.0)
    
    # 1. ANALYSIS RESULTS
    st.subheader("1. Analysis Results")
    fig = design_view.plot_analysis_results(res, spans, sup_df, full_loads, reac)
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()

    # 2. DESIGN SECTION
    st.subheader("2. Design & Calculation")
    
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

    # --- INPUT FORM (Redesigned) ---
    with st.container(border=True):
        st.markdown("#### 🛠️ Reinforcement Input")
        col_sel, col_main, col_shear = st.columns([1, 2, 1.5])
        
        with col_sel:
            sel_span = st.selectbox("Select Span", design_data, format_func=lambda x: f"Span {x['span']}")
        
        with st.form("rebar_input"):
            c1, c2, c3, c4 = st.columns(4)
            n_top = c1.number_input("Top Bars (Qty)", 2, 10, 2)
            n_bot = c2.number_input("Bottom Bars (Qty)", 2, 10, 3) # Min 2 for corners
            db_main = c3.selectbox("Main Rebar DB (mm)", [12, 16, 20, 25, 28], index=1)
            cover = c4.number_input("Cover (mm)", 20, 50, 40)
            
            c5, c6, c7 = st.columns([1, 1, 2])
            s_stir = c5.number_input("Stirrup Spacing (cm)", 5, 50, 15, 5)
            # spacer
            submitted = st.form_submit_button("Update Calculation", type="primary")

    # --- CALCULATION SHEET (Detailed Restoration) ---
    st.markdown('<div class="report-box">', unsafe_allow_html=True)
    st.markdown(f'<div class="header-box"><span class="header-text">📝 ENGINEER CALCULATION SHEET: SPAN {sel_span["span"]}</span></div>', unsafe_allow_html=True)

    # Variables
    b, h = p['b']*1000, p['h']*1000
    d = h - cover - 9 - db_main/2 # Estimate stirrup 9mm
    As_prov = n_bot * 3.1416 * (db_main/2)**2
    
    # --- PART 1: FLEXURE ---
    st.markdown('<div class="sub-header">1. Flexural Strength Design (Moment)</div>', unsafe_allow_html=True)
    st.write("**(a) Factored Load & Moment**")
    st.latex(rf"M_u = \mathbf{{{sel_span['Mu']:.2f}}}\ kNm")
    
    st.write("**(b) Provided Reinforcement**")
    st.latex(rf"A_{{s,prov}} = {n_bot} \times \pi ({db_main}/2)^2 = \mathbf{{{As_prov:.2f}}}\ mm^2")
    
    st.write("**(c) Check Minimum Steel**")
    As_min1 = (0.25 * np.sqrt(p['fc']) / p['fy']) * b * d
    As_min2 = (1.4 / p['fy']) * b * d
    As_min = max(As_min1, As_min2)
    st.latex(rf"A_{{s,min}} = \max \left( \frac{{0.25\sqrt{{f'_c}}}}{{f_y}}b_wd, \frac{{1.4}}{{f_y}}b_wd \right) = {As_min:.2f}\ mm^2")
    
    if As_prov >= As_min: st.markdown('<span class="pass">✅ OK (As > As_min)</span>', unsafe_allow_html=True)
    else: st.markdown('<span class="fail">❌ FAIL (Increase Rebar)</span>', unsafe_allow_html=True)

    st.write("**(d) Flexural Capacity**")
    a_depth = (As_prov * p['fy']) / (0.85 * p['fc'] * b)
    Mn = As_prov * p['fy'] * (d - a_depth/2) * 1e-6
    phi_Mn = f['phi_m'] * Mn
    
    st.latex(rf"a = \frac{{A_s f_y}}{{0.85 f'_c b}} = {a_depth:.2f}\ mm")
    st.latex(rf"\phi M_n = \phi A_s f_y (d - a/2) = {f['phi_m']} \times {Mn:.2f} = \mathbf{{{phi_Mn:.2f}}}\ kNm")
    
    if phi_Mn >= sel_span['Mu']: 
        st.markdown(f'<span class="pass">✅ SAFE (Capacity > Demand)</span>', unsafe_allow_html=True)
    else: 
        st.markdown(f'<span class="fail">❌ UNSAFE (Capacity < Demand)</span>', unsafe_allow_html=True)

    # --- PART 2: SHEAR ---
    st.markdown('<div class="sub-header">2. Shear Strength Design</div>', unsafe_allow_html=True)
    st.write(f"**(a) Shear Demand:** $V_u = {sel_span['Vu']:.2f}$ kN")
    
    Vc = 0.17 * np.sqrt(p['fc']) * b * d / 1000
    phi_Vc = f['phi_v'] * Vc
    st.write("**(b) Concrete Capacity:**")
    st.latex(rf"\phi V_c = {f['phi_v']} \times 0.17\sqrt{{f'_c}} b d = \mathbf{{{phi_Vc:.2f}}}\ kN")
    
    Av = 2 * (3.1416 * 3**2) # RB6 assumed area approx
    Vs = (Av * p['fy'] * d) / (s_stir*10) / 1000
    phi_Vs = f['phi_v'] * Vs
    st.write(f"**(c) Stirrup Capacity (RB6@{s_stir}cm):**")
    st.latex(rf"\phi V_s = \frac{{\phi A_v f_y d}}{{s}} = \mathbf{{{phi_Vs:.2f}}}\ kN")
    
    phi_Vn = phi_Vc + phi_Vs
    st.write("**(d) Total Capacity:**")
    st.latex(rf"\phi V_n = \phi V_c + \phi V_s = \mathbf{{{phi_Vn:.2f}}}\ kN")
    
    if phi_Vn >= sel_span['Vu']: st.markdown('<span class="pass">✅ SAFE</span>', unsafe_allow_html=True)
    else: st.markdown('<span class="fail">❌ UNSAFE (Reduce stirrup spacing)</span>', unsafe_allow_html=True)

    # --- PART 3: DEFLECTION ---
    st.markdown('<div class="sub-header">3. Serviceability Check</div>', unsafe_allow_html=True)
    L_mm = sel_span['L'] * 1000
    d_all = L_mm / 240
    d_act = sel_span['def_act']
    st.latex(rf"\Delta_{{allow}} = L/240 = {d_all:.2f}\ mm")
    st.latex(rf"\Delta_{{actual}} = {d_act:.2f}\ mm")
    if d_act <= d_all: st.markdown('<span class="pass">✅ PASS</span>', unsafe_allow_html=True)
    else: st.markdown('<span class="fail">❌ FAIL</span>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # --- DRAWINGS ---
    st.subheader("3. Detail Drawings")
    c_d1, c_d2 = st.columns([1, 2])
    
    with c_d1:
        st.caption("Cross Section")
        fig_sec = section_plotter.plot_section(p['b'], p['h'], cover, db_main, n_top, n_bot, f"RB6@{s_stir}cm", p['fc'], p['fy'])
        st.pyplot(fig_sec, use_container_width=True)
    
    with c_d2:
        st.caption("Longitudinal Profile")
        fig_long = section_plotter.plot_longitudinal_detailed(sel_span['L'], p['h'], cover, n_top, n_bot, db_main, s_stir, sel_span['span'])
        st.pyplot(fig_long, use_container_width=True)
