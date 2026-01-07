import streamlit as st
import pandas as pd
import numpy as np

# Import modules
try:
    import input_handler
    import file_manager
    import solver
    import rc_design
    import design_view
    import section_plotter
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# --- 1. Page Config ---
st.set_page_config(page_title="Pro Beam Studio", layout="wide")

# Custom CSS for Dashboard Look
st.markdown("""
<style>
    .stMetric {
        background-color: #F0F2F6;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #D1D5DB;
    }
    .stButton>button {
        width: 100%;
        font-weight: bold;
        background-color: #2C3E50;
        color: white;
        border-radius: 8px;
    }
    h4 { color: #1F2937; margin-top: 0px; }
    .block-container { padding-top: 2rem; }
    /* ปรับ Header ใน Sidebar */
    [data-testid=stSidebar] h1, [data-testid=stSidebar] h2, [data-testid=stSidebar] h3 {
        color: #2C3E50;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Session State Management ---
if 'project_data' not in st.session_state:
    st.session_state.project_data = None
if 'load_list' not in st.session_state:
    st.session_state.load_list = []
# Store Analysis Results independently to allow switching Codes without re-solving
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None 
if 'reactions' not in st.session_state:
    st.session_state.reactions = None

# ==========================================
# --- 3. Sidebar (Settings & Inputs) ---
# ==========================================
with st.sidebar:
    st.title("⚙️ Design Settings")
    
    # --- [MOVED HERE] Design Standard Selection ---
    design_std = st.radio(
        "📐 Design Code / Standard",
        ["ACI 318-19 (International)", "EIT Standard (Thailand)"]
    )

    # Determine Factors based on selection (Dynamic Update)
    if design_std == "EIT Standard (Thailand)":
        load_factor = 1.4 # Simplified conservative
        phi_flex = 0.90
        phi_shear = 0.85
        std_label = "EIT (WSD/SDM)"
    else:
        load_factor = 1.4 # Simplified for demo consistency
        phi_flex = 0.90
        phi_shear = 0.75
        std_label = "ACI 318-19"

    # Display active factors in Sidebar
    st.markdown(f"""
    <div style="background-color: #e8f4f8; padding: 10px; border-radius: 5px; border-left: 4px solid #00a8e8;">
        <small>Active Factors:</small><br>
        <b>Load Factor:</b> {load_factor}<br>
        <b>ϕ (Flexure):</b> {phi_flex}<br>
        <b>ϕ (Shear):</b> {phi_shear}
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()

    # --- Model Inputs (Existing) ---
    st.header("1. Model Geometry")
    params = input_handler.render_sidebar_params() # Assuming this function exists in your handler
    n_spans, spans, sup_df, stable = input_handler.render_model_inputs_main(params) # Adjusted call

    st.header("2. Loads Definition")
    loads_df = input_handler.render_loads_main(n_spans, spans, params, sup_df) # Adjusted call

    st.divider()
    
    # --- Run Analysis Button ---
    run_analysis = st.button("🚀 Run Analysis", type="primary")

# ==========================================
# --- 4. Main Area (Results Display) ---
# ==========================================
st.title("🏗️ Professional RC Beam Designer")
st.caption(f"Finite Element Analysis & RC Design | Code: {std_label} | Pro Version 3.3")

# --- Run Analysis Logic ---
if run_analysis:
    if not stable:
        st.error("🚨 Structure is Unstable! Please check supports.")
    else:
        with st.spinner("Computing FEM Analysis..."):
            # Prepare Loads
            sw_kn_m = params['b'] * params['h'] * 24.0
            sup_list = sup_df.to_dict('records') if not sup_df.empty else []
            load_list_raw = loads_df.to_dict('records') if (loads_df is not None and not loads_df.empty) else []
            
            final_load_list = load_list_raw.copy()
            for i in range(len(spans)):
                final_load_list.append({
                    "id": f"sw_{i}", "type": "U", "span_index": i, "x": 0.0,
                    "mag": sw_kn_m * 1000, "dist": spans[i], "case": "DL"
                })
            
            # Solve
            solver_inst = solver.BeamSolver(spans, sup_list, final_load_list, params['E'], params['b'], params['h'], params['I'])
            res_df, reactions, status = solver_inst.solve()
            
            if "error" in status:
                st.error(status['error'])
            else:
                # Store in Session State
                st.session_state.analysis_results = res_df
                st.session_state.reactions = reactions
                st.session_state.final_load_list = final_load_list
                st.success("Analysis Complete!")

# --- 5. Display Results (If Analysis Exists) ---
if st.session_state.analysis_results is not None:
    res_df = st.session_state.analysis_results
    reactions = st.session_state.reactions
    final_load_list = st.session_state.final_load_list
    
    # Pre-process Data for Design
    design_res = []
    cum_dist = [0] + list(pd.Series(spans).cumsum())
    
    for i in range(len(spans)):
        x_start, x_end = cum_dist[i], cum_dist[i+1]
        span_res = res_df[(res_df['x'] >= x_start) & (res_df['x'] <= x_end)]
        
        # Get RAW forces (Unfactored)
        if span_res.empty:
            raw_m_pos, raw_m_neg, raw_v = 0, 0, 0
        else:
            raw_m_pos = max(0, span_res['moment'].max()) / 1000.0
            raw_m_neg = abs(min(0, span_res['moment'].min())) / 1000.0
            raw_v = span_res['shear'].abs().max() / 1000.0
        
        # Apply Factor locally for Initial Auto-Design suggestion
        des_span = rc_design.design_span_expert(
            raw_m_pos * load_factor, raw_m_neg * load_factor, raw_v * load_factor, 
            params['b'], params['h'], 24, 400, 40, 16
        )
        # Attach raw data for dynamic recalculation
        des_span.update({'raw_m_pos': raw_m_pos, 'raw_m_neg': raw_m_neg, 'raw_v': raw_v, 'span_id': i})
        design_res.append(des_span)

    # --- TABS ---
    t1, t2 = st.tabs(["📊 Analysis Results", "🏗️ Design & Calculation"])
    
    # ==========================
    # TAB 1: Dashboard Layout
    # ==========================
    with t1:
        # Row 1: Key Metrics (Service Load)
        m_max = res_df['moment'].max() / 1000.0
        m_min = res_df['moment'].min() / 1000.0
        v_max = res_df['shear'].abs().max() / 1000.0
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Max Moment (+)", f"{m_max:.2f} kNm")
        col_m2.metric("Max Moment (-)", f"{m_min:.2f} kNm")
        col_m3.metric("Max Shear", f"{v_max:.2f} kN")
        col_m4.metric("Max Deflection", f"{res_df['deflection'].abs().max():.2f} mm")
        
        st.markdown("---")
        
        # Row 2: Full Width Graph
        st.markdown("#### 📈 Internal Force Diagrams")
        fig_ana = design_view.plot_analysis_results(res_df, spans, sup_df, final_load_list)
        fig_ana.update_layout(height=500)
        st.plotly_chart(fig_ana, use_container_width=True)
        
        # Row 3: Split Tables (Reactions & Factored Forces)
        st.markdown("---")
        col_b1, col_b2 = st.columns([1, 1])
        
        with col_b1:
            st.markdown("#### 🏁 Reaction Forces (Service)")
            reac_data = [{"Node": k, "Reaction (kN)": f"{v/1000:.2f}"} for k, v in reactions.items()]
            st.table(pd.DataFrame(reac_data))
            
        with col_b2:
            st.markdown(f"#### 💎 Critical Design Forces (Factored x{load_factor})")
            force_data = [
                {"Type": "Moment (+)", "Value": f"{m_max * load_factor:.2f} kNm"},
                {"Type": "Moment (-)", "Value": f"{abs(m_min) * load_factor:.2f} kNm"},
                {"Type": "Shear (V)", "Value": f"{v_max * load_factor:.2f} kN"}
            ]
            st.table(pd.DataFrame(force_data))

    # ==========================
    # TAB 2: Interactive Design
    # ==========================
    with t2:
        # Longitudinal Profile
        st.markdown("#### 📏 Reinforcement Profile")
        # Call the updated plotter function
        fig_long = section_plotter.plot_longitudinal_section_detailed(spans, sup_df, design_res, params['h'])
        fig_long.set_size_inches(12, 3) # Adjust height
        st.pyplot(fig_long, use_container_width=True)
        
        st.divider()
        
        # Loop through Spans
        for i, d_data in enumerate(design_res):
            with st.container():
                st.markdown(f"### 🔹 Span {i+1} (L = {spans[i]} m)")
                
                # Layout: 2 Columns (Left: Inputs+Visual, Right: Calc Sheet)
                c_left, c_right = st.columns([4, 6])
                
                with c_left:
                    # Input Box style
                    with st.expander("🛠️ Reinforcement Config", expanded=True):
                        col_in1, col_in2 = st.columns(2)
                        with col_in1:
                            n_top = st.number_input(f"Top Bars", 1, 10, int(d_data['neg']['n']), key=f"nt_{i}")
                            n_bot = st.number_input(f"Bot Bars", 1, 10, int(d_data['pos']['n']), key=f"nb_{i}")
                        with col_in2:
                            bar_size = st.selectbox(f"Main DB (mm)", [12, 16, 20, 25, 28], index=1, key=f"db_{i}")
                            s_spacing = st.number_input(f"Stirrup @ (cm)", 5, 30, 15, 5, key=f"s_{i}")
                        
                        new_cover = st.number_input(f"Cover (mm)", 20, 75, 40, 5, key=f"cov_{i}")
                        stirrup_db = 6

                    # Section Visual
                    st.markdown("**Section View:**")
                    fig_sec = section_plotter.plot_section(
                        params['b'], params['h'], new_cover, bar_size, n_top, n_bot, 
                        f"RB{stirrup_db}@{s_spacing}cm", 24, 400
                    )
                    # --- [FIX] ปรับขนาดรูปตัดขวางให้กระชับ ---
                    fig_sec.set_size_inches(4, 4)
                    fig_sec.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
                    st.pyplot(fig_sec, use_container_width=True)

                with c_right:
                    # --- Calculation Sheet ---
                    st.markdown("#### 📝 Detailed Calculation Sheet")
                    
                    # Update Forces based on selected Standard
                    Mu_req = d_data['raw_m_pos'] * load_factor
                    Vu_req = d_data['raw_v'] * load_factor
                    
                    fc, fy = 24, 400
                    b_mm, h_mm = params['b'] * 1000, params['h'] * 1000
                    
                    # 1. Flexure Math
                    st.markdown(f"##### 1️⃣ Flexural Capacity (+M) | $\phi={phi_flex}$")
                    d_eff = h_mm - new_cover - stirrup_db - (bar_size/2)
                    As_prov = n_bot * (3.1416 * (bar_size/2)**2)
                    a_depth = (As_prov * fy) / (0.85 * fc * b_mm)
                    Mn_val = As_prov * fy * (d_eff - a_depth/2) / 1e6
                    phi_Mn = phi_flex * Mn_val
                    
                    # Formula -> Sub -> Result
                    st.latex(r"a = \frac{A_s f_y}{0.85 f'_c b}")
                    st.latex(rf"a = \frac{{{As_prov:.0f} \cdot {fy}}}{{0.85 \cdot {fc} \cdot {b_mm:.0f}}} = \mathbf{{{a_depth:.1f}}}\ mm")
                    
                    st.latex(r"\phi M_n = \phi \cdot A_s f_y (d - a/2)")
                    st.latex(rf"\phi M_n = {phi_flex} \cdot [{As_prov:.0f} \cdot {fy} ({d_eff:.1f} - \frac{{{a_depth:.1f}}}{{2}})] \cdot 10^{{-6}}")
                    st.latex(rf"\phi M_n = \mathbf{{{phi_Mn:.2f}}}\ kNm \quad (Req: {Mu_req:.2f})")
                    
                    if phi_Mn >= Mu_req:
                        st.success("✅ FLEXURE PASS")
                    else:
                        st.error("❌ FLEXURE FAIL")
                        
                    st.divider()

                    # 2. Shear Math
                    st.markdown(f"##### 2️⃣ Shear Capacity | $\phi={phi_shear}$")
                    Vc = 0.17 * np.sqrt(fc) * b_mm * d_eff / 1000.0
                    Av = 2 * (3.1416 * (stirrup_db/2)**2)
                    s_mm = s_spacing * 10
                    Vs = (Av * fy * d_eff) / s_mm / 1000.0
                    phi_Vn = phi_shear * (Vc + Vs)
                    
                    st.latex(r"V_c = 0.17 \sqrt{f'_c} b d")
                    st.latex(rf"V_c = 0.17 \sqrt{{{fc}}} \cdot {b_mm:.0f} \cdot {d_eff:.1f} = \mathbf{{{Vc:.2f}}}\ kN")
                    
                    st.latex(r"V_s = \frac{A_v f_{yt} d}{s}")
                    st.latex(rf"V_s = \frac{{{Av:.0f} \cdot {fy} \cdot {d_eff:.1f}}}{{{s_mm}}} = \mathbf{{{Vs:.2f}}}\ kN")
                    
                    st.latex(r"\phi V_n = \phi (V_c + V_s)")
                    st.latex(rf"\phi V_n = {phi_shear} ({Vc:.2f} + {Vs:.2f}) = \mathbf{{{phi_Vn:.2f}}}\ kN")
                    
                    if phi_Vn >= Vu_req:
                        st.success("✅ SHEAR PASS")
                    else:
                        st.error("❌ SHEAR FAIL")

            st.divider()
