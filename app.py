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
st.set_page_config(
    page_title="Pro Beam Studio",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for compact layout
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        font-weight: bold;
        background-color: #2C3E50;
        color: white;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #dee2e6;
    }
    .block-container {padding-top: 1rem;}
    div[data-testid="stVerticalBlock"] > div {padding-top: 0.5rem; padding-bottom: 0.5rem;}
    h4 { color: #2C3E50; }
</style>
""", unsafe_allow_html=True)

# --- 2. Session State ---
if 'project_data' not in st.session_state:
    st.session_state.project_data = None
if 'load_list' not in st.session_state:
    st.session_state.load_list = []

# --- 3. Sidebar & Inputs ---
params = input_handler.render_sidebar()

st.title("🏗️ Professional RC Beam Designer")
st.caption("Finite Element Analysis & RC Design | Pro Version 3.0 (English)")

# File Management
col_file1, col_file2 = st.columns([1, 4])
with col_file1:
    uploaded_file = st.file_uploader("📂 Load Project (.json)", type=["json"])
    if uploaded_file:
        loaded = file_manager.load_data(uploaded_file)
        if loaded:
            st.session_state.project_data = loaded
            if 'loads' in loaded:
                st.session_state.load_list = loaded['loads']
            st.success("Loaded!")

# Inputs
n_spans, spans, sup_df, stable = input_handler.render_model_inputs(params)
loads_df = input_handler.render_loads(n_spans, spans, params, sup_df)

# Save Button
if loads_df is not None:
    load_export = loads_df.to_dict('records') if not loads_df.empty else []
    json_str = file_manager.export_data(params, spans, sup_df, load_export)
    st.sidebar.download_button(
        label="💾 Save Project",
        data=json_str,
        file_name="project.json",
        mime="application/json"
    )

st.markdown("---")

# --- 4. Main Process ---
if st.button("🚀 Run Analysis & Design", type="primary"):
    if not stable:
        st.error("🚨 Structure is Unstable! Please check supports.")
    else:
        # A. Analysis (Solver)
        st.info("Computing Finite Element Analysis...")
        
        sup_list = sup_df.to_dict('records') if not sup_df.empty else []
        load_list_raw = loads_df.to_dict('records') if (loads_df is not None and not loads_df.empty) else []
        
        # Self-weight Calculation
        sw_kn_m = params['b'] * params['h'] * 24.0  # kN/m
        
        # Combine loads
        final_load_list = load_list_raw.copy()
        for i in range(len(spans)):
            final_load_list.append({
                "id": f"sw_{i}",
                "type": "U",
                "span_index": i,
                "x": 0.0,
                "mag": sw_kn_m * 1000, # N/m
                "dist": spans[i],
                "case": "DL"
            })
            
        # Solver
        beam_solver = solver.BeamSolver(spans, sup_list, final_load_list, params['E'], params['b'], params['h'], params['I'])
        res_df, reactions, status = beam_solver.solve()
        
        if "error" in status:
            st.error(f"Analysis Failed: {status['error']}")
        else:
            # B. RC Design logic (Initial Pass)
            st.success("Analysis Complete!")
            
            design_res = []
            cum_dist = [0] + list(pd.Series(spans).cumsum())
            
            # Temporary Factor for initial auto-design (user can change in Tab 2)
            temp_factor = 1.4 
            
            for i in range(len(spans)):
                x_start = cum_dist[i]
                x_end = cum_dist[i+1]
                span_res = res_df[(res_df['x'] >= x_start) & (res_df['x'] <= x_end)]
                
                if span_res.empty:
                    Mu_pos, Mu_neg, vu_val = 0, 0, 0
                else:
                    Mu_pos = (max(0, span_res['moment'].max()) * temp_factor) / 1000.0
                    Mu_neg = (abs(min(0, span_res['moment'].min())) * temp_factor) / 1000.0
                    vu_val = (span_res['shear'].abs().max() * temp_factor) / 1000.0
                
                # Initial Auto Design
                des_span = rc_design.design_span_expert(
                    Mu_pos, Mu_neg, vu_val, 
                    params['b'], params['h'], 
                    24, 400, # fc, fy
                    40, 16   # Cover, db main
                )
                
                # Store Unfactored (Service) Loads for Dynamic Calculation in Tab 2
                des_span['raw_m_pos'] = max(0, span_res['moment'].max()) / 1000.0
                des_span['raw_m_neg'] = abs(min(0, span_res['moment'].min())) / 1000.0
                des_span['raw_v'] = span_res['shear'].abs().max() / 1000.0
                
                des_span['span_id'] = i
                des_span['db'] = 16
                design_res.append(des_span)
            

            # --- 5. Visualization Results ---
            t1, t2 = st.tabs(["📊 Analysis Results", "🏗️ Design & Calculation Report"])
            
            # === TAB 1: Analysis ===
            with t1:
                st.subheader("📊 Analysis Results")
                col_graph, col_loads = st.columns([7, 3])
                
                with col_graph:
                    st.markdown("#### 📈 Diagrams (M, V, D)")
                    fig_ana = design_view.plot_analysis_results(res_df, spans, sup_df, final_load_list)
                    st.plotly_chart(fig_ana, use_container_width=True)
                
                with col_loads:
                    st.markdown("#### 📥 Load Summary")
                    sw_val = params['b'] * params['h'] * 24.0
                    st.info(f"**Self-weight:** {sw_val:.2f} kN/m")
                    
                    if st.session_state.load_list:
                        df_temp = pd.DataFrame(st.session_state.load_list)
                        df_temp['mag'] = (df_temp['mag'].astype(float) / 1000.0).round(2)
                        st.dataframe(df_temp[['type', 'span_index', 'mag']], use_container_width=True, hide_index=True)

                st.markdown("#### 🏁 Reaction Forces (Service Load)")
                reac_data = [{"Node": f"Node {k}", "Vertical Reaction (kN)": v/1000.0} for k, v in reactions.items()]
                st.dataframe(pd.DataFrame(reac_data).T, use_container_width=True)

            # === TAB 2: Interactive Design & Detailed Report ===
            with t2:
                # Header & Settings
                c_head1, c_head2 = st.columns([2, 1])
                with c_head1:
                    st.subheader("🏗️ Interactive Detailing & Calculation")
                with c_head2:
                    # --- Standard Selection ---
                    design_std = st.selectbox(
                        "📐 Design Standard / Code",
                        ["EIT Standard (Thailand)", "ACI 318-19 (International)"]
                    )
                
                # Set Factors based on Standard
                if design_std == "EIT Standard (Thailand)":
                    load_factor = 1.4   # Simple Conservative (1.4DL + 1.7LL approx)
                    phi_flex = 0.90
                    phi_shear = 0.85
                    std_label = "EIT (WSD/SDM Adapted)"
                else:
                    load_factor = 1.2   # ACI (1.2DL + 1.6LL approx) - Simplified to conservative 1.4 for demo or user adjustable
                    # Note: For this demo, let's keep load_factor distinct to show difference
                    load_factor = 1.4 # Keep consistent or make adjustable. Let's use 1.4 for safety.
                    phi_flex = 0.90
                    phi_shear = 0.75
                    std_label = "ACI 318-19"

                st.info(f"**Current Settings:** Code: {std_label} | Load Factor: {load_factor} | $\phi_{{flex}}={phi_flex}$ | $\phi_{{shear}}={phi_shear}$")

                # Longitudinal Plot
                st.markdown("#### 📏 Longitudinal Profile")
                fig_long = section_plotter.plot_longitudinal_section(spans, sup_df, design_res, params['h'], 40)
                fig_long.set_size_inches(12, 2.5) 
                st.pyplot(fig_long, use_container_width=True)
                
                st.divider()
                
                # Interactive Loop
                for i, d_data in enumerate(design_res):
                    with st.container():
                        # Layout: Control (3) | Visual (3) | Detailed Calc (6)
                        c1, c2, c3 = st.columns([3, 3, 6])
                        
                        # --- Col 1: Control Panel ---
                        with c1:
                            st.markdown(f"### 🔹 Span {i+1}")
                            st.caption(f"Size: {params['b']} x {params['h']} m")
                            
                            with st.expander("⚙️ Edit Reinforcement", expanded=True):
                                new_cover = st.number_input(f"Cover (mm) - Sp{i+1}", 20, 75, 40, 5, key=f"cov_{i}")
                                
                                col_b1, col_b2 = st.columns(2)
                                with col_b1:
                                    n_top = st.number_input(f"Top Bars", 1, 10, int(d_data['neg']['n']), key=f"nt_{i}")
                                with col_b2:
                                    n_bot = st.number_input(f"Bot Bars", 1, 10, int(d_data['pos']['n']), key=f"nb_{i}")
                                
                                bar_size = st.selectbox(f"Main Bar DB (mm)", [12, 16, 20, 25, 28], index=1, key=f"db_{i}")
                                
                                st.markdown("---")
                                s_spacing = st.number_input(f"Stirrup Spacing (cm)", 5, 30, 15, 5, key=f"s_{i}")
                                stirrup_db = 6 # Fix RB6 or DB10

                        # --- Col 2: Visualization ---
                        with c2:
                            st.markdown("##### Section View")
                            fig_sec = section_plotter.plot_section(
                                params['b'], params['h'], new_cover, bar_size,
                                n_top, n_bot, 
                                f"RB{stirrup_db}@{s_spacing}cm", 24, 400
                            )
                            fig_sec.set_size_inches(3.5, 3.5)
                            fig_sec.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)
                            st.pyplot(fig_sec, use_container_width=True)

                        # --- Col 3: Detailed Calculation Report ---
                        with c3:
                            st.markdown("#### 📝 Detailed Calculation Sheet")
                            
                            # 1. Recalculate Loads based on chosen Standard
                            Mu_req = d_data['raw_m_pos'] * load_factor
                            Vu_req = d_data['raw_v'] * load_factor
                            
                            # 2. Material Properties
                            fc = 24 # MPa
                            fy = 400 # MPa
                            b_mm = params['b'] * 1000
                            h_mm = params['h'] * 1000
                            
                            # 3. Flexural Calculation (Bottom Steel)
                            st.markdown("**1️⃣ Flexural Capacity Check (Positive Moment)**")
                            
                            d_eff = h_mm - new_cover - stirrup_db - (bar_size/2)
                            As_prov = n_bot * (3.1416 * (bar_size/2)**2)
                            
                            # Calculate 'a' block
                            a_depth = (As_prov * fy) / (0.85 * fc * b_mm)
                            # Calculate Mn
                            Mn_kNm = As_prov * fy * (d_eff - a_depth/2) / 1e6
                            phi_Mn = phi_flex * Mn_kNm
                            
                            # Display Flexure Steps
                            col_calc1, col_calc2 = st.columns(2)
                            with col_calc1:
                                st.latex(rf"d = {h_mm:.0f} - {new_cover} - {stirrup_db} - \frac{{{bar_size}}}{{2}} = \mathbf{{{d_eff:.1f}}}\ mm")
                                st.latex(rf"A_s = {n_bot} \times \pi \frac{{{bar_size}^2}}{{4}} = \mathbf{{{As_prov:.0f}}}\ mm^2")
                            with col_calc2:
                                st.latex(rf"a = \frac{{A_s f_y}}{{0.85 f'_c b}} = \frac{{{As_prov:.0f} \cdot {fy}}}{{0.85 \cdot {fc} \cdot {b_mm:.0f}}} = {a_depth:.1f}\ mm")
                                st.latex(rf"\phi M_n = {phi_flex} \cdot A_s f_y (d - a/2)")
                            
                            st.latex(rf"\phi M_n = \mathbf{{{phi_Mn:.2f}}}\ kNm \quad vs \quad M_u = \mathbf{{{Mu_req:.2f}}}\ kNm")
                            
                            if phi_Mn >= Mu_req:
                                st.success(f"✅ FLEXURE OK (Ratio: {Mu_req/phi_Mn:.2f})")
                            else:
                                st.error(f"❌ FLEXURE FAIL (Req: {Mu_req:.2f} kNm)")

                            st.markdown("---")
                            
                            # 4. Shear Calculation
                            st.markdown("**2️⃣ Shear Capacity Check**")
                            
                            # Vc (Concrete Capacity)
                            # Simplified ACI: 0.17 * sqrt(fc) * b * d
                            Vc = 0.17 * np.sqrt(fc) * b_mm * d_eff / 1000.0 # kN
                            
                            # Vs (Steel Capacity)
                            # Vs = Av * fy * d / s
                            Av = 2 * (3.1416 * (stirrup_db/2)**2) # 2 legs
                            s_mm = s_spacing * 10
                            Vs = (Av * fy * d_eff) / s_mm / 1000.0 # kN
                            
                            Vn = Vc + Vs
                            phi_Vn = phi_shear * Vn
                            
                            # Display Shear Steps
                            col_s1, col_s2 = st.columns(2)
                            with col_s1:
                                st.latex(rf"V_c = 0.17 \sqrt{{f'_c}} b d = \mathbf{{{Vc:.2f}}}\ kN")
                                st.latex(rf"A_v (2\ legs) = \mathbf{{{Av:.0f}}}\ mm^2")
                            with col_s2:
                                st.latex(rf"V_s = \frac{{A_v f_{{yt}} d}}{{s}} = \frac{{{Av:.0f} \cdot {fy} \cdot {d_eff:.0f}}}{{{s_mm}}} = \mathbf{{{Vs:.2f}}}\ kN")
                                st.latex(rf"\phi V_n = {phi_shear}(V_c + V_s) = \mathbf{{{phi_Vn:.2f}}}\ kN")
                            
                            st.latex(rf"\phi V_n = \mathbf{{{phi_Vn:.2f}}}\ kN \quad vs \quad V_u = \mathbf{{{Vu_req:.2f}}}\ kN")
                            
                            if phi_Vn >= Vu_req:
                                st.success(f"✅ SHEAR OK (Ratio: {Vu_req/phi_Vn:.2f})")
                            else:
                                st.error(f"❌ SHEAR FAIL (Increase Stirrups)")

                    st.divider()

                # BOQ Summary at bottom
                st.markdown("### 📋 Bill of Quantities (Estimate)")
                bbs_list = rc_design.generate_bbs(design_res, spans, params['b'], params['h'], 40)
                vol_conc, w_steel = rc_design.get_boq(spans, params['b'], params['h'], bbs_list)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Concrete Vol.", f"{vol_conc:.2f} m³")
                m2.metric("Steel Weight", f"{w_steel:.2f} kg")
                m3.metric("Steel Ratio", f"{(w_steel/vol_conc) if vol_conc>0 else 0:.1f} kg/m³")
