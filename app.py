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

# Custom CSS
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
st.caption("Finite Element Analysis & RC Design | Pro Version 3.1 (English)")

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
        
        # Self-weight
        sw_kn_m = params['b'] * params['h'] * 24.0
        
        # Combine loads
        final_load_list = load_list_raw.copy()
        for i in range(len(spans)):
            final_load_list.append({
                "id": f"sw_{i}",
                "type": "U",
                "span_index": i,
                "x": 0.0,
                "mag": sw_kn_m * 1000,
                "dist": spans[i],
                "case": "DL"
            })
            
        # Solve
        beam_solver = solver.BeamSolver(spans, sup_list, final_load_list, params['E'], params['b'], params['h'], params['I'])
        res_df, reactions, status = beam_solver.solve()
        
        if "error" in status:
            st.error(f"Analysis Failed: {status['error']}")
        else:
            st.success("Analysis Complete!")
            
            design_res = []
            cum_dist = [0] + list(pd.Series(spans).cumsum())
            
            # เก็บค่า Raw Moment/Shear เพื่อไปคูณ Factor ใน Tab 2 ตามมาตรฐานที่เลือก
            for i in range(len(spans)):
                x_start = cum_dist[i]
                x_end = cum_dist[i+1]
                span_res = res_df[(res_df['x'] >= x_start) & (res_df['x'] <= x_end)]
                
                # Initial Auto Design (assume factor 1.4 for default view)
                temp_factor = 1.4
                if span_res.empty:
                    Mu_pos, Mu_neg, vu_val = 0, 0, 0
                    raw_m_pos, raw_m_neg, raw_v = 0, 0, 0
                else:
                    raw_m_pos = max(0, span_res['moment'].max()) / 1000.0
                    raw_m_neg = abs(min(0, span_res['moment'].min())) / 1000.0
                    raw_v = span_res['shear'].abs().max() / 1000.0
                    
                    Mu_pos = raw_m_pos * temp_factor
                    Mu_neg = raw_m_neg * temp_factor
                    vu_val = raw_v * temp_factor
                
                des_span = rc_design.design_span_expert(
                    Mu_pos, Mu_neg, vu_val, 
                    params['b'], params['h'], 24, 400, 40, 16
                )
                
                # Save Raw Forces
                des_span['raw_m_pos'] = raw_m_pos
                des_span['raw_m_neg'] = raw_m_neg
                des_span['raw_v'] = raw_v
                des_span['span_id'] = i
                des_span['db'] = 16
                design_res.append(des_span)

            # --- [NEW] Global Settings Block (Before Tabs) ---
            # ย้ายขึ้นมาเพื่อให้ส่งผลกับทั้ง Tab 1 และ Tab 2
            st.markdown("### ⚙️ Design Settings")
            col_set1, col_set2 = st.columns([1, 3])
            with col_set1:
                design_std = st.selectbox(
                    "📐 Select Code:",
                    ["EIT Standard (Thailand)", "ACI 318-19 (International)"]
                )
            
            # Determine Factors
            if design_std == "EIT Standard (Thailand)":
                load_factor = 1.4
                phi_flex = 0.90
                phi_shear = 0.85
                code_name = "EIT"
            else:
                load_factor = 1.4 # Simplified conservative for demo
                phi_flex = 0.90
                phi_shear = 0.75
                code_name = "ACI 318"
            
            with col_set2:
                st.info(f"**Applied Code:** {code_name} | **Load Factor:** {load_factor} | **$\phi_{{flex}}$:** {phi_flex} | **$\phi_{{shear}}$:** {phi_shear}")

            # --- Visualization Tabs ---
            t1, t2 = st.tabs(["📊 Analysis Results", "🏗️ Design & Detailing"])
            
            # === TAB 1: Analysis & Design Forces ===
            with t1:
                st.subheader("📊 Analysis Results")
                
                col_graph, col_loads = st.columns([7, 3])
                with col_graph:
                    st.markdown("#### 📈 Internal Forces Diagrams")
                    fig_ana = design_view.plot_analysis_results(res_df, spans, sup_df, final_load_list)
                    st.plotly_chart(fig_ana, use_container_width=True)
                
                with col_loads:
                    st.markdown("#### 📥 Load Summary")
                    st.info(f"**Self-weight:** {sw_kn_m:.2f} kN/m")
                    if st.session_state.load_list:
                        df_temp = pd.DataFrame(st.session_state.load_list)
                        df_temp['mag'] = (df_temp['mag'].astype(float) / 1000.0).round(2)
                        st.dataframe(df_temp[['type', 'span_index', 'mag']], use_container_width=True, hide_index=True)

                st.divider()

                # --- [RESTORED] Design Forces Section ---
                c_d1, c_d2 = st.columns(2)
                
                # 1. Moment & Shear Summary
                with c_d1:
                    st.markdown("#### 💎 Design Forces (Factored)")
                    raw_m_max = res_df['moment'].max() / 1000.0
                    raw_m_min = res_df['moment'].min() / 1000.0
                    raw_v_max = res_df['shear'].abs().max() / 1000.0
                    
                    st.write(f"**Max Positive M ($M_u^+$):**")
                    st.latex(rf"{raw_m_max:.2f} \times {load_factor} = \mathbf{{{raw_m_max * load_factor:.2f}}}\ \text{{kNm}}")
                    
                    st.write(f"**Max Negative M ($M_u^-$):**")
                    st.latex(rf"{abs(raw_m_min):.2f} \times {load_factor} = \mathbf{{{abs(raw_m_min) * load_factor:.2f}}}\ \text{{kNm}}")
                    
                    st.write(f"**Max Shear ($V_u$):**")
                    st.latex(rf"{raw_v_max:.2f} \times {load_factor} = \mathbf{{{raw_v_max * load_factor:.2f}}}\ \text{{kN}}")

                # 2. Reaction Forces Summary
                with c_d2:
                    st.markdown("#### 🏁 Reaction Forces ($R_u$)")
                    reac_data = []
                    for r_id, val in reactions.items():
                        r_serv = val / 1000.0
                        r_fact = r_serv * load_factor
                        reac_data.append({
                            "Node": f"Node {r_id}",
                            "Service (kN)": f"{r_serv:.2f}",
                            "Design (kN)": f"{r_fact:.2f}"
                        })
                        st.write(f"**Node {r_id}:** {r_serv:.2f} $\\times$ {load_factor} = **{r_fact:.2f} kN**")
                    
                    # st.dataframe(pd.DataFrame(reac_data), use_container_width=True)

            # === TAB 2: Interactive Design ===
            with t2:
                st.subheader("🏗️ Interactive Detailing")
                
                # Longitudinal Section
                fig_long = section_plotter.plot_longitudinal_section(spans, sup_df, design_res, params['h'], 40)
                fig_long.set_size_inches(12, 2.5) 
                st.pyplot(fig_long, use_container_width=True)
                
                st.divider()
                
                for i, d_data in enumerate(design_res):
                    with st.container():
                        c1, c2, c3 = st.columns([3, 3, 6])
                        
                        # --- Col 1: Inputs ---
                        with c1:
                            st.markdown(f"### 🔹 Span {i+1}")
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
                                stirrup_db = 6 

                        # --- Col 2: Visualization ---
                        with c2:
                            st.markdown("##### Section View")
                            fig_sec = section_plotter.plot_section(
                                params['b'], params['h'], new_cover, bar_size, n_top, n_bot, 
                                f"RB{stirrup_db}@{s_spacing}cm", 24, 400
                            )
                            fig_sec.set_size_inches(3.5, 3.5)
                            fig_sec.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)
                            st.pyplot(fig_sec, use_container_width=True)

                        # --- Col 3: DETAILED SUBSTITUTION ---
                        with c3:
                            st.markdown("#### 📝 Detailed Calculation Sheet")
                            
                            # Update Loads
                            Mu_req = d_data['raw_m_pos'] * load_factor
                            Vu_req = d_data['raw_v'] * load_factor
                            
                            fc, fy = 24, 400
                            b_mm, h_mm = params['b'] * 1000, params['h'] * 1000
                            
                            # --- Flexure (Detailed) ---
                            st.markdown("**1️⃣ Flexural Capacity (Bottom)**")
                            d_eff = h_mm - new_cover - stirrup_db - (bar_size/2)
                            As_prov = n_bot * (3.1416 * (bar_size/2)**2)
                            a_depth = (As_prov * fy) / (0.85 * fc * b_mm)
                            Mn_val = As_prov * fy * (d_eff - a_depth/2) / 1e6
                            phi_Mn = phi_flex * Mn_val
                            
                            # Show Substitution
                            st.latex(rf"d = {h_mm:.0f} - {new_cover} - {stirrup_db} - {bar_size}/2 = \mathbf{{{d_eff:.1f}}}\ mm")
                            st.latex(rf"A_s = {n_bot} \times 3.14 \times ({bar_size}/2)^2 = \mathbf{{{As_prov:.0f}}}\ mm^2")
                            
                            st.write("Stress Block Depth ($a$):")
                            st.latex(rf"a = \frac{{{As_prov:.0f} \cdot {fy}}}{{0.85 \cdot {fc} \cdot {b_mm:.0f}}} = \mathbf{{{a_depth:.1f}}}\ mm")
                            
                            st.write("Nominal Moment ($\phi M_n$):")
                            # แสดงบรรทัดแทนค่าตัวเลข
                            st.latex(rf"\phi M_n = {phi_flex} \left[ {As_prov:.0f} \cdot {fy} \cdot ({d_eff:.1f} - \frac{{{a_depth:.1f}}}{{2}}) \right] \cdot 10^{{-6}}")
                            # แสดงคำตอบ
                            st.latex(rf"= \mathbf{{{phi_Mn:.2f}}}\ kNm \quad (Req: {Mu_req:.2f})")
                            
                            if phi_Mn >= Mu_req:
                                st.success(f"✅ FLEXURE OK")
                            else:
                                st.error(f"❌ FLEXURE FAIL")

                            st.markdown("---")
                            
                            # --- Shear (Detailed) ---
                            st.markdown("**2️⃣ Shear Capacity**")
                            Vc = 0.17 * np.sqrt(fc) * b_mm * d_eff / 1000.0
                            Av = 2 * (3.1416 * (stirrup_db/2)**2)
                            s_mm = s_spacing * 10
                            Vs = (Av * fy * d_eff) / s_mm / 1000.0
                            phi_Vn = phi_shear * (Vc + Vs)
                            
                            st.write("Concrete Capacity ($V_c$):")
                            st.latex(rf"V_c = 0.17 \sqrt{{{fc}}} \cdot {b_mm:.0f} \cdot {d_eff:.1f} = \mathbf{{{Vc:.2f}}}\ kN")
                            
                            st.write("Steel Capacity ($V_s$):")
                            st.latex(rf"V_s = \frac{{{Av:.0f} \cdot {fy} \cdot {d_eff:.1f}}}{{{s_mm}}} = \mathbf{{{Vs:.2f}}}\ kN")
                            
                            st.write("Total Capacity ($\phi V_n$):")
                            # แสดงบรรทัดแทนค่าตัวเลข
                            st.latex(rf"\phi V_n = {phi_shear} \times ({Vc:.2f} + {Vs:.2f})")
                            st.latex(rf"= \mathbf{{{phi_Vn:.2f}}}\ kN \quad (Req: {Vu_req:.2f})")

                            if phi_Vn >= Vu_req:
                                st.success(f"✅ SHEAR OK")
                            else:
                                st.error(f"❌ SHEAR FAIL")

                    st.divider()

                # BOQ
                st.markdown("### 📋 BOQ Estimate")
                bbs_list = rc_design.generate_bbs(design_res, spans, params['b'], params['h'], 40)
                vol_conc, w_steel = rc_design.get_boq(spans, params['b'], params['h'], bbs_list)
                c_boq1, c_boq2 = st.columns(2)
                c_boq1.metric("Concrete", f"{vol_conc:.2f} m³")
                c_boq2.metric("Steel", f"{w_steel:.2f} kg")
