import streamlit as st
import pandas as pd

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
st.caption("Finite Element Analysis & ACI 318 Design | Pro Version 2.0")

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
        file_name="my_beam_project.json",
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
        
        # --- [NEW] Self-weight Calculation ---
        sw_kn_m = params['b'] * params['h'] * 24.0  # kN/m
        
        # รวมโหลดเดิมกับ Self-weight เข้าด้วยกันเป็น final_load_list เพื่อใช้คำนวณและวาดกราฟ
        final_load_list = load_list_raw.copy()
        for i in range(len(spans)):
            final_load_list.append({
                "id": f"sw_{i}",
                "type": "U",
                "span_index": i,
                "x": 0.0,
                "mag": sw_kn_m * 1000, # แปลงกลับเป็น N เพื่อ Solver
                "dist": spans[i],
                "case": "DL"
            })
        
        # ส่งรายการโหลดที่รวม Self-weight แล้วเข้า Solver
        beam_solver = solver.BeamSolver(spans, sup_list, final_load_list, params['E'], params['b'], params['h'], params['I'])
        res_df, reactions, status = beam_solver.solve()
        
        if "error" in status:
            st.error(f"Analysis Failed: {status['error']}")
        else:
            # B. RC Design logic
            st.success("Analysis Complete! Running Concrete Design...")
            
            design_res = []
            cum_dist = [0] + list(pd.Series(spans).cumsum())
            
            for i in range(len(spans)):
                x_start = cum_dist[i]
                x_end = cum_dist[i+1]
                span_res = res_df[(res_df['x'] >= x_start) & (res_df['x'] <= x_end)]
                
                if span_res.empty:
                    Mu_pos, Mu_neg, vu_val = 0, 0, 0
                else:
                    # ใช้ Factor แยกตามที่ตั้งใน Sidebar
                    g_dl = params.get('gamma_dead', 1.4)
                    g_ll = params.get('gamma_live', 1.7)
                    factor = max(g_dl, g_ll) # Simplified factor สำหรับตัวอย่างนี้
                    
                    # แปลงหน่วยจาก N-m เป็น kN-m เพื่อส่งให้ rc_design
                    Mu_pos = (max(0, span_res['moment'].max()) * factor) / 1000.0
                    Mu_neg = (abs(min(0, span_res['moment'].min())) * factor) / 1000.0
                    vu_val = (span_res['shear'].abs().max() * factor) / 1000.0
                
                des_span = rc_design.design_span_expert(
                    Mu_pos, Mu_neg, vu_val, 
                    params['b'], params['h'], 
                    24, 400, # fc, fy
                    40, 16   # Cover, db main
                )
                des_span['span_id'] = i
                des_span['db'] = 16
                design_res.append(des_span)
            
            # --- 5. Visualization Results ---
            t1, t2, t3 = st.tabs(["📊 Analysis Results", "🏗️ Design & Detailing", "📝 Calculation Report"])
            
            with t1:
                st.subheader("Structure & Diagrams")
                # แก้ไข NameError: ใช้ final_load_list เพื่อแสดง Self-weight ในกราฟด้วย
                fig_ana = design_view.plot_analysis_results(res_df, spans, sup_df, final_load_list)
                st.plotly_chart(fig_ana, use_container_width=True)
                
                st.markdown("#### 🏁 Reaction Forces")
                reac_data = []
                uplift_warning = False
                
                for r_id, val in reactions.items():
                    val_disp = round(val / 1000.0, 2) # แสดงผลเป็น kN
                    status_text = "Compression (OK)"
                    if val < -1e-3: 
                         status_text = "⚠️ UPLIFT (แรงยก!)"
                         uplift_warning = True
                    
                    reac_data.append({
                        "Support Node": f"Node {r_id}",
                        "Reaction": val_disp,
                        "Unit": "kN / kNm",
                        "Status": status_text
                    })
                
                st.dataframe(pd.DataFrame(reac_data), use_container_width=True, hide_index=True)
                if uplift_warning:
                    st.warning("⚠️ **Warning:** Found Uplift forces! Ensure supports are anchored properly.")
                
            with t2:
                st.subheader("Reinforcement Detailing")
                fig_long = section_plotter.plot_longitudinal_section(spans, sup_df, design_res, params['h'], 40)
                st.pyplot(fig_long)
                
                st.divider()
                cols = st.columns(len(spans))
                for i, c in enumerate(cols):
                    d = design_res[i]
                    with c:
                        st.markdown(f"**Span {i+1}**")
                        fig_sec = section_plotter.plot_section(
                            params['b'], params['h'], 40, 16,
                            d['neg']['n'], d['pos']['n'], 
                            d['shear_stirrups'], 24, 400
                        )
                        st.pyplot(fig_sec)
                        if d['shear_status'] == 'Fail':
                            st.error("Shear: Fail")
                        else:
                            st.success(f"Shear: {d['shear_status']}")

                st.divider()
                st.markdown("### 📋 Bill of Quantities & Bar Schedule")
                
                bbs_list = rc_design.generate_bbs(design_res, spans, params['b'], params['h'], 40)
                vol_conc, w_steel = rc_design.get_boq(spans, params['b'], params['h'], bbs_list)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Concrete Volume", f"{vol_conc:.2f} m³")
                m2.metric("Total Steel Weight", f"{w_steel:.2f} kg")
                ratio = w_steel / vol_conc if vol_conc > 0 else 0
                m3.metric("Steel Ratio", f"{ratio:.1f} kg/m³")
                
                st.write("**Bar Bending Schedule (BBS)**")
                if bbs_list:
                    st.dataframe(
                        pd.DataFrame(bbs_list),
                        column_config={
                            "No. of Bars": st.column_config.NumberColumn(format="%d"),
                            "Length (m)": st.column_config.NumberColumn(format="%.2f"),
                            "Total Wt (kg)": st.column_config.NumberColumn(format="%.2f"),
                        },
                        use_container_width=True, hide_index=True
                    )

            with t3:
                st.subheader("📝 Detailed Calculation Basis")
                
                # แสดงการคำนวณ Self-weight ให้ผู้ใช้ตรวจสอบ
                st.info(f"""
                **1. Self-weight Analysis (Dead Load):**
                * Section: {params['b']} m (W) x {params['h']} m (H)
                * Concrete Density: 24.0 kN/m³
                * Calculation: {params['b']} x {params['h']} x 24.0 = **{sw_kn_m:.2f} kN/m**
                
                **2. Load Combinations:**
                * Used Factor (Gamma): {factor} (Applied to both DL and LL for this version)
                """)
                
                st.write("### 3. Design Summary Table")
                report_data = []
                for idx, res in enumerate(design_res):
                    report_data.append({
                        "Span": idx+1,
                        "Top Bars": f"{res['neg']['n']}-DB16",
                        "Bot Bars": f"{res['pos']['n']}-DB16",
                        "Stirrups": res['shear_stirrups'],
                        "Moment Capacity (+)": f"{res['pos']['capacity']:.2f} kNm",
                        "Moment Capacity (-)": f"{res['neg']['capacity']:.2f} kNm",
                        "Crack Check": "OK" if res['pos']['crack_ok'] else "⚠️ Spacing too wide",
                        "Note": res['pos']['note']
                    })
                st.dataframe(pd.DataFrame(report_data), use_container_width=True, hide_index=True)
