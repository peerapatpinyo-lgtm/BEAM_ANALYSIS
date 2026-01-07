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
                st.subheader("📊 Analysis Results & Load Summary")
                
                # แบ่งหน้าจอเป็น 2 ฝั่ง: กราฟ (70%) และ รายการโหลด (30%)
                col_graph, col_loads = st.columns([7, 3])
                
                with col_graph:
                    st.markdown("#### 📈 Structural Diagrams")
                    fig_ana = design_view.plot_analysis_results(res_df, spans, sup_df, final_load_list)
                    st.plotly_chart(fig_ana, use_container_width=True)
                
                with col_loads:
                    st.markdown("#### 📥 Applied Loads (kN, m)")
                    
                    # 1. แสดง Self-weight
                    sw_val = params['b'] * params['h'] * 24.0
                    st.info(f"**Self-weight:**\n{sw_val:.2f} kN/m (All Spans)")
                    
                    # 2. แสดง User Loads
                    if st.session_state.load_list:
                        df_temp = pd.DataFrame(st.session_state.load_list)
                        df_temp['mag'] = (df_temp['mag'].astype(float) / 1000.0).round(2)
                        st.dataframe(
                            df_temp[['type', 'span_index', 'mag', 'dist']], 
                            use_container_width=True, hide_index=True
                        )

                    st.divider()

                    # 3. [UPDATED] รายการคำนวณแรงที่ใช้ในการออกแบบ (Design Forces)
                    st.markdown("#### 💎 Design Force Calculation")
                    
                    f_design = 1.4
                    # ดึงค่าดิบจาก Solver (หน่วย N และ N-m)
                    raw_m_max = res_df['moment'].max() / 1000.0
                    raw_m_min = res_df['moment'].min() / 1000.0
                    raw_v_max = res_df['shear'].abs().max() / 1000.0

                    # แสดงรายการคำนวณทีละบรรทัด
                    st.write(f"**1. Max Positive Moment ($M_u^+$):**")
                    st.latex(rf"M_u = {raw_m_max:.2f} \times {f_design} = {raw_m_max * f_design:.2f} \text{{ kNm}}")
                    
                    st.write(f"**2. Max Negative Moment ($M_u^-$):**")
                    st.latex(rf"M_u = {abs(raw_m_min):.2f} \times {f_design} = {abs(raw_m_min) * f_design:.2f} \text{{ kNm}}")
                    
                    st.write(f"**3. Max Design Shear ($V_u$):**")
                    st.latex(rf"V_u = {raw_v_max:.2f} \times {f_design} = {raw_v_max * f_design:.2f} \text{{ kN}}")
                    
                    st.success(f"ใช้ตัวคูณเพิ่มน้ำหนักบรรทุก (Load Factor) = {f_design}")
                
                    st.divider()
                
                # --- [FINAL VERSION] ส่วนแสดงผล Reaction พร้อมรายการคำนวณทุก Node ---
                st.markdown("#### 🏁 รายการคำนวณแรงปฏิกิริยาที่จุดรองรับ (Design Reaction Forces, $R_u$)")
                
                # สร้างข้อมูลสำหรับการแสดงผลรายการคำนวณแบบ Text
                reac_details = []
                f_design = 1.4
                uplift_warning = False

                for r_id, val in reactions.items():
                    val_service = val / 1000.0  # kN
                    val_factored = val_service * f_design
                    
                    if val < -1e-3: uplift_warning = True
                    
                    # บันทึกรูปแบบการคำนวณ: R_u = R_service * 1.4 = Result
                    calc_line = f"Node {r_id}: {val_service:,.2f} kN × {f_design} = **{val_factored:,.2f} kN**"
                    reac_details.append(calc_line)

                # แบ่งส่วนแสดงผล: ตารางสรุป (ซ้าย) และ รายการคำนวณบรรทัดต่อบรรทัด (ขวา)
                col_reac_list, col_reac_math = st.columns([1, 1])

                with col_reac_list:
                    st.write("**ตารางสรุปแรงปฏิกิริยา:**")
                    reac_df_display = pd.DataFrame([
                        {
                            "จุดรองรับ": f"Node {r_id}",
                            "Service R (kN)": round(val / 1000.0, 2),
                            "Design Ru (kN)": round((val / 1000.0) * f_design, 2)
                        } for r_id, val in reactions.items()
                    ])
                    st.dataframe(reac_df_display, use_container_width=True, hide_index=True)

                with col_reac_math:
                    st.write("**สมการการคำนวณ ($R_u = R \times 1.4$):**")
                    for line in reac_details:
                        st.write(line)

                if uplift_warning:
                    st.warning("⚠️ **ตรวจพบแรงยก (Uplift):** ค่าที่เป็นลบหมายถึงแรงดึงขึ้นที่จุดรองรับ")              

                
                with t2:
                # 1. Longitudinal Section (บีบความสูงลง)
                st.markdown("#### 📏 Longitudinal Section")
                fig_long = section_plotter.plot_longitudinal_section(spans, sup_df, design_res, params['h'], 40)
                fig_long.set_size_inches(12, 3) # บีบรูปให้เตี้ยลงเพื่อประหยัดพื้นที่
                st.pyplot(fig_long, use_container_width=True)
                
                st.markdown("---") # ใช้เส้นบางๆ แทน divider หนาๆ
                
                # 2. Cross Sections (จัดกลุ่มให้อยู่ในแถวเดียวกันและเล็กลง)
                st.markdown("#### 🟦 Cross Sections")
                
                # คำนวณจำนวนคอลัมน์ตามจำนวน Span (สูงสุด 4 ต่อแถวเพื่อไม่ให้เล็กเกิน)
                n_cols = min(len(spans), 4)
                cols = st.columns(n_cols)
                
                for i, d in enumerate(design_res):
                    col_idx = i % n_cols
                    with cols[col_idx]:
                        # แสดงหัวข้อแบบกระชับ
                        st.caption(f"**Span {i+1}** ({params['b']}x{params['h']} m)")
                        
                        fig_sec = section_plotter.plot_section(
                            params['b'], params['h'], 40, 16,
                            d['neg']['n'], d['pos']['n'], 
                            d['shear_stirrups'], 24, 400
                        )
                        # ปรับขนาดรูปตัดให้เล็กลงพอดีคอลัมน์
                        fig_sec.set_size_inches(3, 4) 
                        st.pyplot(fig_sec, use_container_width=True)
                        
                        # แสดงสถานะแบบ Compact
                        if d['shear_status'] == 'Fail':
                            st.error("Shear: Fail", icon="❌")
                        else:
                            st.info(f"Shear: {d['shear_status']}")

                # 3. BBS & BOQ (บีบตารางให้เล็กลง)
                st.markdown("---")
                st.markdown("#### 📋 Material Summary")
                
                bbs_list = rc_design.generate_bbs(design_res, spans, params['b'], params['h'], 40)
                vol_conc, w_steel = rc_design.get_boq(spans, params['b'], params['h'], bbs_list)
                
                m1, m2, m3 = st.columns(3)
                # ใช้ Metrics แบบไม่มีพื้นที่ว่างมาก
                m1.metric("Concrete", f"{vol_conc:.2f} m³")
                m2.metric("Steel", f"{w_steel:.2f} kg")
                m3.metric("Ratio", f"{w_steel/vol_conc:.1f} kg/m³")
                
                if bbs_list:
                    with st.expander("ดูตารางเหล็กเสริม (BBS)"):
                        st.dataframe(pd.DataFrame(bbs_list), use_container_width=True, hide_index=True)

            with t3:
                st.subheader("📝 Detailed Calculation Basis")
                st.info(f"""
                **1. Self-weight Analysis (Dead Load):**
                * Section: {params['b']} m (W) x {params['h']} m (H)
                * Concrete Density: 24.0 kN/m³
                * Calculation: {params['b']} x {params['h']} x 24.0 = **{sw_kn_m:.2f} kN/m**
                
                **2. Design Factors:**
                * Gamma ($\gamma$): {factor}
                """)
                
                report_data = []
                for idx, res in enumerate(design_res):
                    report_data.append({
                        "Span": idx+1,
                        "Top Bars": f"{res['neg']['n']}-DB16",
                        "Bot Bars": f"{res['pos']['n']}-DB16",
                        "Stirrups": res['shear_stirrups'],
                        "Capacity (+)": f"{res['pos']['capacity']:.2f} kNm",
                        "Capacity (-)": f"{res['neg']['capacity']:.2f} kNm",
                        "Note": res['pos']['note']
                    })
                st.dataframe(pd.DataFrame(report_data), use_container_width=True, hide_index=True)







