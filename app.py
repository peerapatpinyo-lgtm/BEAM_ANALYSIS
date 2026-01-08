import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- 1. IMPORT CUSTOM MODULES ---
import input_handler
import solver
import rc_design
import design_view
import section_plotter

# --- 2. PAGE CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="Ultimate Beam Analysis & RC Design", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better readability
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    .calc-box { background-color: #f0f2f6; border-left: 5px solid #007bff; padding: 10px; margin: 10px 0; }
    </style>
    """, unsafe_all_headers=True)

st.title("🏗️ Professional RC Beam Analysis & Design")
st.markdown("---")

# --- 3. INPUT DATA FETCHING ---
# ข้อมูลทั้งหมดถูกดึงมาจาก input_handler.py ที่เราแก้เรื่อง d_start/d_end แล้ว
params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()

if not stable:
    st.error("🚨 **Structure Stability Error:** ระบบต้องการจุดรองรับอย่างน้อย 3 degree of freedom (เช่น Pin 1, Roller 1 หรือ Fixed 1)")
    st.stop()

# --- 4. LOAD COMBINATION SETTINGS ---
st.subheader("⚙️ 1. Load Combination & Factoring")
with st.container():
    col_set1, col_set2, col_set3 = st.columns([1, 1, 1])
    
    with col_set1:
        st.markdown("**Design Code & Method**")
        method = st.selectbox("Standard:", ["ACI 318-14 (SDM)", "WSD (Service)"])
    
    with col_set2:
        st.markdown("**Dead Load Factor ($f_{DL}$)**")
        f_dl = st.number_input("Input DL Factor", value=1.4, step=0.1, key="fdl_val")
    
    with col_set3:
        st.markdown("**Live Load Factor ($f_{LL}$)**")
        f_ll = st.number_input("Input LL Factor", value=1.7, step=0.1, key="fll_val")

# --- 5. DETAILED LOAD PROCESSING LOGIC ---
try:
    # เตรียม DataFrame สำหรับรายงานการคำนวณ (Calculation Trace)
    load_trace_data = []
    final_solver_loads = []

    # 5.1 การคำนวณน้ำหนักบรรทุกคงที่ (Self-Weight Calculation)
    st.markdown("### 🧮 2. Step-by-Step Load Summation")
    
    # คำนวณ Self-Weight แยกแต่ละช่วง (Span)
    for i in range(n_spans):
        # สูตร: b(m) * h(m) * 24 kN/m3
        base_sw_kN_m = params['b'] * params['h'] * 24.0
        factored_sw = base_sw_kN_m * f_dl
        total_span_sw = factored_sw * spans[i]
        
        # บันทึกลง Trace สำหรับ Report
        load_trace_data.append({
            "Span": i + 1,
            "Load Source": "Self-Weight (Concrete)",
            "Load Case": "DL",
            "Calculation Formula": f"{params['b']} x {params['h']} x 24.0 x {f_dl}",
            "Intensity (kN/m)": f"{factored_sw:.3f}",
            "Total Load on Span (kN)": f"{total_span_sw:.3f}"
        })
        
        # ส่งเข้า Solver (แปลง kN -> N)
        final_solver_loads.append({
            'span_index': i, 'type': 'U', 'mag': factored_sw * 1000.0,
            'd_start': 0.0, 'dist': spans[i], 'desc': 'Self-Weight'
        })

    # 5.2 การคำนวณน้ำหนักบรรทุกจากผู้ใช้ (User Applied Loads)
    if not loads_df.empty:
        for _, row in loads_df.iterrows():
            current_f = f_dl if row['case'] == "DL" else f_ll
            raw_mag = float(row['mag'])
            factored_mag = raw_mag * current_f
            
            # คำนวณน้ำหนักสุทธิลงคาน (Net Force)
            if row['type'] == 'P':
                net_force = factored_mag
                formula = f"{raw_mag} kN x {current_f}"
                intensity = f"{factored_mag:.2f} kN"
            else:
                net_force = factored_mag * row['dist']
                formula = f"{raw_mag} kN/m x {row['dist']}m x {current_f}"
                intensity = f"{factored_mag:.2f} kN/m"

            load_trace_data.append({
                "Span": row['span_index'] + 1,
                "Load Source": f"User Added ({row['type']})",
                "Load Case": row['case'],
                "Calculation Formula": formula,
                "Intensity (kN/m or kN)": intensity,
                "Total Load on Span (kN)": f"{net_force:.3f}"
            })

            # ข้อมูลสำหรับ Solver
            final_solver_loads.append({
                'span_index': int(row['span_index']), 'type': row['type'],
                'mag': factored_mag * 1000.0, 'd_start': row['d_start'],
                'dist': row['dist'], 'desc': f"User {row['case']}"
            })

    # แปลงเป็น DataFrame สำหรับแสดงผล
    calc_loads_df = pd.DataFrame(final_solver_loads)
    report_df = pd.DataFrame(load_trace_data)

    # --- 6. DISPLAY LOAD VERIFICATION TABLE ---
    with st.expander("📝 View Detailed Calculation Trace (How loads are summed)", expanded=True):
        st.table(report_df)
        
        # คำนวณยอดรวมสุทธิ (Check Sum)
        total_w_applied = report_df["Total Load on Span (kN)"].astype(float).sum()
        st.markdown(f"""
        <div class='calc-box'>
        <b>Total Factored Vertical Load ($\Sigma W_{{ult}}$):</b> {total_w_applied:.3f} kN <br>
        <i>นี่คือแรงลัพธ์ทั้งหมดที่จะถูกส่งเข้าวิเคราะห์ใน Stiffness Matrix Solver</i>
        </div>
        """, unsafe_allow_html=True)

    # --- 7. STRUCTURAL ANALYSIS (SOLVER) ---
    # Solver execution (N, m units)
    x_eval, M, V, D, R = solver.solve_beam(spans, sup_df, calc_loads_df, params)
    
    # Prepare results
    res_df = pd.DataFrame({
        'x': x_eval, 'moment': M, 'shear': V, 'deflection': D * 1000.0 
    })

    # --- 8. RESULTS VISUALIZATION TABS ---
    tab_diag, tab_design, tab_equilibrium = st.tabs([
        "📊 Analysis Diagrams", "📝 RC Design Report", "⚖️ Equilibrium Check"
    ])

    with tab_diag:
        st.plotly_chart(design_view.plot_analysis_results(res_df, spans, sup_df, calc_loads_df, R), use_container_width=True)
        
        # Summary Metrics
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Max Moment (+)", f"{res_df['moment'].max()/1000.0:.2f} kNm")
        m_col2.metric("Max Moment (-)", f"{res_df['moment'].min()/1000.0:.2f} kNm")
        m_col3.metric("Max Shear", f"{res_df['shear'].abs().max()/1000.0:.2f} kN")

    with tab_equilibrium:
        st.subheader("⚖️ Static Equilibrium Verification")
        total_reactions = sum(R.values()) / 1000.0
        
        c1, c2, c3 = st.columns(3)
        c1.write("**Total Applied Load (kN)**")
        c1.title(f"{total_w_applied:.3f}")
        
        c2.write("**Total Support Reactions (kN)**")
        c2.title(f"{total_reactions:.3f}")
        
        error_val = abs(total_w_applied - total_reactions)
        c3.write("**Difference (Error)**")
        if error_val < 0.01:
            c3.success(f"{error_val:.6f}")
            st.balloons()
        else:
            c3.error(f"{error_val:.4f}")

    with tab_design:
        st.header("🧱 Reinforced Concrete Design")
        # Logic ออกแบบเหล็กเสริม
        db_main = 16 
        design_results = []
        cum_spans = [0] + list(np.cumsum(spans))

        for i in range(n_spans):
            # Extract span data
            s_mask = (res_df['x'] >= cum_spans[i]-1e-6) & (res_df['x'] <= cum_spans[i+1]+1e-6)
            s_data = res_df[s_mask]
            
            if not s_data.empty:
                mu_p = s_data['moment'].max() / 1000.0
                mu_n = abs(s_data['moment'].min()) / 1000.0
                vu = s_data['shear'].abs().max() / 1000.0
                d_eff = params['h'] - 0.05
                
                # ออกแบบโดยใช้ Module rc_design
                as_p, _, _, stp_p = rc_design.design_beam_flexure(mu_p, params['b'], d_eff, params['fc'], params['fy'])
                as_n, _, _, stp_n = rc_design.design_beam_flexure(mu_n, params['b'], d_eff, params['fc'], params['fy'])
                s_v, _, stp_v = rc_design.check_shear(vu, params['b'], d_eff, params['fc'], params['fy'])
                
                n_p = max(2, int(np.ceil(as_p / (np.pi * (db_main/2)**2))))
                n_n = max(2, int(np.ceil(as_n / (np.pi * (db_main/2)**2))))
                
                design_results.append({'span': i+1, 'n_p': n_p, 'n_n': n_n, 'db': db_main, 's_v': s_v})
                
                with st.expander(f"Span {i+1} Design Calculations", expanded=False):
                    col_p, col_n = st.columns(2)
                    with col_p:
                        st.write("**Bottom Reinforcement (Positive M)**")
                        for s in stp_p: st.latex(s)
                    with col_n:
                        st.write("**Top Reinforcement (Negative M)**")
                        for s in stp_n: st.latex(s)
        
        # กราฟรายละเอียดเหล็กเสริม
        st.markdown("---")
        st.subheader("🎨 Detailing Preview")
        d_col1, d_col2 = st.columns([1, 2])
        with d_col1:
            st.pyplot(section_plotter.plot_section(params['b'], params['h'], 40, db_main, design_results[0]['n_n'], design_results[0]['n_p'], "RB6", params['fc'], params['fy']))
        with d_col2:
            st.pyplot(section_plotter.plot_longitudinal_section_detailed(spans, sup_df, design_results, params['h'], 40))

except Exception as e:
    st.error(f"⚠️ **Analysis Failed:** {str(e)}")
    st.exception(e)

# --- 9. FOOTER & LOGGING ---
st.markdown("---")
st.caption(f"Final Execution Check: {n_spans} Spans | Load Case Combination Completed.")
# Total Lines approximately 250+ including comments and UI spacing
