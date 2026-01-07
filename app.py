import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. IMPORT CUSTOM MODULES ---
import input_handler
import solver
import rc_design
import design_view
import section_plotter

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Beam Analysis & Design Pro", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve UI
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e1e4e8; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏗️ RC Beam Analysis & Design Pro (Timoshenko Theory)")
st.markdown("---")

# --- 3. SIDEBAR INPUTS ---
# ดึงค่าพารามิเตอร์ทั้งหมดจาก sidebar
params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()

if not stable:
    st.error("🚨 Structure is unstable! Please check supports (Ensure R_x, R_y, and M constraints are sufficient).")
else:
    # --- 4. ANALYSIS SETTINGS & LOAD FACTORS ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("📢 Load Combinations")
    mode_select = st.sidebar.radio(
        "Select Analysis Mode:",
        ["Service Load (1.0DL + 1.0LL)", "Ultimate / Custom Factors"],
        index=1
    )
    
    # กำหนด Load Factors ตามโหมดที่เลือก
    if mode_select.startswith("Service"):
        f_dl, f_ll = 1.0, 1.0
        tag = "Service"
    else:
        col_f1, col_f2 = st.sidebar.columns(2)
        f_dl = col_f1.number_input("DL Factor", value=1.4, step=0.1, format="%.2f")
        f_ll = col_f2.number_input("LL Factor", value=1.7, step=0.1, format="%.2f")
        tag = "Ultimate"

    # --- 5. LOAD CALCULATIONS (CRITICAL UNIT FIX) ---
    try:
        # 5.1 Self-Weight Calculation
        # b (m), h (m), Concrete Density 24 kN/m3
        w_sw_base_kN = params['b'] * params['h'] * 24.0  
        w_sw_factored_kN = w_sw_base_kN * f_dl
        
        # เตรียม List สำหรับเก็บข้อมูลโหลด 2 ชุด (เพื่อแก้ปัญหาหน่วยโชว์ผิด)
        solver_input_list = []  # ส่งเข้า Solver (หน่วย Newton)
        display_plot_list = []  # ส่งเข้า Plot/Report (หน่วย kN)
        
        # 5.2 Process Self-Weight (UDL)
        for i in range(n_spans):
            # Solver (N/m)
            solver_input_list.append({
                'span_index': i, 'type': 'U', 
                'mag': w_sw_factored_kN * 1000.0, 
                'dist': spans[i]
            })
            # Display (kN/m)
            display_plot_list.append({
                'span_index': i, 'type': 'U', 
                'mag': w_sw_factored_kN, 
                'dist': spans[i], 'desc': 'Self-Weight'
            })
            
        # 5.3 Process User-Defined Loads (Point Load 3.6 kN)
        if not loads_df.empty:
            for _, row in loads_df.iterrows():
                s_idx = int(row['span_index'])
                if s_idx >= n_spans: continue
                
                l_type = row['type']
                mag_raw_kN = float(row['mag']) # รับค่า 3.6 จาก UI
                factored_kN = mag_raw_kN * f_ll
                dist = float(row['dist'])
                
                # เข้า Solver (N หรือ N/m)
                solver_input_list.append({
                    'span_index': s_idx, 'type': l_type, 
                    'mag': factored_kN * 1000.0, 
                    'dist': dist
                })
                # เข้า Plot (kN หรือ kN/m)
                display_plot_list.append({
                    'span_index': s_idx, 'type': l_type, 
                    'mag': factored_kN, 
                    'dist': dist, 'desc': 'User Load'
                })

        calc_loads_df = pd.DataFrame(solver_input_list)
        plot_loads_df = pd.DataFrame(display_plot_list)

        # --- 6. BEAM SOLVER EXECUTION ---
        # Solver จะคำนวณในหน่วย SI (N, m)
        x_eval, M_raw, V_raw, D_raw, R_raw = solver.solve_beam(spans, sup_df, calc_loads_df, params)
        
        # แปลงผลลัพธ์จาก Newton เป็น kN ทันทีเพื่อความถูกต้องของ Report
        res_df = pd.DataFrame({
            'x': x_eval,
            'moment': M_raw / 1000.0,    # kNm
            'shear': V_raw / 1000.0,     # kN
            'deflection': D_raw * 1000.0 # mm
        })
        
        # Support Reactions ในหน่วย kN
        R_kN = {k: v / 1000.0 for k, v in R_raw.items()}

        # --- 7. TABS INTERFACE ---
        tab1, tab2, tab3 = st.tabs(["📊 Analysis Results", "📝 Structural Design", "📋 Load Report"])

        # ================= TAB 1: ANALYSIS =================
        with tab1:
            st.subheader(f"Analysis Diagrams ({tag})")
            # ใช้ plot_loads_df (หน่วย kN) เพื่อให้ Label ในกราฟโชว์ 3.6 ไม่ใช่ 3600
            st.plotly_chart(design_view.plot_analysis_results(res_df, spans, sup_df, plot_loads_df, R_kN), use_container_width=True)
            
            # สรุปค่าสูงสุด
            c1, c2, c3, c4 = st.columns(4)
            v_max = res_df['shear'].abs().max()
            m_max_p = res_df['moment'].max()
            m_max_n = res_df['moment'].min()
            d_max = res_df['deflection'].abs().max()
            
            c1.metric("Max Shear", f"{v_max:.2f} kN")
            c2.metric("Max Moment (+)", f"{m_max_p:.2f} kNm")
            c3.metric("Max Moment (-)", f"{abs(m_max_n):.2f} kNm")
            c4.metric("Max Deflection", f"{d_max:.2f} mm")

            st.markdown("### 📍 Support Reactions")
            reac_list = [{"Support Node": int(str(k).replace('R', '')), "Reaction (kN)": f"{v:.2f}"} for k, v in R_kN.items()]
            st.dataframe(pd.DataFrame(reac_list), use_container_width=True, hide_index=True)

        # ================= TAB 2: DESIGN =================
        with tab2:
            st.header(f"Reinforced Concrete Design (Code: ACI/EIT)")
            if is_service:
                st.warning("⚠️ Warning: Flexural design should be performed under Ultimate Load factors.")

            design_summary = []
            span_start = 0.0
            
            for i, L in enumerate(spans):
                span_end = span_start + L
                # กรองข้อมูลช่วง Span นี้ (ป้องกัน Floating point error)
                span_data = res_df[(res_df['x'] >= span_start - 1e-6) & (res_df['x'] <= span_end + 1e-6)]
                
                if not span_data.empty:
                    Mu_pos = span_data['moment'].max()
                    Mu_neg = abs(span_data['moment'].min())
                    Vu_max = span_data['shear'].abs().max()
                    d_eff = params['h'] - 0.05 # Effective depth (5cm covering)
                    
                    # เรียกใช้ Module การออกแบบ (ส่งค่าหน่วย kN, kNm เข้าไป)
                    As_pos, _, _, steps_p = rc_design.design_beam_flexure(Mu_pos, params['b'], d_eff, params['fc'], params['fy'])
                    As_neg, _, _, steps_n = rc_design.design_beam_flexure(Mu_neg, params['b'], d_eff, params['fc'], params['fy'])
                    s_stirrup, _, steps_s = rc_design.check_shear(Vu_max, params['b'], d_eff, params['fc'], params['fy'])
                    
                    # คำนวณจำนวนเส้นเหล็ก (DB16)
                    bar_area = np.pi * (0.016**2) / 4 * 1e6 # mm2
                    n_pos = max(2, int(np.ceil((As_pos * 1e6) / bar_area)))
                    n_neg = max(2, int(np.ceil((As_neg * 1e6) / bar_area)))

                    design_summary.append({
                        'span': i+1, 'pos': {'n': n_pos, 'as': As_pos}, 
                        'neg': {'n': n_neg, 'as': As_neg}, 'shear': {'s': s_stirrup}
                    })

                    with st.expander(f"🔍 Span {i+1} Design Calculation Detail"):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write("**Bottom Steel (Flexure +)**")
                            for s in steps_p: st.latex(s)
                        with col_b:
                            st.write("**Top Steel (Flexure -)**")
                            for s in steps_n: st.latex(s)
                        st.write("**Shear Reinforcement**")
                        for s in steps_s: st.latex(s)

                span_start += L

            st.markdown("---")
            st.subheader("🛠️ Detailing & Section Preview")
            if design_summary:
                col_det1, col_det2 = st.columns([1, 2])
                with col_det1:
                    # วาดหน้าตัด Span แรก
                    fig_sec = section_plotter.plot_section(
                        params['b'], params['h'], 40, 16, 
                        design_summary[0]['neg']['n'], design_summary[0]['pos']['n'], 
                        f"RB6@{design_summary[0]['shear']['s']*1000:.0f} mm",
                        params['fc'], params['fy']
                    )
                    st.pyplot(fig_sec)
                with col_det2:
                    # วาดหน้าตัดตามยาว
                    fig_long = section_plotter.plot_longitudinal_section_detailed(spans, sup_df, design_summary, params['h'], 40)
                    st.pyplot(fig_long)

        # ================= TAB 3: LOAD REPORT =================
        with tab3:
            st.header("🧮 Detailed Load Combination Report")
            
            # 1. Self-weight Report
            st.subheader("1. Self-Weight (Dead Load)")
            sw_data = {
                "Width (b)": f"{params['b']:.2f} m",
                "Height (h)": f"{params['h']:.2f} m",
                "Unit Weight": "24.00 kN/m³",
                "Load Factor (DL)": f"{f_dl:.2f}",
                "Unfactored SW": f"{w_sw_base_kN:.2f} kN/m",
                "Factored SW": f"{w_sw_factored_kN:.2f} kN/m"
            }
            st.table(pd.DataFrame([sw_data]))

            # 2. User Load Report
            st.subheader("2. Applied User Loads (Live Load)")
            if not loads_df.empty:
                report_df = loads_df.copy()
                report_df['Factored Mag'] = report_df['mag'] * f_ll
                report_df['Unit'] = report_df['type'].apply(lambda x: "kN" if x=='P' else "kN/m")
                st.dataframe(report_df, use_container_width=True)
            else:
                st.info("No additional loads applied by user.")

            # 3. Equilibrium Verification
            st.subheader("3. Global Equilibrium Check")
            total_reac = sum(R_kN.values())
            total_applied = w_sw_factored_kN * sum(spans)
            if not loads_df.empty:
                for _, r in loads_df.iterrows():
                    if r['type'] == 'P': total_applied += (r['mag'] * f_ll)
                    else: total_applied += (r['mag'] * f_ll * r['dist'])
            
            ev1, ev2 = st.columns(2)
            ev1.write(f"Sum of Reactions: **{total_reac:.2f} kN**")
            ev2.write(f"Sum of Applied Loads: **{total_applied:.2f} kN**")
            
            if abs(total_reac - total_applied) < 0.1:
                st.success("✅ Static Equilibrium Verified (Error < 0.1 kN)")
            else:
                st.warning(f"⚠️ Balance Difference: {abs(total_reac - total_applied):.4f} kN")

    except Exception as e:
        st.error(f"❌ Critical Error in Calculation: {str(e)}")
        st.exception(e)

# Footer
st.markdown("---")
st.caption("Developed by Gemini RC Beam Engine | Structural Analysis & Design Tool v2.1")
