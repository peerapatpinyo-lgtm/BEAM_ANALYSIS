import streamlit as st
import pandas as pd
import numpy as np

# --- 1. IMPORT CUSTOM MODULES ---
import input_handler
import solver
import rc_design
import design_view
import section_plotter

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Beam Analysis & Design Pro (V.Full)", 
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🏗️ RC Beam Analysis & Design Pro (Standard Report)")
st.markdown("---")

# --- 3. SIDEBAR INPUTS ---
# ดึงค่าพารามิเตอร์ทั้งหมดจาก input_handler.py
params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()

if not stable:
    st.error("🚨 Structure is unstable! Please check supports (Must have at least 3 reaction components or fixed support).")
    st.info("💡 Tip: Ensure you have at least one 'Pin' and one 'Roller', or one 'Fixed' support.")
else:
    # --- 4. ANALYSIS SETTINGS & LOAD FACTORS ---
    st.sidebar.markdown("### ⚙️ Global Analysis Settings")
    mode_select = st.sidebar.radio(
        "Select Analysis Mode:",
        ["Service Load (1.0DL + 1.0LL)", "Ultimate / Custom Factors"],
        horizontal=False
    )
    
    col_fac1, col_fac2 = st.columns(2)
    
    if mode_select.startswith("Service"):
        f_dl, f_ll = 1.0, 1.0
        with col_fac1:
            st.number_input("Dead Load Factor (f_dl)", value=1.0, disabled=True, key="fdl_s")
        with col_fac2:
            st.number_input("Live Load Factor (f_ll)", value=1.0, disabled=True, key="fll_s")
        tag = "Service"
        is_service = True
    else:
        with col_fac1:
            f_dl = st.number_input("Dead Load Factor (f_dl)", value=1.4, step=0.1, format="%.2f", key="fdl_u")
        with col_fac2:
            f_ll = st.number_input("Live Load Factor (f_ll)", value=1.7, step=0.1, format="%.2f", key="fll_u")
        tag = "Ultimate"
        is_service = False
        st.warning(f"⚡ Ultimate Design Mode Enabled: Using {f_dl}DL + {f_ll}LL")

    # --- 5. LOAD PROCESSING & COMBINATIONS ---
    try:
        final_solver_loads = []
        
        # 5.1 Self-Weight Calculation (Fixed Unit Weight = 24 kN/m³)
        # สูตร: w_sw = b * h * 24 * factor
        for i in range(n_spans):
            base_sw_kN_m = params['b'] * params['h'] * 24.0
            sw_mag_N_m = (base_sw_kN_m * 1000.0) * f_dl # แปลงเป็น Newton/m
            final_solver_loads.append({
                'span_index': i,
                'type': 'U',
                'mag': sw_mag_N_m,
                'd_start': 0.0,
                'dist': spans[i],
                'desc': 'Self-Weight'
            })
            
        # 5.2 User Defined Loads (Processing DL/LL from loads_df)
        if not loads_df.empty:
            for _, row in loads_df.iterrows():
                # พิจารณาตัวคูณตามประเภท Case (DL หรือ LL)
                current_factor = f_dl if "DL" in row['case'] else f_ll
                
                # แปลงหน่วย Magnitude จาก kN เป็น N
                mag_N = float(row['mag']) * 1000.0 * current_factor
                
                # จัดรูปแบบข้อมูลเพื่อส่งเข้า Solver
                final_solver_loads.append({
                    'span_index': int(row['span_index']),
                    'type': row['type'],
                    'mag': mag_N,
                    'd_start': float(row['d_start']),
                    'dist': float(row['dist']), # สำหรับ UDL คือช่วงความยาวโหลด
                    'desc': f"User Load ({row['case']})"
                })
        
        calc_loads_df = pd.DataFrame(final_solver_loads)

        # --- 6. CORE BEAM ANALYSIS (TIMOSHENKO SOLVER) ---
        # Solver จะคืนค่า x_eval, Moment (N-m), Shear (N), Deflection (m), Reactions (N)
        x_eval, M, V, D, R = solver.solve_beam(spans, sup_df, calc_loads_df, params)
        
        # รวบรวมผลลัพธ์ลงใน DataFrame สำหรับการ Plot กราฟ
        res_df = pd.DataFrame({
            'x': x_eval,
            'moment': M, 
            'shear': V,  
            'deflection': D * 1000.0 # แปลงหน่วย m เป็น mm
        })
        
        # --- 7. RESULTS DISPLAY (TABS SYSTEM) ---
        tab_analysis, tab_design = st.tabs(["📊 1. Analysis & Calculations", "📝 2. RC Design Report"])
        
        with tab_analysis:
            # 7.1 Plot Analysis Result Diagrams
            st.markdown("### 📈 Analysis Diagrams")
            st.plotly_chart(design_view.plot_analysis_results(res_df, spans, sup_df, calc_loads_df, R), use_container_width=True)
            
            # 7.2 Detailed Calculation Report
            st.markdown("---")
            st.subheader("🧮 Detailed Load Calculation Report")
            
            # Section A: Self-Weight
            with st.expander("🔍 A. Detailed Self-Weight Breakdown", expanded=True):
                st.write("น้ำหนักบรรทุกคงที่จากน้ำหนักตัวเองของคาน (Self-Weight Calculation):")
                st.latex(r"w_{sw} = (b \times h) \times 24.0 \, kN/m^3 \times f_{DL}")
                
                sw_calc_list = []
                for i in range(n_spans):
                    b_val, h_val = params['b'], params['h']
                    unfactored_sw = b_val * h_val * 24.0
                    sw_calc_list.append({
                        "Span": f"Span {i+1}",
                        "Width (m)": f"{b_val:.2f}",
                        "Depth (m)": f"{h_val:.2f}",
                        "Unfactored (kN/m)": f"{unfactored_sw:.3f}",
                        "DL Factor": f"x {f_dl:.2f}",
                        "Factored SW (kN/m)": f"{(unfactored_sw * f_dl):.3f}"
                    })
                st.table(pd.DataFrame(sw_calc_list))

            # Section B: Load Combination
            with st.expander("🔍 B. User Load Combinations (DL/LL)", expanded=True):
                if not loads_df.empty:
                    combo_report = []
                    for _, row in loads_df.iterrows():
                        fac = f_dl if "DL" in row['case'] else f_ll
                        combo_report.append({
                            "Span": row['span_index'] + 1,
                            "Case": row['case'],
                            "Type": "Point (P)" if row['type'] == 'P' else "Uniform (U)",
                            "Magnitude (kN)": f"{row['mag']:.2f}",
                            "Factor": f"x {fac:.2f}",
                            "Design Load": f"{(row['mag'] * fac):.2f}",
                            "Position (m)": f"from {row['d_start']:.2f} to {row['d_end']:.2f}"
                        })
                    st.dataframe(pd.DataFrame(combo_report), use_container_width=True, hide_index=True)
                else:
                    st.info("ไม่มีโหลดเพิ่มเติมจากผู้ใช้ (No user-defined loads).")

            # 7.3 Static Equilibrium Check
            st.markdown("### ✅ Static Equilibrium & Verification")
            with st.container():
                # รวมแรงปฏิกิริยาทั้งหมด
                sum_reactions_kN = sum(R.values()) / 1000.0
                
                # รวมแรงกดทั้งหมด (Applied Loads)
                total_applied_N = 0
                for _, l in calc_loads_df.iterrows():
                    if l['type'] == 'P':
                        total_applied_N += l['mag']
                    else:
                        total_applied_N += (l['mag'] * l['dist'])
                sum_applied_kN = total_applied_N / 1000.0
                
                ec1, ec2, ec3 = st.columns(3)
                ec1.metric("Sum Reactions (ΣRy)", f"{sum_reactions_kN:.3f} kN")
                ec2.metric("Sum Applied Loads (ΣWy)", f"{sum_applied_kN:.3f} kN")
                
                diff_error = abs(sum_reactions_kN - sum_applied_kN)
                if diff_error < 0.05:
                    ec3.success(f"Equilibrium: OK\n(Error: {diff_error:.6f})")
                else:
                    ec3.error(f"Equilibrium: FAIL\n(Diff: {diff_error:.4f})")

        # ================= TAB 2: RC DESIGN REPORT =================
        with tab_design:
            if is_service:
                st.warning("⚠️ โปรดเปลี่ยน Analysis Mode เป็น 'Ultimate' เพื่อทำการคำนวณเหล็กเสริม")
            else:
                st.header(f"Reinforced Concrete Design Results ({tag})")
                
                design_summary = []
                db_main = 16 # กำหนดขนาดเหล็กหลัก 16mm
                offset_x = [0] + list(np.cumsum(spans))
                
                for i in range(n_spans):
                    # กรองข้อมูลช่วงคานนั้นๆ
                    mask = (res_df['x'] >= offset_x[i] - 1e-7) & (res_df['x'] <= offset_x[i+1] + 1e-7)
                    span_res = res_df[mask]
                    
                    if not span_res.empty:
                        mu_pos = span_res['moment'].max() / 1000.0 # kN-m
                        mu_neg = abs(span_res['moment'].min()) / 1000.0 # kN-m
                        vu_max = span_res['shear'].abs().max() / 1000.0 # kN
                        d_eff = params['h'] - 0.05 # assume covering 5cm
                        
                        # คำนวณหน้าตัด (เรียกใช้ rc_design module)
                        as_pos, _, _, steps_p = rc_design.design_beam_flexure(mu_pos, params['b'], d_eff, params['fc'], params['fy'])
                        as_neg, _, _, steps_n = rc_design.design_beam_flexure(mu_neg, params['b'], d_eff, params['fc'], params['fy'])
                        s_stirrup, _, steps_v = rc_design.check_shear(vu_max, params['b'], d_eff, params['fc'], params['fy'])
                        
                        # คำนวณจำนวนเส้นเหล็ก
                        def calc_n_bars(area, db):
                            bar_area = np.pi * (db/2)**2
                            return max(2, int(np.ceil(area / bar_area)))
                        
                        design_summary.append({
                            'span': i+1,
                            'pos_bars': calc_n_bars(as_pos, db_main),
                            'neg_bars': calc_n_bars(as_neg, db_main),
                            'stirrup_spacing': s_stirrup
                        })
                        
                        with st.expander(f"📄 Span {i+1}: Detailed Design Calculation"):
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown("**Flexural Reinforcement (Moment)**")
                                for s in steps_p: st.latex(s)
                            with c2:
                                st.markdown("**Shear Reinforcement (Stirrups)**")
                                for s in steps_v: st.latex(s)
                
                # แสดงผลรูปตัดคานและรายละเอียดเหล็กเสริม
                st.markdown("---")
                st.subheader("🛠️ Detailing & Section Preview")
                col_sec, col_long = st.columns([1, 2])
                with col_sec:
                    # วาดรูปตัดขวาง (Section)
                    fig_s = section_plotter.plot_section(
                        params['b'], params['h'], 40, db_main, 
                        design_summary[0]['neg_bars'], design_summary[0]['pos_bars'], 
                        "RB6", params['fc'], params['fy']
                    )
                    st.pyplot(fig_s)
                with col_long:
                    # วาดรูปตัดตามยาว (Longitudinal Section)
                    fig_l = section_plotter.plot_longitudinal_section_detailed(spans, sup_df, design_summary, params['h'], 40)
                    st.pyplot(fig_l)

    except Exception as e:
        st.error(f"❌ An error occurred during calculation: {e}")
        st.exception(e)

# --- END OF FILE ---
