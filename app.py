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
st.set_page_config(page_title="Beam Analysis & Design Pro", layout="wide")
st.title("🏗️ RC Beam Analysis & Design Pro (Timoshenko)")

# --- 3. SIDEBAR INPUTS ---
params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()

if not stable:
    st.error("🚨 Structure is unstable! Please check supports (Must have at least 3 reaction components).")
else:
    # --- 4. ANALYSIS SETTINGS & LOAD FACTORS ---
    st.markdown("### ⚙️ Analysis Settings & Load Factors")
    
    mode_select = st.radio(
        "Select Analysis Mode:",
        ["Service Load (1.0DL + 1.0LL)", "Ultimate / Custom Factors"],
        horizontal=True
    )
    
    col_fac1, col_fac2, _ = st.columns([1, 1, 2])
    
    is_service = False
    if mode_select.startswith("Service"):
        f_dl, f_ll = 1.0, 1.0
        with col_fac1:
            st.number_input("Dead Load Factor (DL)", value=1.0, disabled=True, key="fdl_serv")
        with col_fac2:
            st.number_input("Live Load Factor (LL)", value=1.0, disabled=True, key="fll_serv")
        tag = "Service"
        st.info("ℹ️ Using **Service Load** (Factors = 1.0) for Deflection & Serviceability checks.")
        is_service = True
    else:
        with col_fac1:
            f_dl = st.number_input("Dead Load Factor (DL)", value=1.4, step=0.1, format="%.2f", key="fdl_ult")
        with col_fac2:
            f_ll = st.number_input("Live Load Factor (LL)", value=1.7, step=0.1, format="%.2f", key="fll_ult")
        tag = "Ultimate"
        st.warning(f"⚡ Using **Factored Load**: {f_dl} DL + {f_ll} LL for Strength Design.")

    # --- 5. LOAD CALCULATIONS & COMBINATIONS ---
    try:
        # 5.1 Self-Weight (Unit Weight = 24 kN/m³)
        w_sw_base_kN = params['b'] * params['h'] * 24.0   
        w_sw_factored_kN = w_sw_base_kN * f_dl

        # 5.2 เตรียม List สำหรับเก็บโหลดที่จะส่งให้ Solver (หน่วย Newton ทั้งหมด)
        final_solver_loads = []
        
        # A. เพิ่ม Self-Weight (DL) ลงไปในทุก Span ก่อน
        for i in range(n_spans):
            final_solver_loads.append({
                'span_index': i,
                'type': 'U',
                'mag': w_sw_factored_kN * 1000.0,  # kN/m -> N/m
                'dist': spans[i],
                'desc': 'Self-Weight'
            })
            
        # B. เพิ่ม User-Defined Loads (DL/LL) จาก Sidebar
        if not loads_df.empty:
            for _, row in loads_df.iterrows():
                # ดึงค่า kN มาคูณ Factor และแปลงเป็น Newton
                # สมมติโหลดที่ผู้ใช้กรอกเป็น Live Load (LL) ทั้งหมดตาม Logic เดิมของคุณ
                mag_N = float(row['mag']) * f_ll * 1000.0 
                
                final_solver_loads.append({
                    'span_index': int(row['span_index']),
                    'type': row['type'],
                    'mag': mag_N,
                    'dist': float(row['dist']),
                    'desc': 'User Load'
                })
        
        # สร้าง DataFrame ชุดสุดท้ายที่จะส่งเข้า Solver
        calc_loads_df = pd.DataFrame(final_solver_loads)

        # --- 6. BEAM SOLVER ---
        # ส่ง calc_loads_df ที่เป็นหน่วย Newton และรวมทุกอย่างแล้วเข้าไป
        x_eval, M, V, D, R = solver.solve_beam(spans, sup_df, calc_loads_df, params)
        
        res_df = pd.DataFrame({
            'x': x_eval,
            'moment': M,  # N-m
            'shear': V,   # N
            'deflection': D * 1000 # mm
        })
        
        # --- 7. DISPLAY ---
        with tab1:
            # [จุดสำคัญ] ต้องส่ง calc_loads_df (หน่วย N) เข้าไปวาด 
            # แต่ใน design_view.py ต้องสั่งให้มันหาร 1000 ก่อนโชว์ kN
            fig_analysis = design_view.plot_analysis_results(res_df, spans, sup_df, calc_loads_df, R)
            st.plotly_chart(fig_analysis, use_container_width=True)
  
        with tab1:
            st.plotly_chart(design_view.plot_analysis_results(res_df, spans, sup_df, calc_loads_df, R), use_container_width=True)
            
            st.subheader("📌 Analysis Summary")
            v_max_kN = res_df['shear'].abs().max() / 1000.0
            m_max_pos_kNm = res_df['moment'].max() / 1000.0
            m_max_neg_kNm = res_df['moment'].min() / 1000.0
            d_abs_max_mm = res_df['deflection'].abs().max()
            
            c_res1, c_res2, c_res3, c_res4 = st.columns(4)
            c_res1.metric(f"Max Shear ({tag})", f"{v_max_kN:.2f} kN")
            c_res2.metric("Max Moment (+)", f"{m_max_pos_kNm:.2f} kNm")
            c_res3.metric("Max Moment (-)", f"{abs(m_max_neg_kNm):.2f} kNm")
            c_res4.metric("Max Deflection", f"{d_abs_max_mm:.2f} mm")

            st.markdown("### 📍 Support Reactions")
            if R:
                reaction_data = [{"Node": int(str(k).replace('R', '')), "Reaction (kN)": v/1000.0} for k, v in R.items()]
                df_reac = pd.DataFrame(reaction_data).sort_values(by="Node")
                st.dataframe(df_reac.style.format({"Reaction (kN)": "{:.2f}"}), use_container_width=True, hide_index=True)

            st.markdown("---")

            with st.expander("🧮 Detailed Load Calculation Report", expanded=True):
                st.markdown("#### A. Self-Weight Calculation (Dead Load)")
                sw_report = []
                for i in range(n_spans):
                    sw_report.append({
                        "Span": i+1,
                        "Dimensions": f"{params['b']}m x {params['h']}m",
                        "Formula": f"b*h * 24 kN/m³ * {f_dl}",
                        "Factored Result": f"{w_sw_factored_kN:.2f} kN/m"
                    })
                st.table(pd.DataFrame(sw_report))

                st.markdown("#### B. Load Combination Breakdown")
                combo_report = []
                for i in range(n_spans):
                    combo_report.append({
                        "Span": i+1, "Type": "Self-Weight (DL)",
                        "Unfactored": f"{w_sw_base_kN:.2f} kN/m", "Factor": f"x{f_dl}", "Factored": f"{w_sw_factored_kN:.2f} kN/m"
                    })
                if not loads_df.empty:
                    for _, row in loads_df.iterrows():
                        combo_report.append({
                            "Span": int(row['span_index'])+1,
                            "Type": "Point (LL)" if row['type'] == 'P' else "Uniform (LL)",
                            "Unfactored": f"{float(row['mag']):.2f} kN(/m)",
                            "Factor": f"x{f_ll}", "Factored": f"{float(row['mag'])*f_ll:.2f} kN(/m)"
                        })
                st.table(pd.DataFrame(combo_report))

            with st.expander("✅ Equilibrium & Deflection Checks", expanded=True):
                ec1, ec2 = st.columns(2)
                with ec1:
                    st.markdown("**Static Equilibrium ($\Sigma F_y = 0$)**")
                    sum_R_kN = sum(R.values()) / 1000.0
                    total_sw_kN = w_sw_factored_kN * sum(spans)
                    total_user_kN = 0
                    if not loads_df.empty:
                        for _, r in loads_df.iterrows():
                            if r['type'] == 'P': total_user_kN += (float(r['mag']) * f_ll)
                            else: total_user_kN += (float(r['mag']) * f_ll * float(r['dist']))
                    total_applied_kN = total_sw_kN + total_user_kN
                    st.write(f"Total Reactions: **{sum_R_kN:.2f} kN**")
                    st.write(f"Total Applied Loads: **{total_applied_kN:.2f} kN**")
                    if abs(sum_R_kN - total_applied_kN) < 1.0: st.success("Balance Check: PASS")
                    else: st.warning(f"Balance Diff: {abs(sum_R_kN - total_applied_kN):.2f} kN")
                
                with ec2:
                    st.markdown("**Deflection Limit Check**")
                    limit_mm = (max(spans) * 1000) / 240.0
                    st.write(f"Max Deflection: {d_abs_max_mm:.2f} mm")
                    st.write(f"Allowable (L/240): {limit_mm:.2f} mm")
                    if d_abs_max_mm <= limit_mm: st.success("Deflection: PASS")
                    else: st.error("Deflection: FAIL")

        # ================= TAB 2: RC DESIGN =================
        with tab2:
            if is_service:
                st.warning("⚠️ **Warning:** Strength Design requires 'Ultimate Load' factors. Switch mode to proceed.")
            
            st.header(f"Reinforced Concrete Design ({tag})")
            
            design_res = []
            span_start = 0
            db_main = 16 # mm

            for i, span_len in enumerate(spans):
                span_end = span_start + span_len
                span_data = res_df[(res_df['x'] >= span_start - 1e-6) & (res_df['x'] <= span_end + 1e-6)]
                
                if not span_data.empty:
                    mu_pos = span_data['moment'].max() / 1000.0
                    mu_neg = abs(span_data['moment'].min()) / 1000.0
                    vu_max = span_data['shear'].abs().max() / 1000.0
                    d_eff = params['h'] - 0.05
                    
                    As_pos, _, _, steps_pos = rc_design.design_beam_flexure(mu_pos, params['b'], d_eff, params['fc'], params['fy'])
                    As_neg, _, _, steps_neg = rc_design.design_beam_flexure(mu_neg, params['b'], d_eff, params['fc'], params['fy'])
                    s_req, _, steps_shear = rc_design.check_shear(vu_max, params['b'], d_eff, params['fc'], params['fy'])
                    
                    # [FIXED UNIT] n_bars: As(mm2) / Area(mm2)
                    def n_bars(As_mm2, db): 
                        area_bar = np.pi * (db/2)**2
                        return max(2, int(np.ceil(As_mm2 / area_bar)))
                    
                    design_res.append({
                        'span': i+1, 'db': db_main,
                        'pos': {'n': n_bars(As_pos, db_main)}, 
                        'neg': {'n': n_bars(As_neg, db_main)}, 
                        'shear': {'s': s_req}
                    })
                    
                    with st.expander(f"📘 Detailed Design: Span {i+1}", expanded=False):
                        st.markdown(f"**Flexural Design (Mu+ = {mu_pos:.2f}, Mu- = {mu_neg:.2f} kNm)**")
                        c1, c2 = st.columns(2)
                        with c1: 
                            st.write("Bottom Steel (Positive Moment):")
                            for s in steps_pos: st.latex(s)
                        with c2: 
                            st.write("Top Steel (Negative Moment):")
                            for s in steps_neg: st.latex(s)
                        st.markdown("**Shear Design**")
                        for s in steps_shear: st.latex(s)

                span_start += span_len

            st.markdown("---")
            st.subheader("🛠️ Detailing Preview")
            if design_res:
                col_det1, col_det2 = st.columns([1, 2])
                with col_det1:
                    # ใช้ข้อมูลจาก Span 1 เป็นตัวอย่างหน้าตัด
                    st.pyplot(section_plotter.plot_section(params['b'], params['h'], 40, db_main, design_res[0]['neg']['n'], design_res[0]['pos']['n'], "RB6", params['fc'], params['fy']))
                with col_det2:
                    st.pyplot(section_plotter.plot_longitudinal_section_detailed(spans, sup_df, design_res, params['h'], 40))

    except Exception as e:
        st.error(f"❌ Calculation Error: {e}")
        st.exception(e)

