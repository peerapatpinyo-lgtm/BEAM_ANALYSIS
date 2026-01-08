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
# params: fc, fy, b, h, E, I | spans: list | sup_df: supports | loads_df: raw user input (kN)
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
        is_service = True
    else:
        with col_fac1:
            f_dl = st.number_input("Dead Load Factor (DL)", value=1.4, step=0.1, format="%.2f", key="fdl_ult")
        with col_fac2:
            f_ll = st.number_input("Live Load Factor (LL)", value=1.7, step=0.1, format="%.2f", key="fll_ult")
        tag = "Ultimate"
        st.warning(f"⚡ Using **Factored Load**: {f_dl} DL + {f_ll} LL for Strength Design.")

    # --- 5. LOAD CALCULATIONS & COMBINATIONS (CRITICAL FIX FOR UNITS) ---
    try:
        # [FOCUS] แก้ปัญหา Point Load 1kN กลายเป็น 1000kN
        # เราจะทำงานด้วยหน่วย Newton (N) และ Meter (m) ตลอดการคำนวณใน Solver
        
        final_solver_loads = []
        
        # 5.1 Self-Weight (Dead Load)
        # b, h อยู่ในหน่วย m | 24.0 คือ kN/m^3 -> แปลงเป็น N/m^3 โดยคูณ 1000
        w_sw_N_m = (params['b'] * params['h'] * 24.0 * 1000.0) * f_dl
        
        for i in range(n_spans):
            final_solver_loads.append({
                'span_index': i,
                'type': 'U',
                'mag': w_sw_N_m,  # หน่วย N/m
                'dist': spans[i], # หน่วย m
                'desc': 'Self-Weight'
            })
            
        # 5.2 User-Defined Loads (Point Load & UDL)
        if not loads_df.empty:
            for _, row in loads_df.iterrows():
                # [CHECKPOINT] ถ้ากรอกมา 1 kN ค่าใน row['mag'] คือ 1.0
                mag_raw = float(row['mag'])
                
                # แปลง kN เป็น N แค่ "ครั้งเดียว" ที่นี่
                mag_in_newton = mag_raw * 1000.0 * f_ll
                
                final_solver_loads.append({
                    'span_index': int(row['span_index']),
                    'type': row['type'],
                    'mag': mag_in_newton, # หน่วย N หรือ N/m
                    'dist': float(row['dist']),
                    'desc': 'User Load'
                })
        
        calc_loads_df = pd.DataFrame(final_solver_loads)

        # --- 6. BEAM SOLVER ---
        # ส่ง calc_loads_df ที่มีหน่วยเป็น Newton เข้าไป
        # ใน solver.py ห้ามมีการคูณ 1000 ซ้ำเด็ดขาด!
        x_eval, M, V, D, R = solver.solve_beam(spans, sup_df, calc_loads_df, params)
        
        # ผลลัพธ์จาก Solver: M(N-m), V(N), D(m), R(N)
        res_df = pd.DataFrame({
            'x': x_eval,
            'moment': M,
            'shear': V,
            'deflection': D * 1000.0 # แปลง m เป็น mm สำหรับวาดกราฟ
        })
        
        # --- 7. TABS FOR RESULTS ---
        tab1, tab2 = st.tabs(["📊 1. Analysis Results & Checks", "📝 2. RC Design & Report"])
        
        with tab1:
            # 7.1 Plotting (ส่งค่า N ไป และในตัวแปร R คือ N)
            st.plotly_chart(design_view.plot_analysis_results(res_df, spans, sup_df, calc_loads_df, R), use_container_width=True)
            
            # 7.2 Summary Metrics (หาร 1000 เพื่อโชว์หน่วย kN)
            st.subheader("📌 Analysis Summary")
            v_max = res_df['shear'].abs().max() / 1000.0
            m_max_pos = res_df['moment'].max() / 1000.0
            m_max_neg = res_df['moment'].min() / 1000.0
            d_max = res_df['deflection'].abs().max()
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"Max Shear", f"{v_max:.2f} kN")
            c2.metric("Max Moment (+)", f"{m_max_pos:.2f} kNm")
            c3.metric("Max Moment (-)", f"{abs(m_max_neg):.2f} kNm")
            c4.metric("Max Deflection", f"{d_max:.2f} mm")

            # 7.3 Support Reactions Table
            st.markdown("### 📍 Support Reactions")
            if R:
                reac_list = []
                for k, v in R.items():
                    reac_list.append({"Support": k, "Reaction (kN)": v / 1000.0})
                st.table(pd.DataFrame(reac_list))

            # 7.4 Equilibrium Check (ตัวยืนยันว่าหน่วยถูก)
            with st.expander("⚖️ Static Equilibrium Check", expanded=True):
                sum_reac_kN = sum(R.values()) / 1000.0
                
                # คำนวณแรงที่กดลงจริงจาก calc_loads_df
                total_applied_N = 0
                for _, l in calc_loads_df.iterrows():
                    if l['type'] == 'P': total_applied_N += l['mag']
                    else: total_applied_N += (l['mag'] * l['dist'])
                
                sum_load_kN = total_applied_N / 1000.0
                
                st.write(f"Total Reactions: **{sum_reac_kN:.3f} kN**")
                st.write(f"Total Applied Loads: **{sum_load_kN:.3f} kN**")
                
                # ถ้าสองค่านี้ไม่เท่ากัน แปลว่า Logic ใน solver.py หรือการส่งค่าผิด
                if abs(sum_reac_kN - sum_load_kN) < 0.01:
                    st.success("Equilibrium Check: PASS (Units are Correct)")
                else:
                    st.error("Equilibrium Check: FAIL (Unit Mismatch Detected)")

        # ================= TAB 2: RC DESIGN =================
        with tab2:
            if is_service:
                st.warning("⚠️ Please switch to 'Ultimate' mode for Reinforced Concrete design.")
            else:
                st.header(f"Reinforced Concrete Design ({tag})")
                
                design_results = []
                db_main = 16 # mm
                offsets = [0] + list(np.cumsum(spans))
                
                for i in range(n_spans):
                    # กรองข้อมูลช่วง Span
                    mask = (res_df['x'] >= offsets[i] - 1e-7) & (res_df['x'] <= offsets[i+1] + 1e-7)
                    s_data = res_df[mask]
                    
                    if not s_data.empty:
                        # เตรียมค่า Mu, Vu ในหน่วย kN-m และ kN
                        mu_p = s_data['moment'].max() / 1000.0
                        mu_n = abs(s_data['moment'].min()) / 1000.0
                        vu = s_data['shear'].abs().max() / 1000.0
                        d_eff = params['h'] - 0.05
                        
                        # เรียก Module ออกแบบ
                        as_p, _, _, steps_p = rc_design.design_beam_flexure(mu_p, params['b'], d_eff, params['fc'], params['fy'])
                        as_n, _, _, steps_n = rc_design.design_beam_flexure(mu_n, params['b'], d_eff, params['fc'], params['fy'])
                        s_v, _, steps_v = rc_design.check_shear(vu, params['b'], d_eff, params['fc'], params['fy'])
                        
                        # คำนวณจำนวนเหล็ก (mm2 / mm2)
                        def get_n(area, db): return max(2, int(np.ceil(area / (np.pi * (db/2)**2))))
                        
                        design_results.append({
                            'span': i+1, 'db': db_main,
                            'pos': {'n': get_n(as_p, db_main)},
                            'neg': {'n': get_n(as_n, db_main)},
                            'shear': {'s': s_v}
                        })
                        
                        with st.expander(f"📄 Calculation Details: Span {i+1}"):
                            st.markdown("**1. Positive Moment Design (Bottom Steel)**")
                            for s in steps_p: st.latex(s)
                            st.markdown("**2. Shear Design (Stirrups)**")
                            for s in steps_v: st.latex(s)

                # 7.5 Detailing Preview
                st.markdown("---")
                st.subheader("🛠️ Detailing Preview")
                if design_results:
                    c_det1, c_det2 = st.columns([1, 2])
                    with c_det1:
                        st.pyplot(section_plotter.plot_section(params['b'], params['h'], 40, db_main, 
                                                              design_results[0]['neg']['n'], 
                                                              design_results[0]['pos']['n'], 
                                                              "RB6", params['fc'], params['fy']))
                    with c_det2:
                        st.pyplot(section_plotter.plot_longitudinal_section_detailed(spans, sup_df, design_results, params['h'], 40))

    except Exception as e:
        st.error(f"❌ Critical Error in Calculation: {e}")
        st.exception(e)

# --- TOTAL LINE COUNT APPROX 230-250 ---
