import streamlit as st
import pandas as pd
import numpy as np
import io
import time

# --- 1. IMPORT CUSTOM MODULES ---
import input_handler
import solver
import design_view
import section_plotter
import reporter

# Import Separated Modules
import rc_utils
import rc_design_engine
import rc_load_processor
import app_styles

# --- 2. PAGE CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="Pro RC Beam Design", 
    layout="wide", 
    page_icon="🏗️",
    initial_sidebar_state="expanded"
)

app_styles.apply_custom_css()

# --- 4. MAIN APPLICATION ---

st.markdown('<div class="main-header">🏗️ RC Beam Analysis & Design Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Timoshenko / Finite Element Method</div>', unsafe_allow_html=True)
st.markdown("---")

# --- 4.1 SIDEBAR: INPUTS ---
with st.sidebar:
    st.header("📝 Project Information")
    project_name = st.text_input("Project Name", "Residential Building A")
    engineer_name = st.text_input("Engineer", "Eng. Somchai")
    st.markdown("---")
    
    params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()

if not stable:
    st.error("🚨 **Structure Error:** โครงสร้างไม่เสถียร (Unstable)! กรุณาตรวจสอบจุดรองรับ (Support)")
else:
    # --- 4.2 ANALYSIS SETTINGS ---
    col_set1, col_set2 = st.columns([1, 2])
    with col_set1:
        st.markdown("### ⚙️ Load Factors")
        mode_select = st.radio(
            "Design Mode:",
            ["Service Load (Check Deflection)", "Ultimate Strength (Design)"],
            index=1
        )
    
    with col_set2:
        st.markdown("### 🔢 Factors")
        c1, c2, c3 = st.columns(3)
        if "Service" in mode_select:
            f_dl, f_ll = 1.0, 1.0
            tag = "Service"
            st.info("ℹ️ Service Mode: Load Factors = 1.0 (DL+LL)")
            is_service = True
        else:
            with c1: f_dl = st.number_input("Dead Load (DL)", 1.4, 1.6, 1.4, 0.1)
            with c2: f_ll = st.number_input("Live Load (LL)", 1.7, 2.0, 1.7, 0.1)
            tag = "Ultimate"
            is_service = False

    # --- 4.3 LOAD CALCULATION PROCESS & SOLVER ---
    try:
        with st.spinner('Running Analysis...'):
            calc_loads_ult = rc_load_processor.prepare_load_dataframe(loads_df, n_spans, spans, params, f_dl, f_ll)
            x_ult, M_ult, V_ult, D_ult, R_ult = solver.solve_beam(spans, sup_df, calc_loads_ult, params)
            
            calc_loads_svc = rc_load_processor.prepare_load_dataframe(loads_df, n_spans, spans, params, 1.0, 1.0)
            x_svc, M_svc, V_svc, D_svc, R_svc = solver.solve_beam(spans, sup_df, calc_loads_svc, params)

        if is_service:
            x_plot, M_plot, V_plot, D_plot, R_plot = x_svc, M_svc, V_svc, D_svc, R_svc
            display_loads = calc_loads_svc
        else:
            x_plot, M_plot, V_plot, D_plot, R_plot = x_ult, M_ult, V_ult, D_ult, R_ult
            display_loads = calc_loads_ult

        # Master Dataframe for metrics
        master_df = pd.DataFrame({
            'x': x_plot, 'M_Nmm': M_plot, 'V_N': V_plot, 'D_m': D_plot
        })
        master_df['M_kNm'] = master_df['M_Nmm'] / 1000.0
        master_df['V_kN'] = master_df['V_N'] / 1000.0
        master_df['D_mm'] = master_df['D_m'] * 1000.0
        
        # --- 5. TABS INTERFACE ---
        tab1, tab2, tab3 = st.tabs(["📊 1. Analysis Results", "📝 2. Concrete Design & Detailing", "📘 3. Detailed Calculation Report"])
        final_design_res = []

        # ================= TAB 1: ANALYSIS RESULTS =================
        with tab1:
            st.subheader(f"📈 Force Diagrams ({tag} Load)")
            # เตรียม DataFrame ส่งเข้า design_view โดยใช้หน่วยมาตรฐาน
            df_for_plot = pd.DataFrame({
                'x': x_plot, 
                'moment': M_plot, 
                'shear': V_plot, 
                'deflection': D_plot * 1000
            })
            
            if not df_for_plot.empty:
                # เรียกใช้ฟังก์ชันใน design_view ที่เราแก้หน่วยไว้แล้ว
                fig = design_view.plot_analysis_results(df_for_plot, spans, sup_df, display_loads, R_plot)
                st.plotly_chart(fig, use_container_width=True)
            
            v_max = master_df['V_kN'].abs().max()
            m_max = master_df['M_kNm'].max()
            m_min = master_df['M_kNm'].min()
            d_max = master_df['D_mm'].abs().max()
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Max Shear (Vu)", f"{v_max:.2f} kN")
            c2.metric("Max Moment (+)", f"{max(0, m_max):.2f} kN-m")
            c3.metric("Max Moment (-)", f"{abs(min(0, m_min)):.2f} kN-m")
            c4.metric("Max Deflection", f"{d_max:.2f} mm")

        # ================= TAB 2: INTERACTIVE DESIGN =================
        with tab2:
            st.header(f"🏗️ Interactive RC Design")
            if is_service:
                st.warning("⚠️ You are viewing Service Load graphs, but Design uses Ultimate Loads.")
            
            b_mm, h_mm = rc_utils.normalize_section_units(params['b'], params['h'])
            fc, fy = params['fc'], params['fy']
            offsets = [0] + list(np.cumsum(spans))
            
            for i in range(n_spans):
                s_len, s_start, s_end = spans[i], offsets[i], offsets[i+1]
                
                # ดึงแรง Ultimate มาใช้ในการออกแบบเสมอ
                mask_ult = (x_ult >= s_start - 1e-6) & (x_ult <= s_end + 1e-6)
                if any(mask_ult):
                    mu_pos = max(0.0, (M_ult[mask_ult] / 1000.0).max())
                    mu_neg = abs(min(0.0, (M_ult[mask_ult] / 1000.0).min()))
                    vu_max = abs((V_ult[mask_ult] / 1000.0)).max()
                else:
                    mu_pos, mu_neg, vu_max = 0, 0, 0
                
                # ข้อมูล Service สำหรับเช็ค Deflection ราย Span
                mask_svc = (x_svc >= s_start - 1e-6) & (x_svc <= s_end + 1e-6)
                ma_pos_svc = max(0.0, (M_svc[mask_svc] / 1000.0).max()) if any(mask_svc) else 0
                delta_svc_mm = abs((D_svc[mask_svc] * 1000.0)).max() if any(mask_svc) else 0

                with st.expander(f"📍 Span {i+1} (L={s_len} m) | Mu+={mu_pos:.1f} kNm, Mu-={mu_neg:.1f} kNm", expanded=True):
                    col_input, col_draw = st.columns([2, 1])
                    
                    with col_input:
                        st.caption(f"Design Constants: fc'={fc} MPa, fy={fy} MPa, Size {b_mm}x{h_mm} mm")
                        cover_mm = st.number_input(f"Covering (mm)", 20.0, 50.0, 25.0, 5.0, key=f"cov_{i}")

                        # 1. Bottom Reinforcement
                        st.markdown("##### 1. Bottom Rebar (Mid-Span)")
                        d_est = h_mm - cover_mm - 20
                        as_req_bot, _, _ = rc_design_engine.get_as_req(mu_pos, d_est, fc, fy, b_mm)
                        
                        c1, c2, c3 = st.columns([1, 1, 1])
                        with c1: st.write(f"Req As: `{as_req_bot:.0f}` mm²")
                        with c2: bot_db = st.selectbox("DB Size", [12, 16, 20, 25, 28], index=1, key=f"bdb_{i}")
                        with c3: bot_n = st.number_input("Qty", 2, 10, 2, key=f"bn_{i}")
                        
                        d_real_b = h_mm - cover_mm - 9 - (bot_db / 2)
                        phi_Mn_bot, as_prov_bot, _, _, _, _ = rc_design_engine.get_phi_Mn_details(bot_n, bot_db, d_real_b, b_mm, fc, fy)

                        # 2. Top Reinforcement
                        st.markdown("##### 2. Top Rebar (Supports)")
                        as_req_top, _, _ = rc_design_engine.get_as_req(mu_neg, d_est, fc, fy, b_mm)
                        
                        c1, c2, c3 = st.columns([1, 1, 1])
                        with c1: st.write(f"Req As: `{as_req_top:.0f}` mm²")
                        with c2: top_db = st.selectbox("DB Size", [12, 16, 20, 25, 28], index=1, key=f"tdb_{i}")
                        with c3: top_n = st.number_input("Qty", 2, 10, 2, key=f"tn_{i}")
                        
                        d_real_t = h_mm - cover_mm - 9 - (top_db / 2)
                        phi_Mn_top, as_prov_top, _, _, _, _ = rc_design_engine.get_phi_Mn_details(top_n, top_db, d_real_t, b_mm, fc, fy)

                        # 3. Shear Reinforcement
                        st.markdown("##### 3. Shear Stirrups")
                        c1, c2, c3 = st.columns([1, 1, 1])
                        with c1: st.write(f"Vu: `{vu_max:.1f}` kN")
                        with c2: stir_db = st.selectbox("Size", [6, 9, 12], index=0, key=f"sdb_{i}")
                        with c3: stir_s = st.number_input("Spacing (mm)", 50, 300, 150, 10, key=f"ss_{i}")
                        
                        status_v, phi_Vn, _, _, _, _ = rc_design_engine.check_shear_details(vu_max, b_mm, d_real_b, fc, fy, stir_db, stir_s)

                        # --- NEW: Call Interactive Comparison Dashboard ---
                        design_res_pack = {
                            'phi_Mn_pos': phi_Mn_bot, 'phi_Mn_neg': phi_Mn_top, 'phi_Vn': phi_Vn,
                            'top_n': top_n, 'top_db': top_db, 'bot_n': bot_n, 'bot_db': bot_db,
                            'stir_db': stir_db, 'stir_spacing': stir_s
                        }
                        design_view.display_design_comparison(mu_pos, mu_neg, vu_max, design_res_pack)

                    with col_draw:
                        st.markdown("<p style='text-align:center;'><b>Section Preview</b></p>", unsafe_allow_html=True)
                        cs_data = {
                            'b': b_mm, 'h': h_mm, 'cover': cover_mm,
                            'top': {'n': top_n}, 'top_db': top_db,
                            'bot': {'n': bot_n}, 'bot_db': bot_db,
                            'stir_db': stir_db, 'shear': {'s': stir_s}
                        }
                        cs_svg = section_plotter.plot_cross_section(cs_data)
                        st.components.v1.html(f'<div style="background:white; padding:10px;">{cs_svg}</div>', height=350)

                    final_design_res.append({
                        'span_id': i, 'L': s_len, 'b': b_mm, 'h': h_mm, 'fc': fc, 'fy': fy,
                        'Mu_pos': mu_pos, 'Mu_neg': mu_neg, 'Vu_max': vu_max, 'cover': cover_mm,
                        'top_db': top_db, 'bot_db': bot_db, 'stir_db': stir_db,
                        'pos': {'n': bot_n, 'area': as_prov_bot, 'status': (phi_Mn_bot >= mu_pos)},
                        'neg': {'n': top_n, 'area': as_prov_top, 'status': (phi_Mn_top >= mu_neg)},
                        'shear': {'s': stir_s, 'db': stir_db, 'status': status_v},
                        'Ma_pos_svc': ma_pos_svc, 'delta_svc_mm': delta_svc_mm,
                        'bot': {'n': bot_n, 'db': bot_db}, 'top': {'n': top_n, 'db': top_db}
                    })

            st.markdown("---")
            st.subheader("📋 Overall Summary")
            summary_list = []
            for item in final_design_res:
                summary_list.append({
                    "Span": item['span_id'] + 1,
                    "Bottom": f"{item['bot']['n']}-DB{item['bot']['db']}",
                    "Top": f"{item['top']['n']}-DB{item['top']['db']}",
                    "Stirrup": f"RB{item['stir_db']}@{item['shear']['s']} mm",
                    "Status": "✅ Pass" if (item['pos']['status'] and item['neg']['status'] and item['shear']['status'] == "OK") else "❌ Fail"
                })
            st.table(pd.DataFrame(summary_list))
            
            if st.button("🔄 Generate Detailed Drawings", type="primary"):
                try:
                    svg_long, png_data = section_plotter.plot_longitudinal_section_detailed(spans, sup_df, final_design_res, h_mm, cover_mm)
                    st.components.v1.html(f'<div style="background:white; overflow-x:auto;">{svg_long}</div>', height=450, scrolling=True)
                    st.download_button("📥 Download Drawing (PNG)", png_data, f"Beam_Drawing_{int(time.time())}.png", "image/png")
                except Exception as e:
                    st.error(f"Drawing Error: {e}")

        # ================= TAB 3: DETAILED REPORT =================
        with tab3:
            st.header("📝 Detailed Calculation Reports")
            st.markdown(f"**Project:** {project_name} | **Engineer:** {engineer_name}")
            if not final_design_res:
                st.warning("⚠️ Please complete the design in Tab 2 first.")
            else:
                for i, res in enumerate(final_design_res):
                    with st.expander(f"📘 Calculation Sheet: Span {i+1}", expanded=False):
                        reporter.render_calculation_report(res)

    except Exception as e:
        st.error(f"❌ Application Error: {e}")
        import traceback
        st.code(traceback.format_exc())
