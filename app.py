# app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import time

# --- IMPORT MODULES ---
import input_handler
import solver
import design_view
import section_plotter
import reporter
import calcs   # <--- Import ไฟล์คำนวณใหม่
import utils   # <--- Import ไฟล์จัดการข้อมูลใหม่

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Pro RC Beam Design", 
    layout="wide", 
    page_icon="🏗️",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1E3A8A; margin-bottom: 0px;}
    .sub-header {font-size: 1.2rem; font-weight: normal; color: #64748B; margin-top: -10px;}
    .stApp {background-color: #F8FAFC;}
    div[data-testid="stMetricValue"] {font-size: 1.5rem; color: #0F172A;}
    .report-box {background-color: #ffffff; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0; font-family: 'Courier New', monospace;}
</style>
""", unsafe_allow_html=True)

# --- MAIN APPLICATION ---
st.markdown('<div class="main-header">🏗️ RC Beam Analysis & Design Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Timoshenko / Finite Element Method</div>', unsafe_allow_html=True)
st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    st.header("📝 Project Information")
    project_name = st.text_input("Project Name", "Residential Building A")
    engineer_name = st.text_input("Engineer", "Eng. Somchai")
    st.markdown("---")
    
    params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()

if not stable:
    st.error("🚨 **Structure Error:** โครงสร้างไม่เสถียร (Unstable)! กรุณาตรวจสอบจุดรองรับ (Support)")
else:
    # --- ANALYSIS SETTINGS ---
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

    # --- SOLVER PROCESS ---
    try:
        with st.spinner('Running Analysis...'):
            # RUN 1: ULTIMATE LOAD
            calc_loads_ult = utils.prepare_load_dataframe(loads_df, n_spans, spans, params, f_dl, f_ll)
            x_ult, M_ult, V_ult, D_ult, R_ult = solver.solve_beam(spans, sup_df, calc_loads_ult, params)
            
            # RUN 2: SERVICE LOAD
            calc_loads_svc = utils.prepare_load_dataframe(loads_df, n_spans, spans, params, 1.0, 1.0)
            x_svc, M_svc, V_svc, D_svc, R_svc = solver.solve_beam(spans, sup_df, calc_loads_svc, params)

        # Plot Data Selection
        if is_service:
            x_plot, M_plot, V_plot, D_plot, R_plot = x_svc, M_svc, V_svc, D_svc, R_svc
            display_loads = calc_loads_svc
        else:
            x_plot, M_plot, V_plot, D_plot, R_plot = x_ult, M_ult, V_ult, D_ult, R_ult
            display_loads = calc_loads_ult

        master_df = pd.DataFrame({
            'x': x_plot, 'M_Nmm': M_plot, 'V_N': V_plot, 'D_m': D_plot
        })
        master_df['M_kNm'] = master_df['M_Nmm'] / 1000.0
        master_df['V_kN'] = master_df['V_N'] / 1000.0
        master_df['D_mm'] = master_df['D_m'] * 1000.0
        
        # --- TABS ---
        tab1, tab2, tab3 = st.tabs(["📊 1. Analysis Results", "📝 2. Concrete Design & Detailing", "📘 3. Detailed Calculation Report"])
        final_design_res = []

        # === TAB 1: ANALYSIS ===
        with tab1:
            st.subheader(f"📈 Force Diagrams ({tag} Load)")
            df_for_plot = pd.DataFrame({'x': x_plot, 'moment': M_plot, 'shear': V_plot, 'deflection': D_plot * 1000})
            if not df_for_plot.empty:
                st.plotly_chart(design_view.plot_analysis_results(df_for_plot, spans, sup_df, display_loads, R_plot), use_container_width=True)
            
            # Metrics
            v_max = master_df['V_kN'].abs().max()
            m_max = master_df['M_kNm'].max()
            m_min = master_df['M_kNm'].min()
            d_max = master_df['D_mm'].abs().max()
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Max Shear", f"{v_max:.2f} kN")
            c2.metric("Max Moment (+)", f"{max(0, m_max):.2f} kNm")
            c3.metric("Max Moment (-)", f"{abs(min(0, m_min)):.2f} kNm")
            c4.metric("Max Deflection", f"{d_max:.2f} mm")

        # === TAB 2: DESIGN ===
        with tab2:
            st.header(f"🏗️ Interactive RC Design")
            if is_service: st.warning("⚠️ Warning: Viewing Service Loads, but Design uses Ultimate Loads.")
            
            # Unit Normalization using calcs module
            b_mm, h_mm = calcs.normalize_section_units(params['b'], params['h'])
            fc, fy = params['fc'], params['fy']
            
            offsets = [0] + list(np.cumsum(spans))
            
            for i in range(n_spans):
                s_len = spans[i]
                s_start, s_end = offsets[i], offsets[i+1]
                
                # Get Forces
                mask_ult = (x_ult >= s_start - 1e-6) & (x_ult <= s_end + 1e-6)
                if any(mask_ult):
                    mu_pos = max(0.0, (M_ult[mask_ult] / 1000.0).max())
                    mu_neg = abs(min(0.0, (M_ult[mask_ult] / 1000.0).min()))
                    vu_max = abs((V_ult[mask_ult] / 1000.0)).max()
                else: mu_pos, mu_neg, vu_max = 0, 0, 0
                
                # Service Forces
                mask_svc = (x_svc >= s_start - 1e-6) & (x_svc <= s_end + 1e-6)
                if any(mask_svc):
                    ma_pos_svc = max(0.0, (M_svc[mask_svc] / 1000.0).max())
                    delta_svc_mm = abs((D_svc[mask_svc] * 1000.0)).max()
                else: ma_pos_svc, delta_svc_mm = 0, 0

                with st.expander(f"📍 **Span {i+1}** (L={s_len} m) | Forces: Mu+={mu_pos:.1f}, Mu-={mu_neg:.1f}, Vu={vu_max:.1f}", expanded=True):
                    c_const, c_cov = st.columns([3, 1])
                    with c_const: st.caption(f"Constants: fc'={fc}, fy={fy}, Size {b_mm:.0f}x{h_mm:.0f} mm")
                    with c_cov: cover_mm = st.number_input(f"Cover (mm)", value=25.0, step=5.0, key=f"cov_{i}")

                    # 1. Bottom Steel
                    st.markdown("##### 1. Bottom Reinforcement (Mid-Span)")
                    d_est = h_mm - cover_mm - 20 
                    as_req_bot, _, _ = calcs.get_as_req(mu_pos, d_est, fc, fy, b_mm)
                    
                    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
                    with c1: st.markdown(f"**Req $A_s$:**\n`{as_req_bot:.0f}` mm²")
                    with c2: bot_db = st.selectbox("DB", [12, 16, 20, 25, 28], index=1, key=f"bdb_{i}")
                    with c3: bot_n = st.number_input("Qty", 2, 10, 2, key=f"bn_{i}")
                    
                    d_real = h_mm - cover_mm - 9 - (bot_db / 2)
                    phi_Mn_bot, as_prov_bot, _, _, _, _ = calcs.get_phi_Mn_details(bot_n, bot_db, d_real, b_mm, fc, fy)
                    pass_b = (phi_Mn_bot >= mu_pos) and (phi_Mn_bot > 0)
                    
                    with c4: 
                        clr_b = "green" if pass_b else "red"
                        msg_b = f"**{phi_Mn_bot:.2f}**" if phi_Mn_bot > 0 else "**FAIL**"
                        st.markdown(f"$\phi M_n$: :{clr_b}[{msg_b}] kNm")

                    # 2. Top Steel
                    st.markdown("##### 2. Top Reinforcement (Supports)")
                    as_req_top, _, _ = calcs.get_as_req(mu_neg, d_est, fc, fy, b_mm)
                    
                    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
                    with c1: st.markdown(f"**Req $A_s$:**\n`{as_req_top:.0f}` mm²")
                    with c2: top_db = st.selectbox("DB", [12, 16, 20, 25, 28], index=1, key=f"tdb_{i}")
                    with c3: top_n = st.number_input("Qty", 2, 10, 2, key=f"tn_{i}")
                    
                    d_top_real = h_mm - cover_mm - 9 - (top_db / 2)
                    phi_Mn_top, as_prov_top, _, _, _, _ = calcs.get_phi_Mn_details(top_n, top_db, d_top_real, b_mm, fc, fy)
                    pass_t = (phi_Mn_top >= mu_neg) and (phi_Mn_top > 0)
                    
                    with c4:
                        clr_t = "green" if pass_t else "red"
                        msg_t = f"**{phi_Mn_top:.2f}**" if phi_Mn_top > 0 else "**FAIL**"
                        st.markdown(f"$\phi M_n$: :{clr_t}[{msg_t}] kNm")

                    # 3. Shear
                    st.markdown("##### 3. Shear Reinforcement")
                    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
                    with c1: st.markdown(f"**$V_u$:** `{vu_max:.2f}` kN")
                    with c2: stir_db = st.selectbox("Stirrup", [6, 9, 12], index=0, key=f"sdb_{i}")
                    with c3: stir_s = st.number_input("Spacing (mm)", 50, 300, 150, 10, key=f"ss_{i}")
                    
                    status_v, phi_Vn, _, _, _, _ = calcs.check_shear_details(vu_max, b_mm, d_real, fc, fy, stir_db, stir_s)
                    
                    with c4:
                        clr_v = "green" if status_v == "OK" else "red"
                        st.markdown(f"$\phi V_n$: :{clr_v}[**{phi_Vn:.1f}**] kN ({status_v})")
                    
                    # Store Results
                    final_design_res.append({
                        'span_id': i, 'L': s_len, 'b': b_mm, 'h': h_mm, 'fc': fc, 'fy': fy,
                        'Mu_pos': mu_pos, 'Mu_neg': mu_neg, 'Vu_max': vu_max,
                        'cover': cover_mm, 'top_db': top_db, 'bot_db': bot_db, 'stir_db': stir_db,
                        'pos': {'n': bot_n, 'area': as_prov_bot, 'status': pass_b},
                        'neg': {'n': top_n, 'area': as_prov_top, 'status': pass_t},
                        'shear': {'s': stir_s, 'db': stir_db, 'status': status_v},
                        'Ma_pos_svc': ma_pos_svc, 'delta_svc_mm': delta_svc_mm,
                        'bot': {'n': bot_n, 'db': bot_db}, 'top': {'n': top_n, 'db': top_db},
                    })

            # Design Summary
            st.markdown("---")
            st.subheader("📋 Design Summary")
            summary_data = []
            for item in final_design_res:
                summary_data.append({
                    "Span": item['span_id'] + 1,
                    "Bottom": f"{item['pos']['n']}-DB{item['bot_db']}",
                    "Top": f"{item['neg']['n']}-DB{item['top_db']}",
                    "Stirrup": f"RB{item['stir_db']}@{item['shear']['s']}",
                    "Status": "✅ Pass" if (item['pos']['status'] and item['neg']['status'] and item['shear']['status'] == "OK") else "❌ Fail"
                })
            st.table(pd.DataFrame(summary_data))
            
            if st.button("🔄 Generate Drawings", type="primary"):
                try:
                    st.write("**Longitudinal Section:**")
                    fig_long = section_plotter.plot_longitudinal_section_detailed(spans, sup_df, final_design_res, h_mm, final_design_res[0]['cover'])
                    st.pyplot(fig_long, use_container_width=True)
                except Exception as e: st.error(f"Drawing Error: {e}")

        # === TAB 3: REPORT ===
        with tab3:
            st.header("📝 Detailed Calculation Reports")
            st.markdown(f"**Project:** {project_name} | **Engineer:** {engineer_name}")
            if not final_design_res: st.warning("⚠️ Please complete the design in Tab 2 first.")
            else:
                for i, res in enumerate(final_design_res):
                    with st.expander(f"📘 Calculation Sheet: Span {i+1}", expanded=False):
                        reporter.render_calculation_report(res)

    except Exception as e:
        st.error(f"❌ Application Error: {e}")
        import traceback
        st.code(traceback.format_exc())
