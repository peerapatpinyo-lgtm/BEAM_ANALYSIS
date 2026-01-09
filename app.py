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

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Pro RC Beam Design", 
    layout="wide", 
    page_icon="🏗️",
    initial_sidebar_state="expanded"
)

app_styles.apply_custom_css()

# --- 3. MAIN APPLICATION ---
st.markdown('<div class="main-header">🏗️ RC Beam Analysis & Design Pro</div>', unsafe_allow_html=True)
st.markdown("---")

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("📝 Project Information")
    project_name = st.text_input("Project Name", "Residential Building A")
    engineer_name = st.text_input("Engineer", "Eng. Somchai")
    st.markdown("---")
    params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()

if not stable:
    st.error("🚨 **Structure Error:** โครงสร้างไม่เสถียร!")
else:
    # --- ANALYSIS LOGIC ---
    try:
        with st.spinner('Calculating...'):
            # Ultimate Load Factor (1.4DL + 1.7LL)
            f_dl, f_ll = 1.4, 1.7
            calc_loads_ult = rc_load_processor.prepare_load_dataframe(loads_df, n_spans, spans, params, f_dl, f_ll)
            x_ult, M_ult, V_ult, D_ult, R_ult = solver.solve_beam(spans, sup_df, calc_loads_ult, params)
            
            # Service Load Factor (1.0DL + 1.0LL)
            calc_loads_svc = rc_load_processor.prepare_load_dataframe(loads_df, n_spans, spans, params, 1.0, 1.0)
            x_svc, M_svc, V_svc, D_svc, R_svc = solver.solve_beam(spans, sup_df, calc_loads_svc, params)

        tab1, tab2, tab3 = st.tabs(["📊 1. Analysis", "📝 2. Design & Detailing", "📘 3. Report"])
        final_design_res = []

        # ================= TAB 1: ANALYSIS =================
        with tab1:
            st.subheader("📈 Force Diagrams (Ultimate Load)")
            df_for_plot = pd.DataFrame({'x': x_ult, 'moment': M_ult, 'shear': V_ult, 'deflection': D_ult * 1000})
            fig = design_view.plot_analysis_results(df_for_plot, spans, sup_df, calc_loads_ult, R_ult)
            st.plotly_chart(fig, use_container_width=True)

        # ================= TAB 2: INTERACTIVE DESIGN =================
        with tab2:
            st.header("🏗️ Beam Reinforcement Design")
            b_mm, h_mm = rc_utils.normalize_section_units(params['b'], params['h'])
            fc, fy = params['fc'], params['fy']
            offsets = [0] + list(np.cumsum(spans))
            
            for i in range(n_spans):
                s_len, s_start, s_end = spans[i], offsets[i], offsets[i+1]
                
                # Get Forces for current span
                mask_ult = (x_ult >= s_start - 1e-6) & (x_ult <= s_end + 1e-6)
                mu_pos = max(0.0, (M_ult[mask_ult] / 1000.0).max()) if any(mask_ult) else 0
                mu_neg = abs(min(0.0, (M_ult[mask_ult] / 1000.0).min())) if any(mask_ult) else 0
                vu_max = abs((V_ult[mask_ult] / 1000.0)).max() if any(mask_ult) else 0
                
                # Get Service values
                mask_svc = (x_svc >= s_start - 1e-6) & (x_svc <= s_end + 1e-6)
                ma_pos_svc = max(0.0, (M_svc[mask_svc] / 1000.0).max()) if any(mask_svc) else 0
                delta_svc_mm = abs((D_svc[mask_svc] * 1000.0)).max() if any(mask_svc) else 0

                with st.expander(f"📍 SPAN {i+1} (L={s_len} m)", expanded=True):
                    col_input, col_draw = st.columns([2, 1])
                    
                    with col_input:
                        cover_mm = st.number_input(f"Cover (mm)", 20, 50, 25, 5, key=f"cov_{i}")
                        d_est = h_mm - cover_mm - 20
                        as_min = max((0.25 * np.sqrt(fc) / fy) * b_mm * d_est, (1.4 / fy) * b_mm * d_est)

                        # --- BOTTOM STEEL ---
                        st.markdown("#### 🔽 Bottom Steel (Mid-Span)")
                        c1, c2 = st.columns(2)
                        with c1: bot_db = st.selectbox("DB Size", [12, 16, 20, 25, 28], index=1, key=f"bdb_{i}")
                        with c2: bot_n = st.number_input("Qty", 2, 12, 3, key=f"bn_{i}")
                        
                        as_req_bot = max(rc_design_engine.get_as_req(mu_pos, d_est, fc, fy, b_mm)[0], as_min)
                        d_real_b = h_mm - cover_mm - 9 - (bot_db / 2)
                        phi_Mn_bot, as_prov_bot, _, _, _, _ = rc_design_engine.get_phi_Mn_details(bot_n, bot_db, d_real_b, b_mm, fc, fy)

                        st.markdown(f"""
                        | Parameter | Required (Min) | Provided (Actual) | Status |
                        | :--- | :--- | :--- | :--- |
                        | **Steel Area ($A_s$)** | {as_req_bot:.0f} mm² | **{as_prov_bot:.0f} mm²** | {"✅" if as_prov_bot >= as_req_bot else "❌"} |
                        | **Moment ($\phi M_n$)** | {mu_pos:.1f} kNm | **{phi_Mn_bot:.1f} kNm** | {"✅" if phi_Mn_bot >= mu_pos else "❌"} |
                        """)

                        # --- TOP STEEL ---
                        st.markdown("#### 🔼 Top Steel (Support)")
                        c3, c4 = st.columns(2)
                        with c3: top_db = st.selectbox("DB Size", [12, 16, 20, 25, 28], index=1, key=f"tdb_{i}")
                        with c4: top_n = st.number_input("Qty", 2, 12, 2, key=f"tn_{i}")
                        
                        as_req_top = max(rc_design_engine.get_as_req(mu_neg, d_est, fc, fy, b_mm)[0], as_min)
                        d_real_t = h_mm - cover_mm - 9 - (top_db / 2)
                        phi_Mn_top, as_prov_top, _, _, _, _ = rc_design_engine.get_phi_Mn_details(top_n, top_db, d_real_t, b_mm, fc, fy)

                        st.markdown(f"""
                        | Parameter | Required (Min) | Provided (Actual) | Status |
                        | :--- | :--- | :--- | :--- |
                        | **Steel Area ($A_s$)** | {as_req_top:.0f} mm² | **{as_prov_top:.0f} mm²** | {"✅" if as_prov_top >= as_req_top else "❌"} |
                        | **Moment ($\phi M_n$)** | {mu_neg:.1f} kNm | **{phi_Mn_top:.1f} kNm** | {"✅" if phi_Mn_top >= mu_neg else "❌"} |
                        """)

                        # --- SHEAR ---
                        st.markdown("#### 🌀 Shear Stirrups")
                        c5, c6 = st.columns(2)
                        with c5: stir_db = st.selectbox("Stirrup Size", [6, 9, 12], index=1, key=f"sdb_{i}")
                        with c6: stir_s = st.number_input("Spacing (mm)", 50, 300, 150, 10, key=f"ss_{i}")
                        status_v, phi_Vn, _, _, _, _ = rc_design_engine.check_shear_details(vu_max, b_mm, d_real_b, fc, fy, stir_db, stir_s)
                        st.info(f"Strength: $\phi V_n$ = {phi_Vn:.1f} kN vs $V_u$ = {vu_max:.1f} kN → **{status_v}**")

                    with col_draw:
                        st.markdown("<p style='text-align:center;'><b>Cross Section Preview</b></p>", unsafe_allow_html=True)
                        cs_data = {'b': b_mm, 'h': h_mm, 'cover': cover_mm, 'top': {'n': top_n}, 'top_db': top_db, 'bot': {'n': bot_n}, 'bot_db': bot_db, 'stir_db': stir_db, 'shear': {'s': stir_s}}
                        st.components.v1.html(f'<div style="background:white; padding:10px;">{section_plotter.plot_cross_section(cs_data)}</div>', height=400)

                    final_design_res.append({
                        'span_id': i, 'L': s_len, 'b': b_mm, 'h': h_mm, 'fc': fc, 'fy': fy,
                        'Mu_pos': mu_pos, 'Mu_neg': mu_neg, 'Vu_max': vu_max, 'cover': cover_mm,
                        'top_db': top_db, 'bot_db': bot_db, 'stir_db': stir_db,
                        'pos': {'n': bot_n, 'area': as_prov_bot, 'status': (phi_Mn_bot >= mu_pos)},
                        'neg': {'n': top_n, 'area': as_prov_top, 'status': (phi_Mn_top >= mu_neg)},
                        'shear': {'s': stir_s, 'db': stir_db, 'status': status_v},
                        'bot': {'n': bot_n, 'db': bot_db}, 'top': {'n': top_n, 'db': top_db},
                        'Ma_pos_svc': ma_pos_svc, 'delta_svc_mm': delta_svc_mm
                    })

            if st.button("🏗️ Generate Detailed Drawing", type="primary"):
                svg_long, png_data = section_plotter.plot_longitudinal_section_detailed(spans, sup_df, final_design_res, h_mm, cover_mm)
                st.components.v1.html(f'<div style="background:white; overflow-x:auto;">{svg_long}</div>', height=450, scrolling=True)

        # ================= TAB 3: REPORT =================
        with tab3:
            st.header("📝 Calculation Reports")
            for i, res in enumerate(final_design_res):
                with st.expander(f"📘 Span {i+1} Details", expanded=False):
                    reporter.render_calculation_report(res)

    except Exception as e:
        st.error(f"❌ Application Error: {e}")
        st.code(time.strftime("%H:%M:%S"))
