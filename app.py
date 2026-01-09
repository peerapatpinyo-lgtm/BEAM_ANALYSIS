import streamlit as st
import pandas as pd
import numpy as np
import io

# --- 1. IMPORT CUSTOM MODULES ---
import input_handler, solver, design_view, section_plotter, reporter
import rc_utils, rc_design_engine, rc_load_processor, app_styles

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(page_title="Pro RC Beam Design", layout="wide", page_icon="🏗️")
app_styles.apply_custom_css()

# --- 3. MAIN HEADER ---
st.markdown('<div class="main-header">🏗️ RC Beam Analysis & Design Pro</div>', unsafe_allow_html=True)

# --- 4. SIDEBAR ---
with st.sidebar:
    params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()

if not stable:
    st.error("🚨 **Structure Error:** โครงสร้างไม่เสถียร!")
else:
    # --- ANALYSIS SETTINGS ---
    col_set1, col_set2 = st.columns([1, 2])
    with col_set1:
        st.markdown("### ⚙️ Load Factors")
        mode_select = st.radio("Design Mode:", ["Service Load (Check Deflection)", "Ultimate Strength (Design)"], index=1)
    
    with col_set2:
        st.markdown("### 🔢 Factors")
        c1, c2 = st.columns(2)
        if "Service" in mode_select:
            f_dl, f_ll = 1.0, 1.0
            tag, is_service = "Service", True
        else:
            f_dl = c1.number_input("Dead Load (DL)", 1.4, 1.6, 1.4, 0.1)
            f_ll = c2.number_input("Live Load (LL)", 1.7, 2.0, 1.7, 0.1)
            tag, is_service = "Ultimate", False

    try:
        # --- ANALYSIS ENGINE ---
        calc_loads_ult = rc_load_processor.prepare_load_dataframe(loads_df, n_spans, spans, params, f_dl, f_ll)
        x_ult, M_ult, V_ult, D_ult, R_ult = solver.solve_beam(spans, sup_df, calc_loads_ult, params)
        calc_loads_svc = rc_load_processor.prepare_load_dataframe(loads_df, n_spans, spans, params, 1.0, 1.0)
        x_svc, M_svc, V_svc, D_svc, R_svc = solver.solve_beam(spans, sup_df, calc_loads_svc, params)

        x_plot, M_plot, V_plot, D_plot, R_plot = (x_svc, M_svc, V_svc, D_svc, R_svc) if is_service else (x_ult, M_ult, V_ult, D_ult, R_ult)

        tab1, tab2, tab3 = st.tabs(["📊 1. Analysis Results", "📝 2. Concrete Design", "📘 3. Report"])
        final_design_res = []

        # ================= TAB 2: CONCRETE DESIGN =================
        with tab2:
            st.header("🏗️ Reinforcement Detailing")
            b_mm, h_mm = rc_utils.normalize_section_units(params['b'], params['h'])
            fc, fy = params['fc'], params['fy']
            offsets = [0] + list(np.cumsum(spans))
            
            for i in range(n_spans):
                s_len, s_start, s_end = spans[i], offsets[i], offsets[i+1]
                
                # --- Analysis Data Extraction ---
                mask_u = (x_ult >= s_start - 1e-6) & (x_ult <= s_end + 1e-6)
                mu_pos, mu_neg = max(0.0, (M_ult[mask_u]/1000.0).max()), abs(min(0.0, (M_ult[mask_u]/1000.0).min()))
                vu_max = abs((V_ult[mask_u] / 1000.0)).max()

                mask_s = (x_svc >= s_start - 1e-6) & (x_svc <= s_end + 1e-6)
                ma_pos_svc, delta_svc_mm = max(0.0, (M_svc[mask_s]/1000.0).max()), abs(D_svc[mask_s] * 1000.0).max()

                with st.expander(f"📍 SPAN {i+1} (L={s_len} m)", expanded=True):
                    col_input, col_draw = st.columns([2, 1])
                    with col_input:
                        cover_mm = st.number_input(f"Cover (mm)", 20, 50, 25, key=f"cov_{i}")

                        # --- 1. TOP STEEL (Support) ---
                        st.markdown("#### 🔼 Top Reinforcement (Negative Moment)")
                        ct1, ct2, ct3 = st.columns([2, 2, 1])
                        with ct1: t_db = st.selectbox("Size", [12, 16, 20, 25, 28], index=1, key=f"tdb_{i}")
                        with ct2: t_qty = st.number_input("Qty", 2, 20, 2, key=f"tn_{i}")
                        with ct3: t_lay = st.selectbox("Layers", [1, 2, 3], index=0, key=f"tl_{i}")
                        
                        d_t = h_mm - (cover_mm + 9 + t_db/2 + (t_lay-1)*(25 + t_db)/2)
                        as_req_t, _, _ = rc_design_engine.get_as_req(mu_neg, d_t, fc, fy, b_mm)
                        as_min_t = max((0.25 * np.sqrt(fc) / fy) * b_mm * d_t, (1.4 / fy) * b_mm * d_t)
                        phi_Mn_t, as_prov_t, _, _, _, _ = rc_design_engine.get_phi_Mn_details(t_qty, t_db, d_t, b_mm, fc, fy)

                        st.markdown(f"""
| **Top Steel Analysis** | **Required** | **Minimum** | **Provided** | **Status** |
| :--- | :---: | :---: | :---: | :---: |
| **Area ($A_s$, mm²)** | {as_req_t:.0f} | {as_min_t:.0f} | **{as_prov_t:.0f}** | {"✅" if as_prov_t >= max(as_req_t, as_min_t) else "❌"} |
| **Capacity (kNm)** | $M_u$: {mu_neg:.1f} | --- | **$\phi M_n$: {phi_Mn_t:.1f}** | {"✅" if phi_Mn_t >= mu_neg else "❌"} |
""")

                        # --- 2. BOTTOM STEEL (Mid-Span) ---
                        st.markdown("#### 🔽 Bottom Reinforcement (Positive Moment)")
                        cb1, cb2, cb3 = st.columns([2, 2, 1])
                        with cb1: b_db = st.selectbox("Size", [12, 16, 20, 25, 28], index=1, key=f"bdb_{i}")
                        with cb2: b_qty = st.number_input("Qty", 2, 20, 3, key=f"bn_{i}")
                        with cb3: b_lay = st.selectbox("Layers", [1, 2, 3], index=0, key=f"bl_{i}")
                        
                        d_b = h_mm - (cover_mm + 9 + b_db/2 + (b_lay-1)*(25 + b_db)/2)
                        as_req_b, _, _ = rc_design_engine.get_as_req(mu_pos, d_b, fc, fy, b_mm)
                        phi_Mn_b, as_prov_b, _, _, _, _ = rc_design_engine.get_phi_Mn_details(b_qty, b_db, d_b, b_mm, fc, fy)

                        st.markdown(f"""
| **Bottom Steel Analysis** | **Required** | **Minimum** | **Provided** | **Status** |
| :--- | :---: | :---: | :---: | :---: |
| **Area ($A_s$, mm²)** | {as_req_b:.0f} | {as_min_t:.0f} | **{as_prov_b:.0f}** | {"✅" if as_prov_b >= max(as_req_b, as_min_t) else "❌"} |
| **Capacity (kNm)** | $M_u$: {mu_pos:.1f} | --- | **$\phi M_n$: {phi_Mn_b:.1f}** | {"✅" if phi_Mn_b >= mu_pos else "❌"} |
""")

                        # --- 3. SHEAR STIRRUPS ---
                        st.markdown("#### 🌀 Shear Reinforcement (Stirrups)")
                        cs1, cs2 = st.columns(2)
                        with cs1: stir_db = st.selectbox("Stirrup Size (mm)", [6, 9, 12], index=1, key=f"sdb_final_{i}")
                        with cs2: stir_s = st.number_input("Spacing (mm)", 50, 300, 150, key=f"ss_{i}")
                        
                        status_v, phi_Vn, _, _, _, _ = rc_design_engine.check_shear_details(vu_max, b_mm, d_b, fc, fy, stir_db, stir_s)
                        if phi_Vn < vu_max: st.error(f"❌ **Shear Failure:** $\phi V_n$ {phi_Vn:.1f} < $V_u$ {vu_max:.1f} kN")
                        else: st.success(f"✅ **Shear Capacity Passed:** $\phi V_n$ {phi_Vn:.1f} ≥ $V_u$ {vu_max:.1f} kN")

                    with col_draw:
                        cs_data = {
                            'b': b_mm, 'h': h_mm, 'cover': cover_mm, 
                            'top': {'n': t_qty, 'layers': t_lay, 'db': t_db}, 
                            'bot': {'n': b_qty, 'layers': b_lay, 'db': b_db}, 
                            'top_db': t_db, 'bot_db': b_db, 'stir_db': stir_db, 
                            'shear': {'s': stir_s}
                        }
                        st.components.v1.html(f'<div style="background:white; padding:10px; border-radius:10px; border:1px solid #ddd;">{section_plotter.plot_cross_section(cs_data)}</div>', height=420)

                    final_design_res.append({
                        'span_id': i, 'L': s_len, 'b': b_mm, 'h': h_mm, 'fc': fc, 'fy': fy, 
                        'Mu_pos': mu_pos, 'Mu_neg': mu_neg, 'Vu_max': vu_max, 'cover': cover_mm,
                        'Ma_pos_svc': ma_pos_svc, 'delta_svc_mm': delta_svc_mm, 
                        'top_db': t_db, 'bot_db': b_db, 'stir_db': stir_db, 
                        'pos': {'n': b_qty, 'area': as_prov_b, 'layers': b_lay, 'db': b_db, 'status': (phi_Mn_b >= mu_pos)},
                        'neg': {'n': t_qty, 'area': as_prov_t, 'layers': t_lay, 'db': t_db, 'status': (phi_Mn_t >= mu_neg)},
                        'shear': {'s': stir_s, 'db': stir_db, 'status': status_v},
                        'top': {'n': t_qty, 'db': t_db, 'layers': t_lay},
                        'bot': {'n': b_qty, 'db': b_db, 'layers': b_lay}
                    })

            st.markdown("---")
            if st.button("🏗️ Generate Detailed Drawing (Longitudinal Section)"):
                svg_long, _ = section_plotter.plot_longitudinal_section_detailed(spans, sup_df, final_design_res, h_mm, cover_mm)
                st.components.v1.html(f'<div style="background:white; overflow-x:auto; border:1px solid #ddd; border-radius:8px; padding:10px;">{svg_long}</div>', height=500, scrolling=True)

        # ================= TAB 1: ANALYSIS RESULTS =================
        with tab1:
            st.subheader(f"📈 Diagrams ({tag} Load)")
            df_for_plot = pd.DataFrame({'x': x_plot, 'moment': M_plot, 'shear': V_plot, 'deflection': D_plot * 1000})
            fig = design_view.plot_analysis_results(df_for_plot, spans, sup_df, calc_loads_ult if not is_service else calc_loads_svc, R_plot)
            st.plotly_chart(fig, use_container_width=True)
            
            c_m1, c_m2, c_m3 = st.columns(3)
            c_m1.metric(f"Max Shear ({tag})", f"{max(abs(V_plot))/1000:.2f} kN")
            c_m2.metric(f"Max Moment ({tag})", f"{max(M_plot)/1000:.2f} kNm")
            c_m3.metric(f"Max Deflection", f"{max(abs(D_plot))*1000:.2f} mm")

        # ================= TAB 3: REPORT =================
        with tab3:
            st.header("📝 Calculation Reports")
            if not final_design_res:
                st.warning("⚠️ กรุณาทำการออกแบบใน Tab 2 ก่อน")
            else:
                for res in final_design_res:
                    with st.expander(f"📘 Span {res['span_id']+1} Details", expanded=(res['span_id']==0)):
                        reporter.render_calculation_report(res)

    except Exception as e:
        st.error(f"❌ **System Error:** {e}")
