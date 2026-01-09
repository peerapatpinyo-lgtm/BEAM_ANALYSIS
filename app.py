import streamlit as st
import pandas as pd
import numpy as np
import io
import time

# --- 1. IMPORT CUSTOM MODULES ---
import input_handler, solver, design_view, section_plotter, reporter
import rc_utils, rc_design_engine, rc_load_processor, app_styles

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(page_title="Pro RC Beam Design", layout="wide", page_icon="🏗️")
app_styles.apply_custom_css()

# --- 3. MAIN APPLICATION ---
st.markdown('<div class="main-header">🏗️ RC Beam Analysis & Design Pro</div>', unsafe_allow_html=True)

# --- 4. SIDEBAR & LOGIC ---
with st.sidebar:
    params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()

if not stable:
    st.error("🚨 **Structure Error:** โครงสร้างไม่เสถียร!")
else:
    try:
        # --- 4.1 ANALYSIS ENGINE ---
        calc_loads_ult = rc_load_processor.prepare_load_dataframe(loads_df, n_spans, spans, params, 1.4, 1.7)
        x_ult, M_ult, V_ult, D_ult, R_ult = solver.solve_beam(spans, sup_df, calc_loads_ult, params)
        
        tab1, tab2, tab3 = st.tabs(["📊 1. Analysis Results", "📝 2. Concrete Design", "📘 3. Report"])
        final_design_res = []

        # ================= TAB 2: INTERACTIVE DESIGN =================
        with tab2:
            st.header("🏗️ Reinforcement Detailing (Multi-Layer Support)")
            b_mm, h_mm = rc_utils.normalize_section_units(params['b'], params['h'])
            fc, fy = params['fc'], params['fy']
            offsets = [0] + list(np.cumsum(spans))
            
            for i in range(n_spans):
                s_len, s_start, s_end = spans[i], offsets[i], offsets[i+1]
                mask = (x_ult >= s_start - 1e-6) & (x_ult <= s_end + 1e-6)
                mu_pos = max(0.0, (M_ult[mask] / 1000.0).max()) if any(mask) else 0
                mu_neg = abs(min(0.0, (M_ult[mask] / 1000.0).min())) if any(mask) else 0
                vu_max = abs((V_ult[mask] / 1000.0)).max() if any(mask) else 0

                with st.expander(f"📍 SPAN {i+1} (L={s_len} m)", expanded=True):
                    col_input, col_draw = st.columns([2, 1])
                    
                    with col_input:
                        cover_mm = st.number_input(f"Cover (mm)", 20, 50, 25, 5, key=f"cov_{i}")
                        stir_db_input = st.selectbox("Stirrup Size (RB/DB)", [6, 9, 12], index=1, key=f"sdb_main_{i}")
                        
                        # --- 1. TOP STEEL (SUPPORT) ---
                        st.markdown("#### 🔼 Top Steel (Support)")
                        ct1, ct2, ct3 = st.columns([2, 2, 1])
                        with ct1: top_db = st.selectbox("Size", [12, 16, 20, 25, 28], index=1, key=f"tdb_{i}")
                        with ct2: top_n = st.number_input("Total Qty", 2, 20, 2, key=f"tn_{i}")
                        with ct3: top_layers = st.selectbox("Layers", [1, 2, 3], index=0, key=f"tl_{i}")
                        
                        # คำนวณ d_effective สำหรับเหล็กหลายชั้น (สมมติระยะห่างระหว่างชั้น 25mm)
                        d_offset_top = cover_mm + stir_db_input + (top_db / 2) + ((top_layers - 1) * 25 / 2)
                        d_real_t = h_mm - d_offset_top
                        
                        as_req_calc_top, _, _ = rc_design_engine.get_as_req(mu_neg, d_real_t, fc, fy, b_mm)
                        as_min = max((0.25 * np.sqrt(fc) / fy) * b_mm * d_real_t, (1.4 / fy) * b_mm * d_real_t)
                        phi_Mn_top, as_prov_top, _, _, _, _ = rc_design_engine.get_phi_Mn_details(top_n, top_db, d_real_t, b_mm, fc, fy)

                        st.markdown(f"""
                        | Parameter | Req. As ($M_u$) | Min As ($A_{{s,min}}$) | Provided As | Status |
                        | :--- | :--- | :--- | :--- | :--- |
                        | **Steel Area** | {as_req_calc_top:.0f} mm² | {as_min:.0f} mm² | **{as_prov_top:.0f} mm²** | {"✅" if as_prov_top >= max(as_req_calc_top, as_min) else "❌"} |
                        | **Capacity** | $M_u$: {mu_neg:.1f} | - | **$\phi M_n$: {phi_Mn_top:.1f}** | {"✅" if phi_Mn_top >= mu_neg else "❌"} |
                        """)

                        st.markdown("---")

                        # --- 2. BOTTOM STEEL (MID-SPAN) ---
                        st.markdown("#### 🔽 Bottom Steel (Mid-Span)")
                        cb1, cb2, cb3 = st.columns([2, 2, 1])
                        with cb1: bot_db = st.selectbox("Size", [12, 16, 20, 25, 28], index=1, key=f"bdb_{i}")
                        with cb2: bot_n = st.number_input("Total Qty", 2, 20, 3, key=f"bn_{i}")
                        with cb3: bot_layers = st.selectbox("Layers", [1, 2, 3], index=0, key=f"bl_{i}")
                        
                        d_offset_bot = cover_mm + stir_db_input + (bot_db / 2) + ((bot_layers - 1) * 25 / 2)
                        d_real_b = h_mm - d_offset_bot
                        
                        as_req_calc_bot, _, _ = rc_design_engine.get_as_req(mu_pos, d_real_b, fc, fy, b_mm)
                        phi_Mn_bot, as_prov_bot, _, _, _, _ = rc_design_engine.get_phi_Mn_details(bot_n, bot_db, d_real_b, b_mm, fc, fy)

                        st.markdown(f"""
                        | Parameter | Req. As ($M_u$) | Min As ($A_{{s,min}}$) | Provided As | Status |
                        | :--- | :--- | :--- | :--- | :--- |
                        | **Steel Area** | {as_req_calc_bot:.0f} mm² | {as_min:.0f} mm² | **{as_prov_bot:.0f} mm²** | {"✅" if as_prov_bot >= max(as_req_calc_bot, as_min) else "❌"} |
                        | **Capacity** | $M_u$: {mu_pos:.1f} | - | **$\phi M_n$: {phi_Mn_bot:.1f}** | {"✅" if phi_Mn_bot >= mu_pos else "❌"} |
                        """)

                        st.markdown("---")

                        # --- 3. SHEAR ---
                        st.markdown("#### 🌀 Shear Stirrups")
                        cs1, cs2 = st.columns(2)
                        with cs1: stir_s = st.number_input("Spacing (mm)", 50, 300, 150, 10, key=f"ss_{i}")
                        status_v, phi_Vn, _, _, _, _ = rc_design_engine.check_shear_details(vu_max, b_mm, d_real_b, fc, fy, stir_db_input, stir_s)
                        if phi_Vn < vu_max: st.error(f"❌ **FAIL:** $\phi V_n$ {phi_Vn:.1f} < $V_u$ {vu_max:.1f} kN")
                        else: st.success(f"✅ **PASS:** $\phi V_n$ {phi_Vn:.1f} ≥ $V_u$ {vu_max:.1f} kN")

                    with col_draw:
                        st.markdown("<p style='text-align:center;'><b>Section Preview</b></p>", unsafe_allow_html=True)
                        cs_data = {
                            'b': b_mm, 'h': h_mm, 'cover': cover_mm,
                            'top': {'n': top_n, 'layers': top_layers}, 'top_db': top_db,
                            'bot': {'n': bot_n, 'layers': bot_layers}, 'bot_db': bot_db,
                            'stir_db': stir_db_input, 'shear': {'s': stir_s}
                        }
                        st.components.v1.html(f'<div style="background:white; padding:10px;">{section_plotter.plot_cross_section(cs_data)}</div>', height=400)

                    final_design_res.append({
                        'span_id': i, 'L': s_len, 'b': b_mm, 'h': h_mm, 'fc': fc, 'fy': fy,
                        'Mu_pos': mu_pos, 'Mu_neg': mu_neg, 'Vu_max': vu_max,
                        'top_db': top_db, 'bot_db': bot_db, 'stir_db': stir_db_input,
                        'pos': {'n': bot_n, 'layers': bot_layers, 'area': as_prov_bot},
                        'neg': {'n': top_n, 'layers': top_layers, 'area': as_prov_top},
                        'shear': {'s': stir_s, 'status': status_v}
                    })

    except Exception as e:
        st.error(f"❌ Error: {e}")
