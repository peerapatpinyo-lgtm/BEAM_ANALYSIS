# app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 1. IMPORT CUSTOM MODULES ---
import input_handler, solver, design_view, section_plotter, reporter
import rc_utils, rc_design_engine, rc_load_processor, app_styles

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(page_title="Pro RC Beam Design", layout="wide", page_icon="🏗️")
app_styles.apply_custom_css()

# --- HELPER: REBAR WEIGHT ---
def get_rebar_weight(d_mm):
    """Calculate weight of rebar per meter (kg/m)"""
    return (d_mm ** 2) / 162.0

# --- 3. INTERNAL HELPER: PLOT CROSS SECTION ---
def plot_cross_section_fixed(b, h, cover, top_layers, bot_layers, shear_res):
    """
    ฟังก์ชันวาดรูปหน้าตัดคาน (Cross Section)
    """
    fig, ax = plt.subplots(figsize=(5, 6))
    
    # Draw Concrete Section
    rect = patches.Rectangle((0, 0), b, h, linewidth=2, edgecolor='black', facecolor='white')
    ax.add_patch(rect)
    
    # Draw Stirrup
    stirrup_rect = patches.Rectangle((cover, cover), b - 2*cover, h - 2*cover, 
                                     linewidth=1.5, edgecolor='#34495e', facecolor='none', linestyle='-')
    ax.add_patch(stirrup_rect)
    
    # Draw Top Rebars
    n_top = sum(l['n'] for l in top_layers)
    dia_top = top_layers[0]['db'] if top_layers else 12
    start_x = cover + dia_top/2
    end_x = b - cover - dia_top/2
    
    if n_top > 1:
        gap = (end_x - start_x) / (n_top - 1)
        for i in range(n_top):
            ax.add_patch(patches.Circle((start_x + i*gap, h - cover - dia_top/2), radius=dia_top/2, color='#c0392b'))
    elif n_top == 1:
        ax.add_patch(patches.Circle((b/2, h - cover - dia_top/2), radius=dia_top/2, color='#c0392b'))

    # Draw Bottom Rebars
    n_bot = sum(l['n'] for l in bot_layers)
    dia_bot = bot_layers[0]['db'] if bot_layers else 12
    start_x = cover + dia_bot/2
    end_x = b - cover - dia_bot/2
    
    if n_bot > 1:
        gap = (end_x - start_x) / (n_bot - 1)
        for i in range(n_bot):
            ax.add_patch(patches.Circle((start_x + i*gap, cover + dia_bot/2), radius=dia_bot/2, color='#27ae60'))
    elif n_bot == 1:
        ax.add_patch(patches.Circle((b/2, cover + dia_bot/2), radius=dia_bot/2, color='#27ae60'))

    # Annotations
    text_x = b + (b * 0.1)
    ax.text(text_x, h - cover, f"Top: {n_top}DB{int(dia_top)}", color='#c0392b', fontsize=12, fontweight='bold', va='center')
    ax.text(text_x, cover + dia_bot, f"Bot: {n_bot}DB{int(dia_bot)}", color='#27ae60', fontsize=12, fontweight='bold', va='center')
    ax.text(text_x, h/2, f"Stirrup: RB{int(shear_res['db'])}@{int(shear_res['s'])}", color='#2c3e50', fontsize=10, fontweight='bold', va='center')

    # Plot Settings
    ax.set_title(f"SECTION {int(b)}x{int(h)} mm", fontsize=14, fontweight='bold', pad=20)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-50, b + 250) 
    ax.set_ylim(-50, h + 50)
    plt.tight_layout()
    return fig

# --- 4. MAIN HEADER ---
st.markdown('<div class="main-header">🏗️ RC Beam Analysis & Design Pro</div>', unsafe_allow_html=True)

# --- 5. SIDEBAR ---
with st.sidebar:
    # Get raw user inputs using input_handler module
    # Note: raw_user_loads_df assume units are [kN, kN/m, m]
    params, n_spans, spans, sup_df, raw_user_loads_df, stable = input_handler.render_all_sidebar_inputs()

if not stable:
    st.error("🚨 **Structure Error:** โครงสร้างไม่เสถียร (Unstable Structure) กรุณาตรวจสอบจุดรองรับ!")
else:
    # --- ANALYSIS SETTINGS ---
    col_set1, col_set2 = st.columns([1, 2])
    with col_set1:
        st.markdown("### ⚙️ Analysis Settings")
        mode_select = st.radio("Design Mode:", ["Service Load (Check Deflection)", "Ultimate Strength (Design)"], index=1)
        
        st.markdown("---")
        # [CRITICAL CHECKBOX]
        include_sw = st.checkbox("➕ Include Beam Self-weight", value=True)
        
        # --- UNIT CHECK: SELF WEIGHT ---
        # Density Concrete = 2400 kg/m^3
        # Gravity = 9.81 m/s^2
        # Weight (N/m^3) = 2400 * 9.81 = 23,544 N/m^3
        # Weight (kN/m^3) = 23.544 kN/m^3
        
        b_m = params.get('b', 300) / 1000.0  # mm -> m
        h_m = params.get('h', 500) / 1000.0  # mm -> m
        
        # คำนวณเป็น kN/m เพื่อให้หน่วยตรงกับ User Input อื่นๆ
        sw_val_kN_m = (b_m * h_m * 2400 * 9.81) / 1000.0 
        
        if include_sw:
            st.caption(f"ℹ️ **Added SW:** {sw_val_kN_m:.3f} kN/m")
        else:
            st.caption("ℹ️ **Excluded:** 0.00 kN/m")
    
    with col_set2:
        st.markdown("### 🔢 Load Factors")
        c1, c2 = st.columns(2)
        if "Service" in mode_select:
            f_dl, f_ll = 1.0, 1.0
            tag, is_service = "Service", True
        else:
            f_dl = c1.number_input("Dead Load (DL)", 1.4, 1.6, 1.4, 0.1)
            f_ll = c2.number_input("Live Load (LL)", 1.7, 2.0, 1.7, 0.1)
            tag, is_service = "Ultimate", False

    try:
        # ==========================================
        # ⚡ 1. PREPARE LOADS (ALL IN kN)
        # ==========================================
        
        # Start with User Loads (Assume User inputs kN)
        clean_user_loads = raw_user_loads_df.copy(deep=True)
        
        # Add Self Weight (in kN/m) if selected
        if include_sw:
            sw_rows = []
            for i in range(n_spans):
                sw_rows.append({
                    'span_index': i, 
                    'type': 'U', 
                    'mag': sw_val_kN_m,  # Unit: kN/m
                    'dist': spans[i], 
                    'd_start': 0, 
                    'case': 'DL'
                })
            df_sw = pd.DataFrame(sw_rows)
            final_loads_kN = pd.concat([clean_user_loads, df_sw], ignore_index=True)
            status_msg = "✅ **Self-Weight Included**"
        else:
            final_loads_kN = clean_user_loads
            status_msg = "❌ **Self-Weight Excluded**"

        # ==========================================
        # ⚡ 2. SOLVER EXECUTION (CONVERT kN -> N)
        # ==========================================
        
        def run_solver(load_df_kN, factor_dl, factor_ll):
            # 2.1 Combine Loads with Factors (Result is still kN)
            factored_loads_kN = rc_load_processor.prepare_load_dataframe(
                load_df_kN, n_spans, spans, params, factor_dl, factor_ll
            )
            
            # 2.2 CRITICAL: CONVERT kN -> N FOR SOLVER
            # Solver expects Base Units (Newtons). 
            # We explicitly multiply by 1000 here before sending to solver.
            solver_input_loads_N = factored_loads_kN.copy()
            if not solver_input_loads_N.empty:
                solver_input_loads_N['mag'] = solver_input_loads_N['mag'] * 1000.0
            
            # 2.3 Solve (Returns: x[m], M[Nm], V[N], D[m], R[N])
            x, M, V, D, R = solver.solve_beam(spans, sup_df, solver_input_loads_N, params)
            
            return x, M, V, D, R, factored_loads_kN

        # Run Ultimate
        x_ult, M_ult_Nm, V_ult_N, D_ult_m, R_ult_N, loads_ult_kN = run_solver(final_loads_kN, f_dl, f_ll)
        
        # Run Service
        x_svc, M_svc_Nm, V_svc_N, D_svc_m, R_svc_N, loads_svc_kN = run_solver(final_loads_kN, 1.0, 1.0)

        # Select Data for Display
        if is_service:
            x_plot = x_svc
            M_plot_raw = M_svc_Nm
            V_plot_raw = V_svc_N
            D_plot_raw = D_svc_m
            R_plot_raw = R_svc_N
            loads_display = loads_svc_kN
        else:
            x_plot = x_ult
            M_plot_raw = M_ult_Nm
            V_plot_raw = V_ult_N
            D_plot_raw = D_ult_m
            R_plot_raw = R_ult_N
            loads_display = loads_ult_kN

        # --- TABS START ---
        tab1, tab2, tab3 = st.tabs(["📊 1. Analysis Results", "📝 2. Concrete Design", "📘 3. Report & BOQ"])
        final_design_res = []

        # ================= TAB 1: ANALYSIS RESULTS =================
        with tab1:
            st.subheader(f"📈 Analysis Diagrams ({tag})")
            
            # --- DEBUGGER BOX ---
            with st.container():
                cols_chk = st.columns([1, 4])
                with cols_chk[0]: st.info(status_msg)
                with cols_chk[1]:
                    # Check Sum in kN
                    total_load_kN = loads_display['mag'].sum() if not loads_display.empty else 0
                    st.caption(f"🔍 **Total Load Check:** {total_load_kN:.2f} kN (Factored)")

            # =========================================================================
            # 🔧 3. UNIT NORMALIZATION FOR PLOTTING (N -> kN, Nm -> kNm)
            # =========================================================================
            
            # Prepare Reactions for Plotter (N -> kN)
            if isinstance(R_plot_raw, dict):
                R_display_kN = {k: v / 1000.0 for k, v in R_plot_raw.items()}
            elif isinstance(R_plot_raw, (list, np.ndarray)):
                 R_display_kN = [r / 1000.0 for r in R_plot_raw]
            else:
                 R_display_kN = R_plot_raw

            # Prepare Dataframe for Plotter
            df_for_plot = pd.DataFrame({
                'x': x_plot, 
                'moment': M_plot_raw / 1000.0,    # N-m -> kN-m
                'shear': V_plot_raw / 1000.0,     # N -> kN
                'deflection': D_plot_raw * 1000.0 # m -> mm
            })
            
            unique_chart_key = f"chart_{include_sw}_{tag}_{np.random.randint(0,100)}"
            
            # Plot (Expects: kN, kNm, mm)
            fig = design_view.plot_analysis_results(
                res_df=df_for_plot, 
                spans=spans, 
                supports=sup_df, 
                loads=loads_display,  # Already in kN
                reactions=R_display_kN # Converted to kN
            )
            st.plotly_chart(fig, use_container_width=True, key=unique_chart_key)
            
            # Metrics
            c_m1, c_m2, c_m3 = st.columns(3)
            c_m1.metric("Max Shear", f"{max(abs(df_for_plot['shear'])):.2f} kN")
            c_m2.metric("Max Moment", f"{max(df_for_plot['moment']):.2f} kNm")
            c_m3.metric("Max Deflection", f"{max(abs(df_for_plot['deflection'])):.2f} mm")
            
            with st.expander("🧐 View Load Data (kN)"):
                st.dataframe(loads_display)

        # ================= TAB 2: CONCRETE DESIGN =================
        with tab2:
            st.header("🏗️ Reinforcement Detailing")
            b_mm, h_mm = rc_utils.normalize_section_units(params['b'], params['h'])
            fc, fy = params['fc'], params['fy']
            offsets = [0] + list(np.cumsum(spans))
            
            for i in range(n_spans):
                s_len, s_start, s_end = spans[i], offsets[i], offsets[i+1]
                
                mask_u = (x_ult >= s_start - 1e-6) & (x_ult <= s_end + 1e-6)
                if not mask_u.any(): continue

                # FIX: Design Engine usually expects kNm and kN
                # Convert Raw Solver Results (N, Nm) -> (kN, kNm)
                mu_pos_kNm = max(0.0, (M_ult_Nm[mask_u]/1000.0).max())
                mu_neg_kNm = abs(min(0.0, (M_ult_Nm[mask_u]/1000.0).min()))
                vu_max_kN = abs((V_ult_N[mask_u] / 1000.0)).max()

                mask_s = (x_svc >= s_start - 1e-6) & (x_svc <= s_end + 1e-6)
                ma_pos_svc_kNm = max(0.0, (M_svc_Nm[mask_s]/1000.0).max())
                delta_elastic_mm = abs(D_svc_m[mask_s]).max() * 1000.0

                with st.expander(f"📍 SPAN {i+1} (L={s_len} m)", expanded=True):
                    col_input, col_draw = st.columns([2, 1])
                    with col_input:
                        cover_mm = st.number_input(f"Cover (mm)", 20, 50, 25, key=f"cov_{i}")

                        # Top Steel Input
                        st.markdown("#### 🔼 Top Reinforcement")
                        num_t_layers = st.selectbox("Top Layers", [1, 2, 3], index=0, key=f"tl_qty_{i}")
                        top_layers = []
                        for l_idx in range(num_t_layers):
                            ct1, ct2 = st.columns(2)
                            with ct1: t_db = st.selectbox(f"L{l_idx+1} Dia", [12, 16, 20, 25, 28], index=1, key=f"tdb_{i}_{l_idx}")
                            with ct2: t_qty = st.number_input(f"L{l_idx+1} No.", 0, 20, 2 if l_idx==0 else 0, key=f"tn_{i}_{l_idx}")
                            top_layers.append({'n': t_qty, 'db': t_db})
                        
                        # Top Steel Calc (Uses kNm)
                        d_t_val, as_prov_t, y_centroid_t = rc_design_engine.get_centroid_and_d(top_layers, h_mm, cover_mm, 9)
                        d_t = h_mm - y_centroid_t if y_centroid_t > 0 else h_mm - (cover_mm + 9 + 16/2)
                        as_req_t, _, _ = rc_design_engine.get_as_req(mu_neg_kNm, d_t, fc, fy, b_mm)
                        phi_Mn_t, _, _, _, _, _ = rc_design_engine.get_phi_Mn_details_multi(top_layers, d_t, b_mm, h_mm, fc, fy)
                        st.markdown(f"**Status (Top):** Prov: {as_prov_t:.0f} mm² | Cap: {phi_Mn_t:.1f} kNm {'✅' if phi_Mn_t >= mu_neg_kNm else '❌'}")

                        # Bottom Steel Input
                        st.markdown("#### 🔽 Bottom Reinforcement")
                        num_b_layers = st.selectbox("Bottom Layers", [1, 2, 3], index=0, key=f"bl_qty_{i}")
                        bot_layers = []
                        for l_idx in range(num_b_layers):
                            cb1, cb2 = st.columns(2)
                            with cb1: b_db = st.selectbox(f"L{l_idx+1} Dia", [12, 16, 20, 25, 28], index=1, key=f"bdb_{i}_{l_idx}")
                            with cb2: b_qty = st.number_input(f"L{l_idx+1} No.", 0, 20, 3 if l_idx==0 else 0, key=f"bn_{i}_{l_idx}")
                            bot_layers.append({'n': b_qty, 'db': b_db})
                        
                        # Bottom Steel Calc (Uses kNm)
                        d_b, as_prov_b, _ = rc_design_engine.get_centroid_and_d(bot_layers, h_mm, cover_mm, 9)
                        if d_b <= 0: d_b = h_mm - (cover_mm + 9 + 16/2)
                        as_req_b, _, _ = rc_design_engine.get_as_req(mu_pos_kNm, d_b, fc, fy, b_mm)
                        phi_Mn_b, _, _, _, _, _ = rc_design_engine.get_phi_Mn_details_multi(bot_layers, d_b, b_mm, h_mm, fc, fy)
                        st.markdown(f"**Status (Bot):** Prov: {as_prov_b:.0f} mm² | Cap: {phi_Mn_b:.1f} kNm {'✅' if phi_Mn_b >= mu_pos_kNm else '❌'}")

                        # Shear Input (Uses kN)
                        st.markdown("#### 🌀 Shear Stirrups")
                        cs1, cs2 = st.columns(2)
                        with cs1: stir_db = st.selectbox("Stirrup Dia", [6, 9, 12], index=1, key=f"sdb_final_{i}")
                        with cs2: stir_s = st.number_input("Spacing @", 50, 300, 150, key=f"ss_{i}")
                        status_v, phi_Vn, _, _, _, _ = rc_design_engine.check_shear_details(vu_max_kN, b_mm, d_b, fc, fy, stir_db, stir_s)
                        if phi_Vn < vu_max_kN: st.error(f"❌ Shear Fail: {phi_Vn:.1f} < {vu_max_kN:.1f} kN")
                        else: st.success(f"✅ Shear OK: {phi_Vn:.1f} ≥ {vu_max_kN:.1f} kN")

                        # Serviceability Checks (Uses kNm, mm)
                        st.markdown("---")
                        d_inst, d_long, Ie, Icr, lambda_d = rc_design_engine.check_serviceability(
                            ma_pos_svc_kNm, delta_elastic_mm, b_mm, h_mm, d_b, as_prov_b, as_prov_t, fc
                        )
                        limit_240 = (s_len * 1000) / 240
                        total_n_bars_bot = sum(l['n'] for l in bot_layers)
                        w_crack, fs_actual = rc_design_engine.check_crack_width(
                            Ma_svc=ma_pos_svc_kNm, b=b_mm, h=h_mm, d=d_b, As=as_prov_b, n_bars=total_n_bars_bot, fc=fc
                        )
                        limit_crack = 0.30
                        status_crack = "✅ Pass" if w_crack <= limit_crack else "⚠️ Warning"

                        col_chk1, col_chk2 = st.columns(2)
                        with col_chk1: st.metric("Deflection (L/240)", f"{d_long:.2f} mm", f"{'Pass' if d_long <= limit_240 else 'Fail'}")
                        with col_chk2: st.metric("Crack Width", f"{w_crack:.3f} mm", f"{'Pass' if w_crack <= limit_crack else 'Warning'}")

                    with col_draw:
                        # Draw Cross Section
                        fig_cs = plot_cross_section_fixed(b=b_mm, h=h_mm, cover=cover_mm, top_layers=top_layers, bot_layers=bot_layers, shear_res={'db': stir_db, 's': stir_s})
                        st.pyplot(fig_cs)
                        plt.close(fig_cs)

                    # Collect Data for Report
                    final_design_res.append({
                        'span_id': i, 'L': s_len, 'b': b_mm, 'h': h_mm, 'fc': fc, 'fy': fy, 
                        'Mu_pos': mu_pos_kNm, 'Mu_neg': mu_neg_kNm, 'Vu_max': vu_max_kN, 'cover': cover_mm,
                        'Ma_pos_svc': ma_pos_svc_kNm, 'delta_svc_mm': d_long, 
                        'top_db': top_layers[0]['db'] if top_layers else 12, 
                        'bot_db': bot_layers[0]['db'] if bot_layers else 12,
                        'stir_db': stir_db, 'stir_s': stir_s,
                        'pos': {'n': sum(l['n'] for l in bot_layers), 'area': as_prov_b, 'layers': bot_layers, 'status': (phi_Mn_b >= mu_pos_kNm)},
                        'neg': {'n': sum(l['n'] for l in top_layers), 'area': as_prov_t, 'layers': top_layers, 'status': (phi_Mn_t >= mu_neg_kNm)},
                        'shear': {'s': stir_s, 'db': stir_db, 'status': status_v},
                        'service': {'delta_long': d_long, 'limit_240': limit_240, 'ok': d_long <= limit_240},
                        'crack': {'w': w_crack, 'limit': limit_crack, 'status': status_crack},
                        'top': {'n': top_layers[0]['n'] if top_layers else 0, 'db': top_layers[0]['db'] if top_layers else 12, 'layers': num_t_layers, 'all_layers': top_layers},
                        'bot': {'n': bot_layers[0]['n'] if bot_layers else 0, 'db': bot_layers[0]['db'] if bot_layers else 12, 'layers': num_b_layers, 'all_layers': bot_layers}
                    })

            st.markdown("---")
            if st.button("🏗️ Generate Detailed Drawing"):
                # Call custom plotter for detailed longitudinal section
                svg_long, _ = section_plotter.plot_longitudinal_section_detailed(spans, sup_df, final_design_res, h_mm, cover_mm)
                st.components.v1.html(f'<div style="background:white; overflow-x:auto; border:1px solid #ddd; padding:10px;">{svg_long}</div>', height=500)

        # ================= TAB 3: REPORT & BOQ =================
        with tab3:
            st.header("📝 Calculation Reports")
            if not final_design_res:
                st.warning("⚠️ Please complete design in Tab 2.")
            else:
                for res in final_design_res:
                    with st.expander(f"📘 Span {res['span_id']+1} Details", expanded=(res['span_id']==0)):
                        reporter.render_calculation_report(res)

            # ================= BOQ SECTION =================
            st.markdown("---")
            st.header("💵 Bill of Quantities (BOQ)")

            c_price1, c_price2, c_price3 = st.columns(3)
            price_conc = c_price1.number_input("Concrete (Baht/m³)", value=2200, step=50)
            price_steel = c_price2.number_input("Rebar (Baht/kg)", value=28.0, step=0.5)
            price_form = c_price3.number_input("Formwork (Baht/m²)", value=300, step=10)

            if final_design_res:
                total_conc_vol = 0.0
                total_form_area = 0.0
                total_steel_weight = 0.0

                for res in final_design_res:
                    L = res['L']
                    b_m = res['b'] / 1000.0
                    h_m = res['h'] / 1000.0
                    
                    # Concrete
                    vol = b_m * h_m * L
                    total_conc_vol += vol
                    
                    # Formwork (2 Sides + Bottom)
                    area = (2*h_m + b_m) * L
                    total_form_area += area
                    
                    # Longitudinal Steel (Add 5% for laps/waste)
                    w_top = sum(get_rebar_weight(l['db']) * l['n'] for l in res['top']['all_layers'])
                    w_bot = sum(get_rebar_weight(l['db']) * l['n'] for l in res['bot']['all_layers'])
                    total_steel_weight += (w_top + w_bot) * L * 1.05 
                    
                    # Stirrups
                    stir_len_m = (2 * (res['b'] + res['h']) / 1000.0) 
                    num_stir = (L * 1000.0) / res['shear']['s'] + 1
                    w_stir = get_rebar_weight(res['shear']['db']) * stir_len_m * num_stir
                    total_steel_weight += w_stir

                boq_data = [
                    {"Item": "Concrete Structure (240 ksc)", "Quantity": total_conc_vol, "Unit": "m³", "Unit Price": price_conc},
                    {"Item": "Deformed Bars (DB) + Stirrups", "Quantity": total_steel_weight, "Unit": "kg", "Unit Price": price_steel},
                    {"Item": "Formwork", "Quantity": total_form_area, "Unit": "m²", "Unit Price": price_form},
                ]
                
                df_boq = pd.DataFrame(boq_data)
                df_boq["Amount (THB)"] = df_boq["Quantity"] * df_boq["Unit Price"]
                
                c_boq1, c_boq2, c_boq3, c_boq4 = st.columns(4)
                c_boq1.metric("Concrete", f"{total_conc_vol:.2f} m³")
                c_boq2.metric("Steel", f"{total_steel_weight:.2f} kg")
                c_boq3.metric("Formwork", f"{total_form_area:.2f} m²")
                c_boq4.metric("TOTAL COST", f"{df_boq['Amount (THB)'].sum():,.0f} ฿", border=True)
                
                st.dataframe(
                    df_boq.style.format({"Quantity": "{:.2f}", "Unit Price": "{:,.2f}", "Amount (THB)": "{:,.2f}"}), 
                    use_container_width=True, hide_index=True
                )

    except Exception as e:
        st.error(f"Error during calculation or rendering: {e}")
        st.write("Please check input parameters.")
