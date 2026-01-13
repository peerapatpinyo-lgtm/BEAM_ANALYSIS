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
    """Calculate weight of rebar per meter (kg/m) based on standard density"""
    # Weight = Area * Density (7850 kg/m3) -> simplified to d^2/162
    return (d_mm ** 2) / 162.0

# --- 3. HELPER: CROSS SECTION PLOT ---
def plot_cross_section_fixed(b, h, cover, top_layers, bot_layers, shear_res):
    fig, ax = plt.subplots(figsize=(5, 6))
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

    text_x = b + (b * 0.1)
    ax.text(text_x, h - cover, f"Top: {n_top}DB{int(dia_top)}", color='#c0392b', fontsize=12, fontweight='bold', va='center')
    ax.text(text_x, cover + dia_bot, f"Bot: {n_bot}DB{int(dia_bot)}", color='#27ae60', fontsize=12, fontweight='bold', va='center')
    ax.text(text_x, h/2, f"Stirrup: RB{int(shear_res['db'])}@{int(shear_res['s'])}", color='#2c3e50', fontsize=10, fontweight='bold', va='center')

    ax.set_title(f"SECTION {int(b)}x{int(h)} mm", fontsize=14, fontweight='bold', pad=20)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-50, b + 250) 
    ax.set_ylim(-50, h + 50)
    plt.tight_layout()
    return fig

# --- 4. MAIN LAYOUT ---
st.markdown('<div class="main-header">🏗️ RC Beam Analysis & Design Pro</div>', unsafe_allow_html=True)

# --- SIDEBAR INPUTS ---
with st.sidebar:
    # params: {'b': mm, 'h': mm, 'fc': ksc, 'fy': ksc ...}
    # raw_user_loads_df: assumed to be in [kN] and [m]
    params, n_spans, spans, sup_df, raw_user_loads_df, stable = input_handler.render_all_sidebar_inputs()

if not stable:
    st.error("🚨 **Structure Error:** Unstable Structure (Check Supports)")
else:
    # --- CONFIG SECTION ---
    col_set1, col_set2 = st.columns([1, 2])
    with col_set1:
        st.markdown("### ⚙️ Settings")
        mode_select = st.radio("Mode:", ["Service (Deflection)", "Ultimate (Design)"], index=1)
        include_sw = st.checkbox("Include Self-weight", value=True)
        
        # --- [FIX 1] EXPLICIT SELF-WEIGHT CALCULATION ---
        # 1. Dimensions (mm -> m)
        b_m = params.get('b', 300) / 1000.0
        h_m = params.get('h', 500) / 1000.0
        
        # 2. Density
        # Concrete Density ~ 2400 kg/m3
        # Gravity = 9.81 m/s2
        # Specific Weight = 2400 * 9.81 = 23544 N/m3 = 23.544 kN/m3
        gamma_concrete_kN_m3 = 23.544 
        
        # 3. Line Load (kN/m)
        sw_val_kN_m = b_m * h_m * gamma_concrete_kN_m3
        
        if include_sw:
            st.info(f"SW = {sw_val_kN_m:.3f} kN/m\n(Based on 2400 kg/m³)")
    
    with col_set2:
        st.markdown("### 🔢 Factors")
        c1, c2 = st.columns(2)
        if "Service" in mode_select:
            f_dl, f_ll = 1.0, 1.0
            tag, is_service = "Service", True
        else:
            f_dl = c1.number_input("Dead Load Factor", 1.4, 1.6, 1.4)
            f_ll = c2.number_input("Live Load Factor", 1.7, 2.0, 1.7)
            tag, is_service = "Ultimate", False

    try:
        # =========================================================
        # [FIX 2] LOAD PREPARATION (STRICT kN UNIT)
        # =========================================================
        
        # 1. Base User Loads (Assume Input is kN)
        combined_loads_df = raw_user_loads_df.copy(deep=True)
        
        # 2. Add Self-Weight (Explicitly kN/m)
        if include_sw:
            sw_list = []
            for i in range(n_spans):
                sw_list.append({
                    'span_index': i, 'type': 'U', 
                    'mag': sw_val_kN_m,  # Value in kN/m
                    'dist': spans[i], 'd_start': 0, 'case': 'DL'
                })
            df_sw = pd.DataFrame(sw_list)
            combined_loads_df = pd.concat([combined_loads_df, df_sw], ignore_index=True)

        # 3. Apply Factors (Result is still kN)
        # rc_load_processor handles DL/LL separation and Factors only. NO UNIT SCALING HERE.
        factored_loads_kN = rc_load_processor.prepare_load_dataframe(
            combined_loads_df, n_spans, spans, params, f_dl, f_ll
        )

        # =========================================================
        # [FIX 3] SOLVER INTERFACE (kN -> N CONVERSION)
        # =========================================================
        
        # Clone for Solver
        solver_input_N = factored_loads_kN.copy()
        
        # CRITICAL: Convert kN -> N for Solver Matrix
        # Solver expects N/m for Distributed and N for Point
        solver_input_N['mag'] = solver_input_N['mag'] * 1000.0
        
        # Call Solver (Input: N, m | Output: N, N-m, m)
        x_res, M_res_Nm, V_res_N, D_res_m, R_res_N = solver.solve_beam(spans, sup_df, solver_input_N, params)

        # =========================================================
        # [FIX 4] POST-PROCESSING (N -> kN CONVERSION)
        # =========================================================
        
        # 1. Arrays for Plotting/Design
        x_plot = x_res
        M_plot_kNm = M_res_Nm / 1000.0     # N-m -> kN-m
        V_plot_kN  = V_res_N  / 1000.0     # N -> kN
        D_plot_mm  = D_res_m  * 1000.0     # m -> mm
        
        # 2. Reactions for Display
        if isinstance(R_res_N, dict):
            R_display_kN = {k: v / 1000.0 for k, v in R_res_N.items()}
        else:
            R_display_kN = [r / 1000.0 for r in R_res_N]

        # 3. Separate Service Run for Deflection/Crack Check (Unfactored)
        # (We need to run solver again with factors 1.0)
        svc_loads_kN = rc_load_processor.prepare_load_dataframe(combined_loads_df, n_spans, spans, params, 1.0, 1.0)
        svc_input_N = svc_loads_kN.copy()
        svc_input_N['mag'] = svc_input_N['mag'] * 1000.0
        x_svc, M_svc_Nm, _, D_svc_m, _ = solver.solve_beam(spans, sup_df, svc_input_N, params)
        
        M_svc_kNm = M_svc_Nm / 1000.0
        D_svc_mm  = D_svc_m * 1000.0

        # Decide what to show in graphs based on user selection
        if is_service:
            plot_M, plot_V, plot_D = M_svc_kNm, V_plot_kN, D_svc_mm # Use Service Moment/Deflection
        else:
            plot_M, plot_V, plot_D = M_plot_kNm, V_plot_kN, D_plot_mm # Use Ultimate Moment

        # --- TABS ---
        tab1, tab2, tab3 = st.tabs(["📊 Analysis", "🏗️ Design", "📝 Report"])

        # TAB 1: ANALYSIS
        with tab1:
            st.subheader(f"Analysis Results ({tag})")
            
            # Prepare Dataframe for Plotting Tool
            df_plot = pd.DataFrame({
                'x': x_plot,
                'moment': plot_M,
                'shear': plot_V,
                'deflection': plot_D
            })
            
            # Use Random Key to force redraw
            key_chart = f"chart_{include_sw}_{tag}_{np.random.randint(0,1000)}"
            
            fig = design_view.plot_analysis_results(
                res_df=df_plot, spans=spans, supports=sup_df, 
                loads=factored_loads_kN if not is_service else svc_loads_kN, # Show Loads in kN
                reactions=R_display_kN
            )
            st.plotly_chart(fig, use_container_width=True, key=key_chart)
            
            # Max Values
            c1, c2, c3 = st.columns(3)
            c1.metric("Max Shear (Vu)", f"{np.max(np.abs(V_plot_kN)):.2f} kN")
            c2.metric("Max Moment (Mu)", f"{np.max(np.abs(M_plot_kNm)):.2f} kNm") # Fixed abs() for max magnitude
            c3.metric("Max Deflection", f"{np.max(np.abs(D_svc_mm)):.2f} mm")

        # TAB 2: DESIGN
        with tab2:
            st.subheader("Reinforcement Design")
            b_mm, h_mm = params['b'], params['h']
            fc, fy = params['fc'], params['fy']
            
            final_design_res = []
            offsets = [0] + list(np.cumsum(spans))

            for i in range(n_spans):
                s_len = spans[i]
                start, end = offsets[i], offsets[i+1]
                
                # Slicing Results for this Span
                mask = (x_plot >= start - 1e-6) & (x_plot <= end + 1e-6)
                if not mask.any(): continue
                
                # Design Forces (Always Ultimate)
                mu_pos = max(0.0, np.max(M_plot_kNm[mask]))
                mu_neg = abs(min(0.0, np.min(M_plot_kNm[mask])))
                vu_max = np.max(np.abs(V_plot_kN[mask]))
                
                # Service Forces (Always Service)
                mask_svc = (x_svc >= start - 1e-6) & (x_svc <= end + 1e-6)
                ma_svc = max(0.0, np.max(M_svc_kNm[mask_svc]))
                delta_svc = np.max(np.abs(D_svc_mm[mask_svc]))

                with st.expander(f"📍 SPAN {i+1} : Mu+ {mu_pos:.1f}, Mu- {mu_neg:.1f} kNm", expanded=True):
                    c_in, c_out = st.columns([1, 1])
                    
                    with c_in:
                        cover = st.number_input(f"Cover (mm) S{i+1}", 20, 50, 25, key=f"c{i}")
                        
                        # -- TOP STEEL --
                        st.caption("🔼 Top Steel (Negative Moment)")
                        n_top = st.number_input(f"Top Bars S{i+1}", 2, 10, 2, key=f"nt{i}")
                        db_top = st.selectbox(f"Top Dia S{i+1}", [12,16,20,25], index=1, key=f"dt{i}")
                        top_layers = [{'n': n_top, 'db': db_top}]
                        
                        # -- BOT STEEL --
                        st.caption("🔽 Bottom Steel (Positive Moment)")
                        n_bot = st.number_input(f"Bot Bars S{i+1}", 2, 10, 3, key=f"nb{i}")
                        db_bot = st.selectbox(f"Bot Dia S{i+1}", [12,16,20,25], index=1, key=f"db{i}")
                        bot_layers = [{'n': n_bot, 'db': db_bot}]
                        
                        # -- STIRRUP --
                        st.caption("🌀 Stirrups")
                        db_stir = st.selectbox(f"Stirrup Dia S{i+1}", [6,9], 0, key=f"ds{i}")
                        s_stir = st.number_input(f"Spacing (mm) S{i+1}", 50, 300, 150, key=f"ss{i}")
                        
                    with c_out:
                        # 1. Check Capacity (Top)
                        d_t = h_mm - cover - db_top/2
                        phi_Mn_t, _, _, _, _, _ = rc_design_engine.get_phi_Mn_details_multi(top_layers, d_t, b_mm, h_mm, fc, fy)
                        st.write(f"**Top Cap:** {phi_Mn_t:.1f} kNm {'✅' if phi_Mn_t >= mu_neg else '❌'}")
                        
                        # 2. Check Capacity (Bot)
                        d_b = h_mm - cover - db_bot/2
                        phi_Mn_b, _, _, _, _, _ = rc_design_engine.get_phi_Mn_details_multi(bot_layers, d_b, b_mm, h_mm, fc, fy)
                        st.write(f"**Bot Cap:** {phi_Mn_b:.1f} kNm {'✅' if phi_Mn_b >= mu_pos else '❌'}")
                        
                        # 3. Check Shear
                        _, phi_Vn, _, _, _, _ = rc_design_engine.check_shear_details(vu_max, b_mm, d_b, fc, fy, db_stir, s_stir)
                        st.write(f"**Shear Cap:** {phi_Vn:.1f} kN {'✅' if phi_Vn >= vu_max else '❌ (Add stirrups)'}")
                        
                        # 4. Draw Section
                        fig_sec = plot_cross_section_fixed(b_mm, h_mm, cover, top_layers, bot_layers, {'db': db_stir, 's': s_stir})
                        st.pyplot(fig_sec)
                        
                    # Save for Report
                    final_design_res.append({
                        'span_id': i, 'L': s_len, 'b': b_mm, 'h': h_mm, 'fc': fc, 'fy': fy,
                        'Mu_pos': mu_pos, 'Mu_neg': mu_neg, 'Vu_max': vu_max,
                        'top': {'all_layers': top_layers}, 'bot': {'all_layers': bot_layers},
                        'shear': {'db': db_stir, 's': s_stir},
                        'service': {'delta_long': delta_svc} # Simplified for display
                    })

        # TAB 3: BOQ
        with tab3:
            st.subheader("Bill of Quantities")
            if final_design_res:
                vol_conc, w_steel, area_form = 0, 0, 0
                for r in final_design_res:
                    # Concrete
                    vol_conc += (r['b']/1000 * r['h']/1000 * r['L'])
                    # Formwork
                    area_form += (r['b']/1000 + 2*r['h']/1000) * r['L']
                    # Steel
                    w_t = sum(get_rebar_weight(l['db']) * l['n'] for l in r['top']['all_layers']) * r['L'] * 1.05
                    w_b = sum(get_rebar_weight(l['db']) * l['n'] for l in r['bot']['all_layers']) * r['L'] * 1.05
                    # Stirrup approx
                    len_stir = 2*(r['b']+r['h'])/1000
                    n_stir = (r['L']*1000 / r['shear']['s']) + 1
                    w_s = get_rebar_weight(r['shear']['db']) * len_stir * n_stir
                    w_steel += (w_t + w_b + w_s)
                
                # Prices
                p_conc = 2200
                p_steel = 26
                p_form = 300
                
                cost_conc = vol_conc * p_conc
                cost_steel = w_steel * p_steel
                cost_form = area_form * p_form
                
                st.dataframe(pd.DataFrame([
                    {"Item": "Concrete", "Qty": f"{vol_conc:.2f} m3", "Unit Price": p_conc, "Amount": cost_conc},
                    {"Item": "Steel", "Qty": f"{w_steel:.2f} kg", "Unit Price": p_steel, "Amount": cost_steel},
                    {"Item": "Formwork", "Qty": f"{area_form:.2f} m2", "Unit Price": p_form, "Amount": cost_form},
                ]))
                st.success(f"**Total Cost: {cost_conc + cost_steel + cost_form:,.2f} THB**")

    except Exception as e:
        st.error(f"Calculation Error: {str(e)}")
