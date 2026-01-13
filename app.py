import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import gridspec

# --- 1. IMPORT CUSTOM MODULES ---
import input_handler, solver, section_plotter, reporter
import rc_utils, rc_design_engine, rc_load_processor, app_styles

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(page_title="Pro RC Beam Design", layout="wide", page_icon="🏗️")
app_styles.apply_custom_css()

# --- 3. HELPER FUNCTIONS ---
def get_rebar_weight(d_mm):
    return (d_mm ** 2) / 162.0

# =========================================================================
# 🛑 ORIGINAL PLOTTING ENGINE (RESTORED & UNITS FIXED)
# ใช้ Matplotlib เหมือนเดิม แต่ปรับหน่วยให้แสดงผลถูกต้อง (kN, kNm, mm)
# =========================================================================
def plot_analysis_results_original_style(res_df, spans, supports, loads, reactions):
    # res_df เข้ามาต้องเป็นหน่วย Engineering แล้ว (kN, kNm, mm)
    
    # 1. Setup Figure
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 1], hspace=0.5)
    
    # --- LOAD DIAGRAM ---
    ax0 = plt.subplot(gs[0])
    total_len = sum(spans)
    ax0.plot([0, total_len], [0, 0], 'k-', linewidth=3)
    
    # Supports
    sup_x = 0
    for i, s_type in enumerate(supports['type']):
        ax0.plot(sup_x, 0, marker='^', markersize=14, color='black', markeredgecolor='black')
        r_val = reactions.get(f"R{i}", 0.0)
        ax0.text(sup_x, -0.5, f"R{i}={r_val:.2f} kN", ha='center', va='top', fontsize=9, color='green', fontweight='bold')
        if i < len(spans): sup_x += spans[i]

    # Loads
    max_mag = 1.0
    if not loads.empty:
        max_mag = loads['mag'].abs().max() if loads['mag'].abs().max() > 0 else 1.0
        
    for _, load in loads.iterrows():
        if load['type'] == 'P':
            x = load['d_start']
            mag = load['mag']
            dy = -1.0 if mag > 0 else 1.0
            ax0.arrow(x, dy, 0, -dy*0.7, head_width=0.15, head_length=0.2, fc='red', ec='red')
            ax0.text(x, dy, f"{mag:.2f} kN", ha='center', va='bottom' if mag>0 else 'top', color='red', fontsize=9)
        elif load['type'] == 'U':
            x1 = load['d_start']
            x2 = x1 + load['dist']
            mag = load['mag']
            ax0.fill_between([x1, x2], [0, 0], [mag/max_mag, mag/max_mag], color='blue', alpha=0.3)
            ax0.text((x1+x2)/2, mag/max_mag, f"{mag:.2f} kN/m", ha='center', va='bottom', color='blue', fontsize=9)

    ax0.set_title("Load Diagram", fontsize=10, fontweight='bold')
    ax0.set_ylim(-2, 2)
    ax0.axis('off')

    # --- SHEAR DIAGRAM ---
    ax1 = plt.subplot(gs[1], sharex=ax0)
    ax1.plot(res_df['x'], res_df['shear'], 'b-', linewidth=1.5)
    ax1.fill_between(res_df['x'], res_df['shear'], 0, color='blue', alpha=0.1)
    ax1.set_ylabel("Shear (kN)", fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_title("Shear Force Diagram", fontsize=10, fontweight='bold')
    # Max values
    v_max = res_df['shear'].max()
    v_min = res_df['shear'].min()
    ax1.text(res_df.loc[res_df['shear'].idxmax(), 'x'], v_max, f"{v_max:.2f}", color='blue', fontsize=8)
    ax1.text(res_df.loc[res_df['shear'].idxmin(), 'x'], v_min, f"{v_min:.2f}", color='blue', fontsize=8)

    # --- MOMENT DIAGRAM ---
    ax2 = plt.subplot(gs[2], sharex=ax0)
    ax2.plot(res_df['x'], res_df['moment'], 'r-', linewidth=1.5)
    ax2.fill_between(res_df['x'], res_df['moment'], 0, color='red', alpha=0.1)
    ax2.set_ylabel("Moment (kNm)", fontsize=9)
    ax2.invert_yaxis()
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.set_title("Bending Moment Diagram", fontsize=10, fontweight='bold')
    # Max values
    m_max = res_df['moment'].max()
    m_min = res_df['moment'].min()
    ax2.text(res_df.loc[res_df['moment'].idxmax(), 'x'], m_max, f"{m_max:.2f}", color='red', fontsize=8)
    ax2.text(res_df.loc[res_df['moment'].idxmin(), 'x'], m_min, f"{m_min:.2f}", color='red', fontsize=8)

    # --- DEFLECTION DIAGRAM ---
    ax3 = plt.subplot(gs[3], sharex=ax0)
    ax3.plot(res_df['x'], res_df['deflection'], 'm-', linewidth=1.5)
    ax3.set_ylabel("Deflection (mm)", fontsize=9)
    ax3.set_xlabel("Distance (m)", fontsize=9)
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.set_title("Deflection", fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig

# --- CROSS SECTION PLOT ---
def plot_cross_section_original(b, h, cover, top_layers, bot_layers, shear_res):
    fig, ax = plt.subplots(figsize=(4, 5))
    rect = patches.Rectangle((0, 0), b, h, linewidth=2, edgecolor='black', facecolor='white')
    ax.add_patch(rect)
    stirrup_rect = patches.Rectangle((cover, cover), b - 2*cover, h - 2*cover, linewidth=1, edgecolor='black', facecolor='none', linestyle='--')
    ax.add_patch(stirrup_rect)
    
    # Draw bars
    def draw_bars(layers, y_pos_func, color):
        dia = layers[0]['db'] if layers else 12
        n = sum(l['n'] for l in layers)
        if n > 0:
            start_x = cover + dia/2
            gap = (b - 2*cover - dia) / (n - 1) if n > 1 else 0
            for i in range(n):
                cx = start_x + i*gap if n > 1 else b/2
                cy = y_pos_func(dia)
                ax.add_patch(patches.Circle((cx, cy), radius=dia/2, color=color))

    draw_bars(top_layers, lambda d: h - cover - d/2, 'red')
    draw_bars(bot_layers, lambda d: cover + d/2, 'blue')

    ax.set_xlim(-50, b + 50)
    ax.set_ylim(-50, h + 50)
    ax.axis('equal')
    ax.axis('off')
    return fig

# --- 4. MAIN APP LOGIC ---
st.markdown('<div class="main-header">🏗️ RC Beam Analysis & Design Pro</div>', unsafe_allow_html=True)

with st.sidebar:
    params, n_spans, spans, sup_df, raw_user_loads_df, stable = input_handler.render_all_sidebar_inputs()
    
    b_val = params.get('b', 300)
    h_val = params.get('h', 500)
    unit_w_conc = 2400 * 9.81 / 1000 
    sw_calc_val = (b_val / 1000) * (h_val / 1000) * unit_w_conc 

if not stable:
    st.error("🚨 **Structure Error:** โครงสร้างไม่เสถียร!")
else:
    col_set1, col_set2 = st.columns([1, 2])
    with col_set1:
        st.markdown("### ⚙️ Analysis Settings")
        mode_select = st.radio("Design Mode:", ["Service Load (Check Deflection)", "Ultimate Strength (Design)"], index=1)
        st.markdown("---")
        include_sw = st.checkbox("➕ Include Beam Self-weight", value=True)
        if include_sw: st.info(f"ℹ️ **SW:** {sw_calc_val:.2f} kN/m")
    
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
        params.update({'dl_factor': f_dl, 'll_factor': f_ll, 'include_sw': include_sw})

    try:
        # --- DATA PREP ---
        clean_user_loads = raw_user_loads_df.copy(deep=True)
        if not clean_user_loads.empty:
            # Assume user input < 2000 is kN -> convert to N
            clean_user_loads['mag'] = clean_user_loads['mag'].apply(lambda x: x * 1000.0 if abs(x) < 2000.0 else x)
        
        if include_sw:
            sw_mag_newton = sw_calc_val * 1000.0 
            sw_rows = [{'span_index': i, 'type': 'U', 'mag': sw_mag_newton, 'dist': spans[i], 'd_start': 0, 'case': 'DL'} for i in range(n_spans)]
            final_calc_loads = pd.concat([clean_user_loads, pd.DataFrame(sw_rows)], ignore_index=True)
        else:
            final_calc_loads = clean_user_loads

        # --- SOLVER ---
        # 1. Ultimate
        calc_loads_ult = rc_load_processor.prepare_load_dataframe(final_calc_loads, n_spans, spans, params, f_dl, f_ll)
        x_ult, M_ult, V_ult, D_ult, R_ult = solver.solve_beam(spans, sup_df, calc_loads_ult, params)
        # 2. Service
        calc_loads_svc = rc_load_processor.prepare_load_dataframe(final_calc_loads, n_spans, spans, params, 1.0, 1.0)
        x_svc, M_svc, V_svc, D_svc, R_svc = solver.solve_beam(spans, sup_df, calc_loads_svc, params)

        # Select display data
        x_raw, M_raw, V_raw, D_raw, R_raw = (x_svc, M_svc, V_svc, D_svc, R_svc) if is_service else (x_ult, M_ult, V_ult, D_ult, R_ult)
        current_loads_raw = calc_loads_svc if is_service else calc_loads_ult

        tab1, tab2, tab3 = st.tabs(["📊 1. Analysis Results", "📝 2. Concrete Design", "📘 3. Report & BOQ"])
        final_design_res = []

        with tab1:
            st.subheader(f"📈 Analysis Diagrams ({tag})")
            
            # --- PREPARE DATA FOR PLOTTING (CONVERT UNITS HERE) ---
            df_plot = pd.DataFrame({
                'x': x_raw,
                'moment': M_raw / 1000.0,    # N-m -> kNm
                'shear': V_raw / 1000.0,     # N -> kN
                'deflection': D_raw * 1000.0 # m -> mm
            })
            loads_plot = current_loads_raw.copy(deep=True)
            if not loads_plot.empty:
                loads_plot['mag'] = loads_plot['mag'] / 1000.0
            reactions_plot = {k: v / 1000.0 for k, v in R_raw.items()}

            # --- CALL MATPLOTLIB FUNCTION ---
            fig = plot_analysis_results_original_style(df_plot, spans, sup_df, loads_plot, reactions_plot)
            st.pyplot(fig)

            # Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Max Shear", f"{max(abs(df_plot['shear'])):.2f} kN")
            c2.metric("Max Moment", f"{max(abs(df_plot['moment'])):.2f} kNm")
            c3.metric("Max Deflection", f"{max(abs(df_plot['deflection'])):.2f} mm")

        with tab2:
            st.header("🏗️ Reinforcement Detailing")
            b_mm, h_mm = rc_utils.normalize_section_units(params['b'], params['h'])
            fc, fy = params['fc'], params['fy']
            offsets = [0] + list(np.cumsum(spans))
            
            for i in range(n_spans):
                s_len, s_start, s_end = spans[i], offsets[i], offsets[i+1]
                mask_u = (x_ult >= s_start) & (x_ult <= s_end)
                if not mask_u.any(): continue

                mu_pos = max(0.0, (M_ult[mask_u]/1000.0).max())
                mu_neg = abs(min(0.0, (M_ult[mask_u]/1000.0).min()))
                vu_max = abs((V_ult[mask_u]/1000.0)).max()
                
                mask_s = (x_svc >= s_start) & (x_svc <= s_end)
                ma_svc = max(0.0, (M_svc[mask_s]/1000.0).max())
                delta_svc = abs(D_svc[mask_s]).max() * 1000.0

                with st.expander(f"📍 SPAN {i+1} (L={s_len} m)", expanded=True):
                    c_in, c_plt = st.columns([2, 1])
                    with c_in:
                        cover = st.number_input(f"Cover (mm)", 20, 50, 25, key=f"c_{i}")
                        
                        st.markdown("##### Top Bars")
                        n_top = st.number_input(f"Top No.", 2, 10, 2, key=f"nt_{i}")
                        db_top = st.selectbox(f"Top Dia", [12, 16, 20, 25], index=1, key=f"dt_{i}")
                        top_layers = [{'n': n_top, 'db': db_top}]
                        
                        st.markdown("##### Bottom Bars")
                        n_bot = st.number_input(f"Bot No.", 2, 10, 3, key=f"nb_{i}")
                        db_bot = st.selectbox(f"Bot Dia", [12, 16, 20, 25], index=1, key=f"db_{i}")
                        bot_layers = [{'n': n_bot, 'db': db_bot}]
                        
                        # Calcs
                        d_eff_t, as_t, _ = rc_design_engine.get_centroid_and_d(top_layers, h_mm, cover, 9)
                        d_eff_b, as_b, _ = rc_design_engine.get_centroid_and_d(bot_layers, h_mm, cover, 9)
                        
                        phi_Mn_t, _, _, _, _, _ = rc_design_engine.get_phi_Mn_details_multi(top_layers, h_mm-cover-db_top/2, b_mm, h_mm, fc, fy)
                        phi_Mn_b, _, _, _, _, _ = rc_design_engine.get_phi_Mn_details_multi(bot_layers, d_eff_b, b_mm, h_mm, fc, fy)

                        st.write(f"**Neg M:** Req {mu_neg:.2f} | Cap {phi_Mn_t:.2f} kNm {'✅' if phi_Mn_t>=mu_neg else '❌'}")
                        st.write(f"**Pos M:** Req {mu_pos:.2f} | Cap {phi_Mn_b:.2f} kNm {'✅' if phi_Mn_b>=mu_pos else '❌'}")
                        
                        # Shear
                        st.markdown("##### Shear")
                        s_db = st.selectbox("Stirrup", [6, 9], key=f"sdb_{i}")
                        s_sp = st.number_input("Space (mm)", 50, 300, 150, key=f"ssp_{i}")
                        res_v, phi_vn, _, _, _, _ = rc_design_engine.check_shear_details(vu_max, b_mm, d_eff_b, fc, fy, s_db, s_sp)
                        st.write(f"**Shear:** Req {vu_max:.2f} | Cap {phi_vn:.2f} kN {'✅' if phi_vn>=vu_max else '❌'}")

                    with c_plt:
                        fig_cs = plot_cross_section_original(b_mm, h_mm, cover, top_layers, bot_layers, {'db': s_db, 's': s_sp})
                        st.pyplot(fig_cs)

                    final_design_res.append({
                        'span_id': i, 'L': s_len, 'b': b_mm, 'h': h_mm, 
                        'top': {'all_layers': top_layers}, 'bot': {'all_layers': bot_layers},
                        'shear': {'db': s_db, 's': s_sp}
                    })

        with tab3:
            st.header("💵 Bill of Quantities")
            c1, c2, c3 = st.columns(3)
            p_conc = c1.number_input("Concrete (B/m3)", 2000, 3000, 2400)
            p_steel = c2.number_input("Steel (B/kg)", 20.0, 40.0, 28.0)
            p_form = c3.number_input("Formwork (B/m2)", 200, 500, 300)

            if final_design_res:
                q_conc, q_steel, q_form = 0, 0, 0
                for r in final_design_res:
                    L = r['L']
                    q_conc += (r['b']/1000 * r['h']/1000 * L)
                    q_form += (2*r['h']/1000 + r['b']/1000) * L
                    
                    w_t = sum(get_rebar_weight(l['db']) * l['n'] for l in r['top']['all_layers'])
                    w_b = sum(get_rebar_weight(l['db']) * l['n'] for l in r['bot']['all_layers'])
                    q_steel += (w_t + w_b) * L * 1.05
                    
                    # Stirrups estimate
                    len_stir = 2*(r['b']+r['h'])/1000
                    n_stir = (L*1000 / r['shear']['s']) + 1
                    q_steel += get_rebar_weight(r['shear']['db']) * len_stir * n_stir

                data = [
                    {"Item": "Concrete", "Qty": q_conc, "Unit": "m3", "Rate": p_conc, "Amount": q_conc*p_conc},
                    {"Item": "Rebar", "Qty": q_steel, "Unit": "kg", "Rate": p_steel, "Amount": q_steel*p_steel},
                    {"Item": "Formwork", "Qty": q_form, "Unit": "m2", "Rate": p_form, "Amount": q_form*p_form}
                ]
                df_boq = pd.DataFrame(data)
                st.metric("Total Cost", f"{df_boq['Amount'].sum():,.2f} THB")
                
                # --- FIX: Apply formatting only to specific columns ---
                st.dataframe(df_boq.style.format({
                    "Qty": "{:.2f}",
                    "Rate": "{:.2f}",
                    "Amount": "{:,.2f}"
                }), use_container_width=True)

    except Exception as e:
        st.error(f"System Error: {e}")
