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
# 🛑 CORRECTED PLOTTING ENGINE (MATPLOTLIB - CLASSIC STYLE)
# วาดกราฟด้วย Matplotlib เพื่อให้เหมือนเดิม แต่แก้หน่วยให้ถูกต้อง
# =========================================================================
def plot_analysis_results_matplotlib(res_df, spans, supports, loads, reactions):
    """
    res_df: DataFrame (x, moment, shear, deflection) -> หน่วยต้องแปลงมาแล้ว (kN, kNm, mm)
    loads: DataFrame -> หน่วยต้องแปลงมาแล้ว (kN, kN/m)
    reactions: Dict -> หน่วยต้องแปลงมาแล้ว (kN)
    """
    # Create Figure
    fig = plt.figure(figsize=(10, 12))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 1.2], hspace=0.4)
    
    # 1. LOAD DIAGRAM
    ax0 = plt.subplot(gs[0])
    total_len = sum(spans)
    ax0.plot([0, total_len], [0, 0], 'k-', linewidth=3) # Beam line
    
    # Supports
    sup_x = 0
    for i, s_type in enumerate(supports['type']):
        ax0.plot(sup_x, 0, marker='^', markersize=12, color='black', markeredgecolor='black')
        # Reaction Text
        r_val = reactions.get(f"R{i}", 0.0)
        ax0.text(sup_x, -0.5, f"R{i}={r_val:.2f} kN", ha='center', va='top', fontsize=10, color='green', fontweight='bold')
        if i < len(spans): sup_x += spans[i]

    # Loads
    max_mag = 1.0
    if not loads.empty:
        max_mag = loads['mag'].abs().max() if loads['mag'].abs().max() > 0 else 1.0
        scale = 2.0 / max_mag # Scale factor for arrows
        
        for _, load in loads.iterrows():
            if load['type'] == 'P': # Point Load
                x = load['d_start']
                mag = load['mag']
                dy = -1.5 if mag > 0 else 1.5
                ax0.arrow(x, dy, 0, -dy*0.8, head_width=0.2, head_length=0.3, fc='red', ec='red')
                ax0.text(x, dy, f"{mag:.2f} kN", ha='center', va='bottom' if mag>0 else 'top', color='red')
            elif load['type'] == 'U': # Uniform Load
                x1 = load['d_start']
                x2 = x1 + load['dist']
                mag = load['mag']
                ax0.fill_between([x1, x2], [0, 0], [mag/max_mag, mag/max_mag], color='blue', alpha=0.3)
                ax0.text((x1+x2)/2, mag/max_mag, f"{mag:.2f} kN/m", ha='center', va='bottom', color='blue')

    ax0.set_title("Load Diagram (kN, kN/m)", fontsize=11, fontweight='bold')
    ax0.set_ylim(-2.5, 2.5)
    ax0.axis('off')

    # 2. SHEAR DIAGRAM (kN)
    ax1 = plt.subplot(gs[1], sharex=ax0)
    ax1.plot(res_df['x'], res_df['shear'], 'b-', linewidth=1.5)
    ax1.fill_between(res_df['x'], res_df['shear'], 0, color='blue', alpha=0.1)
    ax1.set_ylabel("Shear (kN)", fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    # Annotate Max Shear
    v_max = res_df['shear'].max()
    v_min = res_df['shear'].min()
    ax1.text(0, v_max, f"{v_max:.2f}", color='blue', fontsize=9)
    ax1.text(0, v_min, f"{v_min:.2f}", color='blue', fontsize=9)

    # 3. MOMENT DIAGRAM (kNm)
    ax2 = plt.subplot(gs[2], sharex=ax0)
    ax2.plot(res_df['x'], res_df['moment'], 'r-', linewidth=1.5)
    ax2.fill_between(res_df['x'], res_df['moment'], 0, color='red', alpha=0.1)
    ax2.set_ylabel("Moment (kNm)", fontsize=10)
    ax2.invert_yaxis() # Flip for RC convention
    ax2.grid(True, linestyle=':', alpha=0.6)
    # Annotate Max Moment
    m_max = res_df['moment'].max()
    m_min = res_df['moment'].min()
    ax2.text(total_len/2, m_max, f"{m_max:.2f}", color='red', fontsize=9)
    ax2.text(total_len/2, m_min, f"{m_min:.2f}", color='red', fontsize=9)

    # 4. DEFLECTION DIAGRAM (mm)
    ax3 = plt.subplot(gs[3], sharex=ax0)
    ax3.plot(res_df['x'], res_df['deflection'], 'm-', linewidth=1.5)
    ax3.set_ylabel("Deflection (mm)", fontsize=10) # Fixed Unit Label
    ax3.set_xlabel("Distance (m)", fontsize=10)
    ax3.grid(True, linestyle=':', alpha=0.6)
    # Annotate Max Deflection
    d_max_abs = res_df['deflection'].abs().max()
    ax3.text(total_len/2, -d_max_abs, f"Max: {d_max_abs:.2f} mm", color='purple', fontsize=9)

    plt.tight_layout()
    return fig

def plot_cross_section_fixed(b, h, cover, top_layers, bot_layers, shear_res):
    fig, ax = plt.subplots(figsize=(4, 5))
    # Main Concrete Rect
    rect = patches.Rectangle((0, 0), b, h, linewidth=2, edgecolor='black', facecolor='white')
    ax.add_patch(rect)
    # Stirrup
    stirrup_rect = patches.Rectangle((cover, cover), b - 2*cover, h - 2*cover, 
                                     linewidth=1.5, edgecolor='#34495e', facecolor='none', linestyle='-')
    ax.add_patch(stirrup_rect)
    
    # Top Bars
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

    # Bot Bars
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

    # Text
    text_x = b + (b * 0.15)
    ax.text(text_x, h - cover, f"Top: {n_top}DB{int(dia_top)}", color='#c0392b', fontsize=11, fontweight='bold', va='center')
    ax.text(text_x, cover + dia_bot, f"Bot: {n_bot}DB{int(dia_bot)}", color='#27ae60', fontsize=11, fontweight='bold', va='center')
    ax.text(text_x, h/2, f"Stir: RB{int(shear_res['db'])}@{int(shear_res['s'])}", color='#2c3e50', fontsize=10, fontweight='bold', va='center')

    ax.set_title(f"SECTION {int(b)}x{int(h)} mm", fontsize=12, fontweight='bold', pad=15)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-50, b + 250) 
    ax.set_ylim(-50, h + 50)
    plt.tight_layout()
    return fig

# --- 4. MAIN APP LOGIC ---
st.markdown('<div class="main-header">🏗️ RC Beam Analysis & Design Pro (Fixed)</div>', unsafe_allow_html=True)

with st.sidebar:
    params, n_spans, spans, sup_df, raw_user_loads_df, stable = input_handler.render_all_sidebar_inputs()
    
    # Calculate SW (kN/m)
    b_val = params.get('b', 300)
    h_val = params.get('h', 500)
    unit_w_conc = 2400 * 9.81 / 1000 # kg -> N -> kN
    sw_calc_val = (b_val / 1000) * (h_val / 1000) * unit_w_conc # kN/m

if not stable:
    st.error("🚨 **Structure Error:** โครงสร้างไม่เสถียร!")
else:
    col_set1, col_set2 = st.columns([1, 2])
    with col_set1:
        st.markdown("### ⚙️ Analysis Settings")
        mode_select = st.radio("Design Mode:", ["Service Load (Check Deflection)", "Ultimate Strength (Design)"], index=1)
        st.markdown("---")
        include_sw = st.checkbox("➕ Include Beam Self-weight", value=True)
        if include_sw:
            st.info(f"ℹ️ **SW:** {sw_calc_val:.2f} kN/m")
    
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
        # --- DATA PREP (kN -> N for Solver) ---
        clean_user_loads = raw_user_loads_df.copy(deep=True)
        if not clean_user_loads.empty:
            clean_user_loads['mag'] = clean_user_loads['mag'].apply(lambda x: x * 1000.0 if abs(x) < 2000.0 else x)
        
        sw_rows = []
        if include_sw:
            sw_mag_newton = sw_calc_val * 1000.0 
            for i in range(n_spans):
                sw_rows.append({'span_index': i, 'type': 'U', 'mag': sw_mag_newton, 'dist': spans[i], 'd_start': 0, 'case': 'DL'})
            df_sw_only = pd.DataFrame(sw_rows)
            final_calc_loads = pd.concat([clean_user_loads, df_sw_only], ignore_index=True)
        else:
            final_calc_loads = clean_user_loads

        # --- RUN SOLVER ---
        # 1. Ultimate
        calc_loads_ult = rc_load_processor.prepare_load_dataframe(final_calc_loads, n_spans, spans, params, f_dl, f_ll)
        x_ult, M_ult, V_ult, D_ult, R_ult = solver.solve_beam(spans, sup_df, calc_loads_ult, params)
        
        # 2. Service
        calc_loads_svc = rc_load_processor.prepare_load_dataframe(final_calc_loads, n_spans, spans, params, 1.0, 1.0)
        x_svc, M_svc, V_svc, D_svc, R_svc = solver.solve_beam(spans, sup_df, calc_loads_svc, params)

        # Select Data
        x_raw, M_raw, V_raw, D_raw, R_raw = (x_svc, M_svc, V_svc, D_svc, R_svc) if is_service else (x_ult, M_ult, V_ult, D_ult, R_ult)
        current_loads_raw = calc_loads_svc if is_service else calc_loads_ult

        tab1, tab2, tab3 = st.tabs(["📊 1. Analysis Results", "📝 2. Concrete Design", "📘 3. Report & BOQ"])
        final_design_res = []

        with tab1:
            st.subheader(f"📈 Analysis Diagrams ({tag})")
            
            # --- UNIT CONVERSION FOR PLOT ---
            df_plot = pd.DataFrame({
                'x': x_raw,
                'moment': M_raw / 1000.0,    # kNm
                'shear': V_raw / 1000.0,     # kN
                'deflection': D_raw * 1000.0 # mm
            })
            loads_plot = current_loads_raw.copy(deep=True)
            if not loads_plot.empty:
                loads_plot['mag'] = loads_plot['mag'] / 1000.0
            reactions_plot = {k: v / 1000.0 for k, v in R_raw.items()}

            # --- PLOT (Matplotlib) ---
            fig = plot_analysis_results_matplotlib(df_plot, spans, sup_df, loads_plot, reactions_plot)
            st.pyplot(fig)

            # --- METRICS ---
            c_m1, c_m2, c_m3 = st.columns(3)
            c_m1.metric("Max Shear (Vu)", f"{max(abs(df_plot['shear'])):.2f} kN")
            c_m2.metric("Max Moment (Mu)", f"{max(abs(df_plot['moment'])):.2f} kNm")
            c_m3.metric("Max Deflection (Δ)", f"{max(abs(df_plot['deflection'])):.2f} mm")

            st.markdown("#### 🏗️ Support Reactions")
            r_data = [{"Support": k, "Reaction (kN)": f"{v:.2f}"} for k, v in reactions_plot.items()]
            st.dataframe(pd.DataFrame(r_data), use_container_width=True, hide_index=True)

        with tab2:
            st.header("🏗️ Reinforcement Detailing")
            b_mm, h_mm = rc_utils.normalize_section_units(params['b'], params['h'])
            fc, fy = params['fc'], params['fy']
            offsets = [0] + list(np.cumsum(spans))
            
            for i in range(n_spans):
                s_len, s_start, s_end = spans[i], offsets[i], offsets[i+1]
                mask_u = (x_ult >= s_start - 1e-6) & (x_ult <= s_end + 1e-6)
                if not mask_u.any(): continue

                mu_pos_knm = max(0.0, (M_ult[mask_u]/1000.0).max())
                mu_neg_knm = abs(min(0.0, (M_ult[mask_u]/1000.0).min()))
                vu_max_kn = abs((V_ult[mask_u] / 1000.0)).max()

                mask_s = (x_svc >= s_start - 1e-6) & (x_svc <= s_end + 1e-6)
                ma_pos_svc_knm = max(0.0, (M_svc[mask_s]/1000.0).max())
                delta_elastic_mm = abs(D_svc[mask_s]).max() * 1000.0

                with st.expander(f"📍 SPAN {i+1} (L={s_len} m)", expanded=True):
                    col_input, col_draw = st.columns([2, 1])
                    with col_input:
                        cover_mm = st.number_input(f"Cover (mm)", 20, 50, 25, key=f"cov_{i}")

                        # Top Steel
                        st.markdown("#### 🔼 Top Reinforcement")
                        num_t_layers = st.selectbox("Top Layers", [1, 2, 3], index=0, key=f"tl_qty_{i}")
                        top_layers = []
                        for l_idx in range(num_t_layers):
                            ct1, ct2 = st.columns(2)
                            with ct1: t_db = st.selectbox(f"L{l_idx+1} Dia", [12, 16, 20, 25, 28], index=1, key=f"tdb_{i}_{l_idx}")
                            with ct2: t_qty = st.number_input(f"L{l_idx+1} No.", 0, 20, 2 if l_idx==0 else 0, key=f"tn_{i}_{l_idx}")
                            top_layers.append({'n': t_qty, 'db': t_db})
                        
                        d_t_val, as_prov_t, y_centroid_t = rc_design_engine.get_centroid_and_d(top_layers, h_mm, cover_mm, 9)
                        d_t = h_mm - y_centroid_t if y_centroid_t > 0 else h_mm - (cover_mm + 9 + 16/2)
                        phi_Mn_t, _, _, _, _, _ = rc_design_engine.get_phi_Mn_details_multi(top_layers, d_t, b_mm, h_mm, fc, fy)
                        st.markdown(f"req $M_u^-$: **{mu_neg_knm:.2f}** kNm | cap: **{phi_Mn_t:.2f}** kNm {'✅' if phi_Mn_t >= mu_neg_knm else '❌'}")

                        # Bottom Steel
                        st.markdown("#### 🔽 Bottom Reinforcement")
                        num_b_layers = st.selectbox("Bottom Layers", [1, 2, 3], index=0, key=f"bl_qty_{i}")
                        bot_layers = []
                        for l_idx in range(num_b_layers):
                            cb1, cb2 = st.columns(2)
                            with cb1: b_db = st.selectbox(f"L{l_idx+1} Dia", [12, 16, 20, 25, 28], index=1, key=f"bdb_{i}_{l_idx}")
                            with cb2: b_qty = st.number_input(f"L{l_idx+1} No.", 0, 20, 3 if l_idx==0 else 0, key=f"bn_{i}_{l_idx}")
                            bot_layers.append({'n': b_qty, 'db': b_db})
                        
                        d_b, as_prov_b, _ = rc_design_engine.get_centroid_and_d(bot_layers, h_mm, cover_mm, 9)
                        if d_b <= 0: d_b = h_mm - (cover_mm + 9 + 16/2)
                        phi_Mn_b, _, _, _, _, _ = rc_design_engine.get_phi_Mn_details_multi(bot_layers, d_b, b_mm, h_mm, fc, fy)
                        st.markdown(f"req $M_u^+$: **{mu_pos_knm:.2f}** kNm | cap: **{phi_Mn_b:.2f}** kNm {'✅' if phi_Mn_b >= mu_pos_knm else '❌'}")

                        # Shear
                        st.markdown("#### 🌀 Shear Stirrups")
                        cs1, cs2 = st.columns(2)
                        with cs1: stir_db = st.selectbox("Stirrup Dia", [6, 9, 12], index=1, key=f"sdb_final_{i}")
                        with cs2: stir_s = st.number_input("Spacing @ (mm)", 50, 300, 150, key=f"ss_{i}")
                        status_v, phi_Vn, _, _, _, _ = rc_design_engine.check_shear_details(vu_max_kn, b_mm, d_b, fc, fy, stir_db, stir_s)
                        if phi_Vn < vu_max_kn: st.error(f"❌ Shear Fail: {phi_Vn:.1f} < {vu_max_kn:.1f} kN")
                        else: st.success(f"✅ Shear OK: {phi_Vn:.1f} ≥ {vu_max_kn:.1f} kN")

                        # Checks
                        st.markdown("---")
                        d_inst, d_long, Ie, Icr, lambda_d = rc_design_engine.check_serviceability(ma_pos_svc_knm, delta_elastic_mm, b_mm, h_mm, d_b, as_prov_b, as_prov_t, fc)
                        limit_240 = (s_len * 1000) / 240
                        total_n_bars_bot = sum(l['n'] for l in bot_layers)
                        w_crack, fs_actual = rc_design_engine.check_crack_width(Ma_svc=ma_pos_svc_knm, b=b_mm, h=h_mm, d=d_b, As=as_prov_b, n_bars=total_n_bars_bot, fc=fc)
                        
                        col_chk1, col_chk2 = st.columns(2)
                        with col_chk1: st.metric("Deflection (L/240)", f"{d_long:.2f} mm", f"{'Pass' if d_long <= limit_240 else 'Fail'}")
                        with col_chk2: st.metric("Crack Width (0.3mm)", f"{w_crack:.3f} mm", f"{'Pass' if w_crack <= 0.3 else 'Warning'}")

                    with col_draw:
                        fig_cs = plot_cross_section_fixed(b=b_mm, h=h_mm, cover=cover_mm, top_layers=top_layers, bot_layers=bot_layers, shear_res={'db': stir_db, 's': stir_s})
                        st.pyplot(fig_cs)
                        plt.close(fig_cs)

                    final_design_res.append({
                        'span_id': i, 'L': s_len, 'b': b_mm, 'h': h_mm, 'fc': fc, 'fy': fy, 
                        'Mu_pos': mu_pos_knm, 'Mu_neg': mu_neg_knm, 'Vu_max': vu_max_kn, 'cover': cover_mm,
                        'Ma_pos_svc': ma_pos_svc_knm, 'delta_svc_mm': d_long, 
                        'top_db': top_layers[0]['db'] if top_layers else 12, 
                        'bot_db': bot_layers[0]['db'] if bot_layers else 12,
                        'stir_db': stir_db, 'stir_s': stir_s,
                        'pos': {'n': sum(l['n'] for l in bot_layers), 'area': as_prov_b, 'layers': bot_layers, 'status': (phi_Mn_b >= mu_pos_knm)},
                        'neg': {'n': sum(l['n'] for l in top_layers), 'area': as_prov_t, 'layers': top_layers, 'status': (phi_Mn_t >= mu_neg_knm)},
                        'shear': {'s': stir_s, 'db': stir_db, 'status': status_v},
                        'service': {'delta_long': d_long, 'limit_240': limit_240, 'ok': d_long <= limit_240},
                        'crack': {'w': w_crack, 'limit': 0.3, 'status': "Pass" if w_crack<=0.3 else "Fail"},
                        'top': {'n': top_layers[0]['n'] if top_layers else 0, 'db': top_layers[0]['db'] if top_layers else 12, 'layers': num_t_layers, 'all_layers': top_layers},
                        'bot': {'n': bot_layers[0]['n'] if bot_layers else 0, 'db': bot_layers[0]['db'] if bot_layers else 12, 'layers': num_b_layers, 'all_layers': bot_layers}
                    })

        with tab3:
            st.header("💵 Bill of Quantities")
            c_price1, c_price2, c_price3 = st.columns(3)
            price_conc = c_price1.number_input("Concrete (Baht/m³)", 2200, step=50)
            price_steel = c_price2.number_input("Rebar (Baht/kg)", 28.0, step=0.5)
            price_form = c_price3.number_input("Formwork (Baht/m²)", 300, step=10)

            if final_design_res:
                total_conc_vol, total_form_area, total_steel_weight = 0.0, 0.0, 0.0
                for res in final_design_res:
                    L, b_m, h_m = res['L'], res['b']/1000.0, res['h']/1000.0
                    total_conc_vol += b_m * h_m * L
                    total_form_area += (2*h_m + b_m) * L
                    w_top = sum(get_rebar_weight(l['db']) * l['n'] for l in res['top']['all_layers'])
                    w_bot = sum(get_rebar_weight(l['db']) * l['n'] for l in res['bot']['all_layers'])
                    total_steel_weight += (w_top + w_bot) * L * 1.05 
                    stir_len_m = (2 * (res['b'] + res['h']) / 1000.0) 
                    num_stir = (L * 1000.0) / res['shear']['s'] + 1
                    total_steel_weight += get_rebar_weight(res['shear']['db']) * stir_len_m * num_stir

                df_boq = pd.DataFrame([
                    {"Item": "Concrete", "Quantity": total_conc_vol, "Unit": "m³", "Unit Price": price_conc},
                    {"Item": "Rebar", "Quantity": total_steel_weight, "Unit": "kg", "Unit Price": price_steel},
                    {"Item": "Formwork", "Quantity": total_form_area, "Unit": "m²", "Unit Price": price_form},
                ])
                df_boq["Amount (THB)"] = df_boq["Quantity"] * df_boq["Unit Price"]
                
                c_boq1, c_boq2, c_boq3, c_boq4 = st.columns(4)
                c_boq1.metric("Concrete", f"{total_conc_vol:.2f} m³")
                c_boq2.metric("Steel", f"{total_steel_weight:.2f} kg")
                c_boq3.metric("Formwork", f"{total_form_area:.2f} m²")
                c_boq4.metric("TOTAL COST", f"{df_boq['Amount (THB)'].sum():,.0f} ฿", border=True)
                
                # FIXED: Apply format only to numeric columns
                st.dataframe(
                    df_boq.style.format({
                        "Quantity": "{:.2f}", 
                        "Unit Price": "{:.2f}", 
                        "Amount (THB)": "{:,.2f}"
                    }), 
                    use_container_width=True, hide_index=True
                )

    except Exception as e:
        st.error(f"Error: {e}")
