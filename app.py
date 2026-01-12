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

# --- 3. INTERNAL HELPER: PLOT CROSS SECTION (FIXED) ---
def plot_cross_section_fixed(b, h, cover, top_layers, bot_layers, shear_res):
    """
    Generate Cross Section Image using Matplotlib.
    Fixed: Aspect ratio and x-limits to prevent text clipping.
    """
    # 1. Setup Figure (Portrait)
    fig, ax = plt.subplots(figsize=(5, 6))
    
    # 2. Draw Concrete
    rect = patches.Rectangle((0, 0), b, h, linewidth=2, edgecolor='black', facecolor='white')
    ax.add_patch(rect)
    
    # 3. Draw Stirrup
    stirrup_rect = patches.Rectangle((cover, cover), b - 2*cover, h - 2*cover, 
                                     linewidth=1.5, edgecolor='#34495e', facecolor='none', linestyle='-')
    ax.add_patch(stirrup_rect)
    
    # 4. Draw Rebars
    # --- Top Rebar ---
    # Simplified: Visualize based on total count, using dia of first layer
    n_top = sum(l['n'] for l in top_layers)
    if top_layers:
        dia_top = top_layers[0]['db']
    else:
        dia_top = 12 # Default fallback

    start_x = cover + dia_top/2
    end_x = b - cover - dia_top/2
    
    if n_top > 1:
        gap = (end_x - start_x) / (n_top - 1)
        for i in range(n_top):
            cx = start_x + i*gap
            cy = h - cover - dia_top/2
            ax.add_patch(patches.Circle((cx, cy), radius=dia_top/2, color='#c0392b'))
    elif n_top == 1:
        ax.add_patch(patches.Circle((b/2, h - cover - dia_top/2), radius=dia_top/2, color='#c0392b'))

    # --- Bot Rebar ---
    n_bot = sum(l['n'] for l in bot_layers)
    if bot_layers:
        dia_bot = bot_layers[0]['db']
    else:
        dia_bot = 12 # Default fallback
    
    start_x = cover + dia_bot/2
    end_x = b - cover - dia_bot/2
    
    if n_bot > 1:
        gap = (end_x - start_x) / (n_bot - 1)
        for i in range(n_bot):
            cx = start_x + i*gap
            cy = cover + dia_bot/2
            ax.add_patch(patches.Circle((cx, cy), radius=dia_bot/2, color='#27ae60'))
    elif n_bot == 1:
        ax.add_patch(patches.Circle((b/2, cover + dia_bot/2), radius=dia_bot/2, color='#27ae60'))

    # 5. Add Labels (Fixed Position)
    text_x = b + (b * 0.1) # Offset text to the right
    
    # Top Label
    ax.text(text_x, h - cover, f"Top:\n{n_top}DB{int(dia_top)}", 
            color='#c0392b', fontsize=12, fontweight='bold', va='center')

    # Bot Label
    ax.text(text_x, cover + dia_bot, f"Bot:\n{n_bot}DB{int(dia_bot)}", 
            color='#27ae60', fontsize=12, fontweight='bold', va='center')
    
    # Shear Label (Boxed)
    shear_text = f"Shear:\nRB{int(shear_res['db'])} @ {int(shear_res['s'])} mm"
    ax.text(text_x, h/2, shear_text, 
            color='#2c3e50', fontsize=10, fontweight='bold', va='center',
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#34495e", alpha=0.9))

    # Title
    ax.set_title(f"SECTION {int(b)}x{int(h)} mm", fontsize=14, fontweight='bold', pad=20)

    # 6. Final Adjustments
    ax.set_aspect('equal')
    ax.axis('off')
    # *** KEY FIX: Expand Limits ***
    ax.set_xlim(-50, b + 250) 
    ax.set_ylim(-50, h + 50)
    
    plt.tight_layout()
    return fig

# --- 4. MAIN HEADER ---
st.markdown('<div class="main-header">🏗️ RC Beam Analysis & Design Pro</div>', unsafe_allow_html=True)

# --- 5. SIDEBAR ---
with st.sidebar:
    params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()
    
    # --- ADDED: UNIT COST INPUTS ---
    st.markdown("---")
    st.markdown("### 💰 Cost Estimation (BOQ)")
    price_conc = st.number_input("Concrete (Baht/m³)", value=2200, step=50)
    price_steel = st.number_input("Rebar (Baht/kg)", value=28.0, step=0.5)
    price_form = st.number_input("Formwork (Baht/m²)", value=300, step=10)

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
        # 1. Ultimate Run (For Reinforcement Design)
        calc_loads_ult = rc_load_processor.prepare_load_dataframe(loads_df, n_spans, spans, params, f_dl, f_ll)
        x_ult, M_ult, V_ult, D_ult, R_ult = solver.solve_beam(spans, sup_df, calc_loads_ult, params)
        
        # 2. Service Run (For Deflection & Crack Check) - Fix factors to 1.0
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
                mu_pos = max(0.0, (M_ult[mask_u]/1000.0).max())
                mu_neg = abs(min(0.0, (M_ult[mask_u]/1000.0).min()))
                vu_max = abs((V_ult[mask_u] / 1000.0)).max()

                # Service Forces
                mask_s = (x_svc >= s_start - 1e-6) & (x_svc <= s_end + 1e-6)
                ma_pos_svc = max(0.0, (M_svc[mask_s]/1000.0).max())
                delta_elastic_mm = abs(D_svc[mask_s]).max() * 1000.0

                with st.expander(f"📍 SPAN {i+1} (L={s_len} m)", expanded=True):
                    col_input, col_draw = st.columns([2, 1])
                    with col_input:
                        cover_mm = st.number_input(f"Cover (mm)", 20, 50, 25, key=f"cov_{i}")

                        # --- 1. TOP STEEL ---
                        st.markdown("#### 🔼 Top Reinforcement (Negative Moment)")
                        num_t_layers = st.selectbox("Number of Top Layers", [1, 2, 3], index=0, key=f"tl_qty_{i}")
                        top_layers = []
                        for l_idx in range(num_t_layers):
                            ct1, ct2 = st.columns([1, 1])
                            with ct1: t_db = st.selectbox(f"L{l_idx+1} Size", [12, 16, 20, 25, 28], index=1, key=f"tdb_{i}_{l_idx}")
                            with ct2: t_qty = st.number_input(f"L{l_idx+1} Qty", 0, 20, 2 if l_idx==0 else 0, key=f"tn_{i}_{l_idx}")
                            top_layers.append({'n': t_qty, 'db': t_db})
                        
                        d_t_val, as_prov_t, y_centroid_t = rc_design_engine.get_centroid_and_d(top_layers, h_mm, cover_mm, 9)
                        d_t = h_mm - y_centroid_t if y_centroid_t > 0 else h_mm - (cover_mm + 9 + 16/2)
                        
                        as_req_t, _, _ = rc_design_engine.get_as_req(mu_neg, d_t, fc, fy, b_mm)
                        as_min_t = max((0.25 * np.sqrt(fc) / fy) * b_mm * d_t, (1.4 / fy) * b_mm * d_t)
                        phi_Mn_t, _, _, _, _, _ = rc_design_engine.get_phi_Mn_details_multi(top_layers, d_t, b_mm, h_mm, fc, fy)

                        st.markdown(f"""
                        | **Top Steel Analysis** | **Required** | **Minimum** | **Provided** | **Status** |
                        | :--- | :---: | :---: | :---: | :---: |
                        | **Area ($A_s$, mm²)** | {as_req_t:.0f} | {as_min_t:.0f} | **{as_prov_t:.0f}** | {"✅" if as_prov_t >= max(as_req_t, as_min_t) else "❌"} |
                        | **Capacity (kNm)** | $M_u$: {mu_neg:.1f} | --- | **$\phi M_n$: {phi_Mn_t:.1f}** | {"✅" if phi_Mn_t >= mu_neg else "❌"} |
                        """)

                        # --- 2. BOTTOM STEEL ---
                        st.markdown("#### 🔽 Bottom Reinforcement (Positive Moment)")
                        num_b_layers = st.selectbox("Number of Bottom Layers", [1, 2, 3], index=0, key=f"bl_qty_{i}")
                        bot_layers = []
                        for l_idx in range(num_b_layers):
                            cb1, cb2 = st.columns([1, 1])
                            with cb1: b_db = st.selectbox(f"L{l_idx+1} Size", [12, 16, 20, 25, 28], index=1, key=f"bdb_{i}_{l_idx}")
                            with cb2: b_qty = st.number_input(f"L{l_idx+1} Qty", 0, 20, 3 if l_idx==0 else 0, key=f"bn_{i}_{l_idx}")
                            bot_layers.append({'n': b_qty, 'db': b_db})
                        
                        d_b, as_prov_b, _ = rc_design_engine.get_centroid_and_d(bot_layers, h_mm, cover_mm, 9)
                        if d_b <= 0: d_b = h_mm - (cover_mm + 9 + 16/2)
                        
                        as_req_b, _, _ = rc_design_engine.get_as_req(mu_pos, d_b, fc, fy, b_mm)
                        phi_Mn_b, _, _, _, _, _ = rc_design_engine.get_phi_Mn_details_multi(bot_layers, d_b, b_mm, h_mm, fc, fy)

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

                        # --- 4. DEFLECTION CHECK ---
                        st.markdown("---")
                        st.markdown("#### 📉 Deflection Control (Serviceability)")
                        d_inst, d_long, Ie, Icr, lambda_d = rc_design_engine.check_serviceability(
                            ma_pos_svc, delta_elastic_mm, b_mm, h_mm, d_b, as_prov_b, as_prov_t, fc
                        )
                        limit_240 = (s_len * 1000) / 240
                        limit_480 = (s_len * 1000) / 480
                        status_def = "✅ OK" if d_long <= limit_240 else "❌ Fail"
                        
                        st.markdown(f"""
                        | Check Item | Value | Limit (L/240) | Limit (L/480) | Status |
                        | :--- | :---: | :---: | :---: | :---: |
                        | Immediate $\Delta$ | {d_inst:.2f} mm | - | - | - |
                        | **Long-term $\Delta$** | **{d_long:.2f} mm** | **{limit_240:.2f} mm** | {limit_480:.2f} mm | **{status_def}** |
                        """)
                        st.caption(f"*Calculated using ACI 318-19, $I_e$={Ie/1e4:.0f}cm⁴, $\lambda_\Delta$={lambda_d:.2f}")

                        # --- 5. CRACK WIDTH CONTROL (NEW FEATURE) ---
                        st.markdown("#### ⚡ Crack Width Control")
                        
                        # รวมจำนวนเหล็กล่างทั้งหมด
                        total_n_bars_bot = sum(l['n'] for l in bot_layers)
                        
                        w_crack, fs_actual = rc_design_engine.check_crack_width(
                            Ma_svc=ma_pos_svc, # โมเมนต์ใช้งานจริง (Service)
                            b=b_mm, 
                            h=h_mm, 
                            d=d_b, 
                            As=as_prov_b, 
                            n_bars=total_n_bars_bot, 
                            fc=fc
                        )
                        
                        limit_crack = 0.30 # mm (Standard for internal use)
                        status_crack = "✅ Pass" if w_crack <= limit_crack else "⚠️ Warning"
                        
                        c_cr1, c_cr2, c_cr3 = st.columns(3)
                        c_cr1.metric("Service Stress ($f_s$)", f"{fs_actual:.1f} MPa", help="เหล็กรับแรงดึงทำงานจริงที่ Service Load")
                        c_cr2.metric("Crack Width ($w$)", f"{w_crack:.3f} mm", help="คำนวณด้วยสูตร Gergely-Lutz")
                        c_cr3.metric("Limit (General)", f"{limit_crack} mm", status_crack)
                        
                        if w_crack > limit_crack:
                            st.warning(f"⚠️ รอยร้าวคำนวณได้ {w_crack:.3f} mm เกินมาตรฐาน {limit_crack} mm -> แนะนำให้เพิ่มจำนวนเหล็กแต่ลดขนาดหน้าตัดลง (ใช้เหล็กเล็กแต่เยอะขึ้น)")

                    with col_draw:
                        # --- UPDATED: Use Matplotlib Fix ---
                        fig_cs = plot_cross_section_fixed(
                            b=b_mm, h=h_mm, cover=cover_mm,
                            top_layers=top_layers,
                            bot_layers=bot_layers,
                            shear_res={'db': stir_db, 's': stir_s}
                        )
                        st.pyplot(fig_cs)
                        plt.close(fig_cs) # Clean memory

                    final_design_res.append({
                        'span_id': i, 'L': s_len, 'b': b_mm, 'h': h_mm, 'fc': fc, 'fy': fy, 
                        'Mu_pos': mu_pos, 'Mu_neg': mu_neg, 'Vu_max': vu_max, 'cover': cover_mm,
                        'Ma_pos_svc': ma_pos_svc, 'delta_svc_mm': d_long, 
                        'top_db': top_layers[0]['db'] if top_layers else 12, 
                        'bot_db': bot_layers[0]['db'] if bot_layers else 12,
                        'stir_db': stir_db, 
                        'pos': {'n': sum(l['n'] for l in bot_layers), 'area': as_prov_b, 'layers': bot_layers, 'status': (phi_Mn_b >= mu_pos)},
                        'neg': {'n': sum(l['n'] for l in top_layers), 'area': as_prov_t, 'layers': top_layers, 'status': (phi_Mn_t >= mu_neg)},
                        'shear': {'s': stir_s, 'db': stir_db, 'status': status_v},
                        'service': {'delta_long': d_long, 'limit_240': limit_240, 'ok': d_long <= limit_240},
                        'crack': {'w': w_crack, 'limit': limit_crack, 'status': status_crack},
                        'top': {'n': top_layers[0]['n'] if top_layers else 0, 'db': top_layers[0]['db'] if top_layers else 12, 'layers': num_t_layers, 'all_layers': top_layers},
                        'bot': {'n': bot_layers[0]['n'] if bot_layers else 0, 'db': bot_layers[0]['db'] if bot_layers else 12, 'layers': num_b_layers, 'all_layers': bot_layers}
                    })

            st.markdown("---")
            if st.button("🏗️ Generate Detailed Drawing (Longitudinal Section)"):
                svg_long, _ = section_plotter.plot_longitudinal_section_detailed(spans, sup_df, final_design_res, h_mm, cover_mm)
                st.components.v1.html(f'<div style="background:white; overflow-x:auto; border:1px solid #ddd; border-radius:8px; padding:10px;">{svg_long}</div>', height=500, scrolling=True)

        # ================= TAB 1: ANALYSIS RESULTS =================
        with tab1:
            st.subheader(f"📈 Diagrams ({tag} Load)")
            df_for_plot = pd.DataFrame({'x': x_plot, 'moment': M_plot, 'shear': V_plot, 'deflection': D_plot * 1000})
            
            # เรียกใช้ฟังก์ชัน plot จาก design_view โดยใช้ชื่อ argument ที่ถูกต้อง
            fig = design_view.plot_analysis_results(
                res_df=df_for_plot, 
                spans=spans, 
                supports=sup_df, 
                loads=calc_loads_ult if not is_service else calc_loads_svc, 
                reactions=R_plot
            )
            st.plotly_chart(fig, use_container_width=True)
            
            c_m1, c_m2, c_m3 = st.columns(3)
            c_m1.metric(f"Max Shear ({tag})", f"{max(abs(V_plot))/1000:.2f} kN")
            c_m2.metric(f"Max Moment ({tag})", f"{max(M_plot)/1000:.2f} kNm")
            c_m3.metric(f"Max Deflection (Elastic)", f"{max(abs(D_plot))*1000:.2f} mm")

        # ================= TAB 3: REPORT =================
        with tab3:
            st.header("📝 Calculation Reports")
            if not final_design_res:
                st.warning("⚠️ กรุณาทำการออกแบบใน Tab 2 ก่อน")
            else:
                for res in final_design_res:
                    with st.expander(f"📘 Span {res['span_id']+1} Details", expanded=(res['span_id']==0)):
                        reporter.render_calculation_report(res)
        
        # ================= BOQ SECTION (CORRECTED & LINKED) =================
        st.markdown("---")
        st.header("💵 Bill of Quantities (BOQ)")
        
        if final_design_res:
            try:
                # 1. ดึงปริมาณงานจากฟังก์ชัน design_view.calculate_boq_summary ที่แก้ไขใหม่
                boq_df = design_view.calculate_boq_summary(final_design_res, spans)
                
                # 2. ใส่ราคาต่อหน่วยจาก Sidebar (Mapping)
                price_map = {
                    "Concrete Structure (240 ksc)": price_conc,
                    "Formwork (Beam sides & bottom)": price_form,
                    "Deformed Bars (DB) + Stirrups (RB)": price_steel
                }
                
                # 3. คำนวณราคารวม
                boq_df["Unit Price (THB)"] = boq_df["Item"].map(price_map)
                boq_df["Amount (THB)"] = boq_df["Quantity"] * boq_df["Unit Price (THB)"]
                
                # 4. จัด Format ตาราง
                formatted_df = boq_df.copy()
                total_cost = formatted_df["Amount (THB)"].sum()
                
                # แสดงผล Metrics ด้านบนตาราง
                c_boq1, c_boq2, c_boq3, c_boq4 = st.columns(4)
                
                conc_row = formatted_df[formatted_df['Item'].str.contains("Concrete")].iloc[0]
                steel_row = formatted_df[formatted_df['Item'].str.contains("Bars")].iloc[0]
                form_row = formatted_df[formatted_df['Item'].str.contains("Formwork")].iloc[0]
                
                c_boq1.metric("Concrete", f"{conc_row['Quantity']:.2f} m³", f"{conc_row['Amount (THB)']:,.0f} ฿")
                c_boq2.metric("Rebar (+Stir)", f"{steel_row['Quantity']:.2f} kg", f"{steel_row['Amount (THB)']:,.0f} ฿")
                c_boq3.metric("Formwork", f"{form_row['Quantity']:.2f} m²", f"{form_row['Amount (THB)']:,.0f} ฿")
                c_boq4.metric("TOTAL COST", f"{total_cost:,.0f} ฿", border=True)
                
                # แสดงตาราง
                st.dataframe(
                    formatted_df.style.format({
                        "Quantity": "{:.2f}", 
                        "Unit Price (THB)": "{:,.2f}", 
                        "Amount (THB)": "{:,.2f}"
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                # ปุ่ม Download
                csv = formatted_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download BOQ (CSV)",
                    data=csv,
                    file_name='beam_boq_estimate.csv',
                    mime='text/csv',
                )

            except AttributeError:
                st.error("⚠️ ไม่พบฟังก์ชัน 'calculate_boq_summary' ใน design_view.py")
                st.info("กรุณาอัปเดตไฟล์ design_view.py ตามโค้ดที่ให้ไปในข้อความก่อนหน้านี้ครับ")
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการคำนวณ BOQ: {e}")

    except Exception as e:
        st.error(f"❌ **System Error:** {e}")
        st.info("รายละเอียด Error สำหรับการ Debug:")
        st.exception(e)
