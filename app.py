import streamlit as st
import pandas as pd
import numpy as np
import io
import time

# --- 1. IMPORT CUSTOM MODULES (ต้องมีไฟล์เหล่านี้ในโฟลเดอร์เดียวกัน) ---
import input_handler
import solver
import design_view
import section_plotter

# --- 2. PAGE CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="Pro RC Beam Design", 
    layout="wide", 
    page_icon="🏗️",
    initial_sidebar_state="expanded"
)

# Custom CSS เพื่อความสวยงามและอ่านง่าย
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1E3A8A; margin-bottom: 0px;}
    .sub-header {font-size: 1.2rem; font-weight: normal; color: #64748B; margin-top: -10px;}
    .stApp {background-color: #F8FAFC;}
    div[data-testid="stMetricValue"] {font-size: 1.5rem; color: #0F172A;}
    .report-box {background-color: #ffffff; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0; font-family: 'Courier New', monospace;}
    .pass-tag {color: #166534; font-weight: bold; background-color: #DCFCE7; padding: 2px 8px; border-radius: 4px;}
    .fail-tag {color: #991B1B; font-weight: bold; background-color: #FEE2E2; padding: 2px 8px; border-radius: 4px;}
</style>
""", unsafe_allow_html=True)

# --- 3. HELPER FUNCTIONS: RC DESIGN LOGIC ---

def get_as_req(Mu_kNm, d_eff_mm, fc, fy, b_mm):
    """
    คำนวณปริมาณเหล็กเสริมที่ต้องการ (As required) ตามวิธี WSD/SDM (ในที่นี้ใช้ Concept USD)
    """
    if Mu_kNm == 0: return 0.0, 0.0, False
    Mu = abs(Mu_kNm) * 1e6 # แปลงหน่วยเป็น N-mm
    phi = 0.9 # Reduction factor for tension controlled
    
    # คำนวณ Rho
    m = fy / (0.85 * fc)
    Rn = Mu / (phi * b_mm * d_eff_mm**2)
    
    rho = 0.0
    is_error = False
    
    try:
        # Check if Rn is too high (Section failure)
        term = 1 - (2 * m * Rn) / fy
        if term < 0:
            rho = 0.0 
            is_error = True # หน้าตัดเล็กเกินไป รับโมเมนต์ไม่ไหว
        else:
            rho = (1/m) * (1 - np.sqrt(term))
    except:
        rho = 0.0
        is_error = True
    
    as_req = rho * b_mm * d_eff_mm
    
    # Min Reinforcement Check (ACI 318)
    if as_req > 0 and not is_error: 
        as_min1 = (0.25 * np.sqrt(fc) / fy) * b_mm * d_eff_mm
        as_min2 = (1.4 / fy) * b_mm * d_eff_mm
        as_min = max(as_min1, as_min2)
        return max(as_req, as_min), rho, False
    
    return as_req, rho, is_error

def get_phi_Mn_details(n, db, d_eff, b, fc, fy):
    """
    คำนวณ Moment Capacity (Phi Mn) และรายละเอียด Strain
    """
    Ast = n * (np.pi * (db/2)**2)
    if Ast == 0: return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Whitney Stress Block
    a = (Ast * fy) / (0.85 * fc * b)
    beta1 = 0.85 if fc <= 30 else max(0.65, 0.85 - 0.05 * (fc - 30) / 7)
    c = a / beta1
    
    # Strain Check
    dt = d_eff 
    if c > 0:
        strain_t = 0.003 * (dt - c) / c
    else:
        strain_t = 0.005 # Infinite ductility assumption
        
    phi = 0.9 # Simplified phi (ควรเช็ค Strain เพื่อปรับค่า phi จริงๆ แต่ใช้ 0.9 สำหรับคานดัดทั่วไป)

    Mn = Ast * fy * (d_eff - a/2)
    phi_Mn = phi * Mn / 1e6 # kNm
    return phi_Mn, Ast, a, Mn, c, strain_t

def check_shear_details(Vu_kN, b, d, fc, fy, stir_db, spacing):
    """
    คำนวณ Shear Capacity (Phi Vn)
    """
    Vu = abs(Vu_kN) * 1000 # N
    
    # Shear Strength provided by Concrete (Vc)
    Vc = 0.17 * np.sqrt(fc) * b * d
    phi = 0.85
    phi_Vc = phi * Vc
    
    # Shear Strength provided by Steel (Vs)
    Av = 2 * (np.pi * (stir_db/2)**2) # 2 legs (เหล็กปลอก 2 ขา)
    if spacing <= 0: spacing = 1000 
    
    Vs = (Av * fy * d) / spacing
    phi_Vs = phi * Vs
    phi_Vn = phi_Vc + phi_Vs
    
    status = "OK" if phi_Vn >= Vu else "FAIL"
    return status, phi_Vn/1000, phi_Vc/1000, phi_Vs/1000, Vc, Vs

# --- 4. MAIN APPLICATION ---

# Header Section
st.markdown('<div class="main-header">🏗️ RC Beam Analysis & Design Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Timoshenko / Finite Element Method</div>', unsafe_allow_html=True)
st.markdown("---")

# --- 4.1 SIDEBAR: INPUTS ---
with st.sidebar:
    st.header("📝 Project Information")
    project_name = st.text_input("Project Name", "Residential Building A")
    engineer_name = st.text_input("Engineer", "Eng. Somchai")
    st.markdown("---")
    
    # เรียกใช้ Input Handler เดิม
    params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()

# Check Stability
if not stable:
    st.error("🚨 **Structure Error:** โครงสร้างไม่เสถียร (Unstable)! กรุณาตรวจสอบจุดรองรับ (Support) ต้องมี Reaction อย่างน้อย 3 Component")
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
            st.info("ℹ️ Service Mode: Load Factors = 1.0")
            is_service = True
        else:
            with c1: f_dl = st.number_input("Dead Load (DL)", 1.4, 1.6, 1.4, 0.1)
            with c2: f_ll = st.number_input("Live Load (LL)", 1.7, 2.0, 1.7, 0.1)
            tag = "Ultimate"
            is_service = False
            # st.warning(f"⚡ Ultimate Mode: {f_dl}DL + {f_ll}LL")

    # --- 4.3 LOAD CALCULATION PROCESS ---
    # (Logic เดิม: คำนวณ Load รวม Self-weight)
    try:
        w_sw_base_kN = params['b'] * params['h'] * 24.0      
        w_sw_factored_kN = w_sw_base_kN * f_dl
        
        span_total_udl_N = {i: w_sw_factored_kN * 1000.0 for i in range(n_spans)} 
        combined_loads_list = []
        
        if not loads_df.empty:
            for _, row in loads_df.iterrows():
                try:
                    s_idx = int(row['span_index'])
                    if s_idx >= n_spans: continue 
                    
                    l_type = row['type']
                    u_factor = f_dl if row['case'] == 'DL' else f_ll
                    mag_base_kN = float(row['mag']) 
                    mag_factored_N = mag_base_kN * u_factor * 1000.0 
                    dist = float(row['dist'])
                    d_start = float(row['d_start'])
                    
                    if l_type == 'P':
                        combined_loads_list.append({
                            'span_index': s_idx, 'type': 'P', 'mag': mag_factored_N, 
                            'd_start': d_start, 'dist': 0.0, 'desc': f'User Point ({row["case"]})'
                        })
                    elif l_type == 'U':
                        if d_start <= 0.01 and dist >= (spans[s_idx] - 0.01):
                            span_total_udl_N[s_idx] += mag_factored_N
                        else:
                            combined_loads_list.append({
                                'span_index': s_idx, 'type': 'U', 'mag': mag_factored_N, 
                                'd_start': d_start, 'dist': dist, 'desc': f'User Partial UDL ({row["case"]})'
                            })
                except Exception: continue
        
        for i in range(n_spans):
            if span_total_udl_N[i] > 0:
                combined_loads_list.append({
                    'span_index': i, 'type': 'U', 'mag': span_total_udl_N[i], 
                    'd_start': 0.0, 'dist': spans[i], 'desc': 'Total Combined UDL (Incl. SW)'
                })
        
        calc_loads_df = pd.DataFrame(combined_loads_list)

   # --- 4.4 RUN SOLVER ---
        with st.spinner('Running Analysis...'):
            x_eval, M, V, D, R = solver.solve_beam(spans, sup_df, calc_loads_df, params)
        
        # --- FIX: สร้าง res_df ด้วยชื่อคอลัมน์เดิม (x, moment, shear, deflection) ---
        # เพื่อให้ design_view.py ทำงานได้ถูกต้อง
        res_df = pd.DataFrame({
            'x': x_eval,
            'moment': M, 
            'shear': V,    
            'deflection': D * 1000 # แปลงเป็น mm ตาม Logic เดิม
        })

        # --- สร้าง DataFrame สำหรับแสดงผล (เปลี่ยนชื่อให้สวยงามที่นี่แทน) ---
        res_df_display = res_df.copy()
        res_df_display.rename(columns={
            'x': 'x (m)',
            'moment': 'Moment (N-mm)',
            'shear': 'Shear (N)',
            'deflection': 'Deflection (mm)'
        }, inplace=True)
        
        # เพิ่มหน่วย kNm และ kN เพื่อใช้ในการดึงค่ามาคำนวณ Design
        res_df_display['Moment (kNm)'] = res_df_display['Moment (N-mm)'] / 1e6
        res_df_display['Shear (kN)'] = res_df_display['Shear (N)'] / 1000
        
        # --- 5. TABS INTERFACE ---
        tab1, tab2 = st.tabs(["📊 1. Analysis Results", "📝 2. Concrete Design & Detailing"])
        
        # ================= TAB 1: ANALYSIS =================
        with tab1:
            st.subheader("📈 Force Diagrams")
            # ส่ง res_df (ตัวที่มีคอลัมน์ 'x') ไปให้ฟังก์ชัน plot
            st.plotly_chart(design_view.plot_analysis_results(res_df, spans, sup_df, calc_loads_df, R), use_container_width=True)
            
            # Key Metrics
            st.markdown("### 📌 Critical Values")
            v_max_kN = res_df_display['Shear (kN)'].abs().max()
            m_max_pos_kNm = res_df_display['Moment (kNm)'].max()
            m_max_neg_kNm = res_df_display['Moment (kNm)'].min()
            d_abs_max_mm = res_df_display['Deflection (mm)'].abs().max()
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Max Shear (Vu)", f"{v_max_kN:.2f} kN", delta_color="off")
            m2.metric("Max Moment (+)", f"{m_max_pos_kNm:.2f} kNm", delta_color="off")
            m3.metric("Max Moment (-)", f"{m_max_neg_kNm:.2f} kNm", delta_color="inverse")
            m4.metric("Max Deflection", f"{d_abs_max_mm:.2f} mm", delta_color="off")

            # Support Reactions
            st.markdown("### 📍 Support Reactions")
            if R:
                reaction_data = [{"Node": int(str(k).replace('R', '')), "Reaction (kN)": v/1000.0} for k, v in R.items()]
                df_reac = pd.DataFrame(reaction_data).sort_values(by="Node")
                
                col_r1, col_r2 = st.columns([1, 2])
                with col_r1:
                    st.dataframe(df_reac.style.format({"Reaction (kN)": "{:.2f}"}), use_container_width=True, hide_index=True)
                with col_r2:
                    # Equilibrium Check
                    sum_R_kN = sum(R.values()) / 1000.0
                    total_applied_N = 0
                    for _, l in calc_loads_df.iterrows():
                        if l['type'] == 'P': total_applied_N += l['mag']
                        else: total_applied_N += (l['mag'] * l['dist'])
                    total_applied_kN = total_applied_N / 1000.0
                    
                    st.write(f"**Equilibrium Check (ΣFy = 0):**")
                    st.info(f"Total Reactions ({sum_R_kN:.2f} kN) ≈ Total Loads ({total_applied_kN:.2f} kN)")
                    if abs(sum_R_kN - total_applied_kN) > 0.1:
                        st.error(f"⚠️ Warning: Unbalanced Forces! Diff: {abs(sum_R_kN - total_applied_kN):.3f} kN")
            
            # Export Data Button
            csv = res_df_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Analysis Results (CSV)",
                data=csv,
                file_name=f'Analysis_Results_{project_name}.csv',
                mime='text/csv',
            )

        # ================= TAB 2: INTERACTIVE DESIGN =================
        with tab2:
            st.header(f"🏗️ Interactive RC Design ({tag})")
            
            if is_service:
                st.warning("⚠️ Warning: คุณเลือกโหมด Service Load (Factor=1.0) กรุณาเปลี่ยนเป็น Ultimate เพื่อการออกแบบเหล็กเสริมที่ถูกต้อง")
            
            # Design Constants
            b_mm, h_mm = params['b'] * 1000, params['h'] * 1000
            fc, fy = params['fc'], params['fy']
            
            final_design_res = []
            offsets = [0] + list(np.cumsum(spans))
            
            # Prepare Report Header
            full_cal_report = f"PROJECT: {project_name}\nENGINEER: {engineer_name}\n"
            full_cal_report += f"DATE: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            full_cal_report += "="*60 + "\n"
            full_cal_report += f"DESIGN PARAMETERS:\n  fc' = {fc} MPa\n  fy = {fy} MPa\n  Section: {b_mm:.0f}x{h_mm:.0f} mm\n"
            full_cal_report += f"  Load Factors: DL={f_dl}, LL={f_ll}\n"
            full_cal_report += "="*60 + "\n\n"

# --- SPAN LOOP (Final Fix: Units & Display) ---
            for i in range(n_spans):
                s_len = spans[i]
                s_start, s_end = offsets[i], offsets[i+1]
                
                # Extract Forces
                span_data = res_df_display[(res_df_display['x (m)'] >= s_start - 1e-6) & (res_df_display['x (m)'] <= s_end + 1e-6)]
                
                if not span_data.empty:
                    mu_pos = span_data['Moment (kNm)'].max()
                    mu_neg = abs(span_data['Moment (kNm)'].min())
                    vu_max = span_data['Shear (kN)'].abs().max()
                else:
                    mu_pos, mu_neg, vu_max = 0, 0, 0

                # Report Logic
                full_cal_report += f"\n>> SPAN {i+1} (Length {s_len} m)\n"
                full_cal_report += f"   Design Forces: Mu(+)={mu_pos:.2f} kNm, Mu(-)={mu_neg:.2f} kNm, Vu={vu_max:.2f} kN\n"

                # UI Layout
                with st.expander(f"📍 **Span {i+1}** (L={s_len} m) | Forces: $M_u^+$ {mu_pos:.2f}, $M_u^-$ {mu_neg:.2f}, $V_u$ {vu_max:.2f}", expanded=True):
                    
                    # Covering Input
                    c_const, c_cov = st.columns([3, 1])
                    with c_const:
                        st.caption(f"Design Constants: fc'={fc}, fy={fy}, Size {b_mm:.0f}x{h_mm:.0f} mm")
                    with c_cov:
                        cover_mm = st.number_input(f"Covering (mm)", value=25.0, step=5.0, key=f"cov_{i}")

                    # ==================================================
                    # 1. POSITIVE MOMENT DESIGN (Bottom Steel)
                    # ==================================================
                    st.markdown("##### 1. Bottom Reinforcement (Mid-Span, $+M_u$)")
                    d_eff_bot_est = h_mm - cover_mm - 9 - 10 
                    as_req_bot, rho_bot, err_bot = get_as_req(mu_pos, d_eff_bot_est, fc, fy, b_mm)
                    
                    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
                    with c1: st.markdown(f"**Req $A_s$:**\n`{as_req_bot:.0f}` mm²")
                    with c2: bot_db = st.selectbox("DB", [12, 16, 20, 25, 28], index=1, key=f"bdb_{i}")
                    with c3: bot_n = st.number_input("Qty", 2, 10, 2, key=f"bn_{i}")
                    
                    d_eff_bot_real = h_mm - cover_mm - 9 - (bot_db / 2)
                    phi_Mn_bot, as_prov_bot, _, _, _, _ = get_phi_Mn_details(bot_n, bot_db, d_eff_bot_real, b_mm, fc, fy)
                    pass_b = phi_Mn_bot >= mu_pos
                    
                    with c4: 
                        clr_b = "green" if pass_b else "red"
                        icon_b = "✅ OK" if pass_b else "❌ Fail"
                        
                        # Display Fixed
                        st.markdown(f"**$A_{{s,prov}}$**: :{clr_b}[**{as_prov_bot:.0f}**] vs **{as_req_bot:.0f}** mm²")
                        st.markdown(f"**Strength**: $\phi M_n =$ :{clr_b}[**{phi_Mn_bot:.2f}**] **kNm** $\ge M_u =$ **{mu_pos:.2f}** **kNm**")
                        st.caption(f"Status: {icon_b}")
                    
                    full_cal_report += f"   [Bottom] Prov: {bot_n}-DB{bot_db} (As={as_prov_bot:.0f}), phiMn={phi_Mn_bot:.2f} >= Mu={mu_pos:.2f} -> {icon_b}\n"

                    # ==================================================
                    # 2. NEGATIVE MOMENT DESIGN (Top Steel)
                    # ==================================================
                    st.markdown("##### 2. Top Reinforcement (Supports, $-M_u$)")
                    d_eff_top_est = h_mm - cover_mm - 9 - 10 
                    as_req_top, rho_top, err_top = get_as_req(mu_neg, d_eff_top_est, fc, fy, b_mm)
                    
                    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
                    with c1: st.markdown(f"**Req $A_s$:**\n`{as_req_top:.0f}` mm²")
                    with c2: top_db = st.selectbox("DB", [12, 16, 20, 25, 28], index=1, key=f"tdb_{i}")
                    with c3: top_n = st.number_input("Qty", 2, 10, 2, key=f"tn_{i}")
                    
                    d_eff_top_real = h_mm - cover_mm - 9 - (top_db / 2)
                    phi_Mn_top, as_prov_top, _, _, _, _ = get_phi_Mn_details(top_n, top_db, d_eff_top_real, b_mm, fc, fy)
                    pass_t = phi_Mn_top >= mu_neg
                    
                    with c4:
                        clr_t = "green" if pass_t else "red"
                        icon_t = "✅ OK" if pass_t else "❌ Fail"
                        
                        st.markdown(f"**$A_{{s,prov}}$**: :{clr_t}[**{as_prov_top:.0f}**] vs **{as_req_top:.0f}** mm²")
                        st.markdown(f"**Strength**: $\phi M_n =$ :{clr_t}[**{phi_Mn_top:.2f}**] **kNm** $\ge M_u =$ **{mu_neg:.2f}** **kNm**")
                        st.caption(f"Status: {icon_t}")

                    full_cal_report += f"   [Top]    Prov: {top_n}-DB{top_db} (As={as_prov_top:.0f}), phiMn={phi_Mn_top:.2f} >= Mu={mu_neg:.2f} -> {icon_t}\n"

                    # ==================================================
                    # 3. SHEAR DESIGN (Stirrups)
                    # ==================================================
                    st.markdown("##### 3. Shear Reinforcement (Stirrups, $V_u$)")
                    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
                    with c1: st.markdown(f"**Design $V_u$:**\n`{vu_max:.2f}` kN")
                    with c2: stir_db = st.selectbox("Stirrup", [6, 9, 12], index=0, key=f"sdb_{i}")
                    with c3: stir_s = st.number_input("Spacing (mm)", 50, 300, 150, 10, key=f"ss_{i}")
                    
                    d_shear = d_eff_bot_real 
                    status_v, phi_Vn, phi_Vc, phi_Vs, _, _ = check_shear_details(vu_max, b_mm, d_shear, fc, fy, stir_db, stir_s)
                    
                    with c4:
                        clr_v = "green" if status_v == "OK" else "red"
                        icon_v = "✅ OK" if status_v == "OK" else "❌ Fail"
                        
                        st.markdown(f"**Strength**: $\phi V_n =$ :{clr_v}[**{phi_Vn:.1f}**] **kN** $\ge V_u =$ **{vu_max:.1f}** **kN**")
                        st.caption(f"($\phi V_c={phi_Vc:.1f} + \phi V_s={phi_Vs:.1f}$ kN)")
                    
                    full_cal_report += f"   [Shear]  Prov: RB{stir_db}@{stir_s}, phiVn={phi_Vn:.2f} >= Vu={vu_max:.2f} -> {icon_v}\n"
                    full_cal_report += "-"*30

                    # Collect Data
                    final_design_res.append({
                        'span': i+1, 'cover': cover_mm,
                        'top_db': top_db, 'bot_db': bot_db, 'stir_db': stir_db,
                        'pos': {'n': bot_n, 'area': as_prov_bot, 'status': pass_b},
                        'neg': {'n': top_n, 'area': as_prov_top, 'status': pass_t},
                        'shear': {'s': stir_s, 'status': status_v}
                    })
                    
            # --- SUMMARY & REPORT ---
            st.markdown("---")
            st.subheader("📋 Design Summary & Drawing")
            
            # Create Summary Table
            summary_data = []
            for item in final_design_res:
                summary_data.append({
                    "Span": item['span'],
                    "Bottom Rebar": f"{item['pos']['n']}-DB{item['bot_db']}",
                    "Top Rebar": f"{item['neg']['n']}-DB{item['top_db']}",
                    "Stirrup": f"RB{item['stir_db']}@{item['shear']['s']}",
                    "Result": "✅ Pass" if (item['pos']['status'] and item['neg']['status'] and item['shear']['status'] == "OK") else "❌ Fail"
                })
            st.table(pd.DataFrame(summary_data))

            # Actions
            col_act1, col_act2 = st.columns(2)
            with col_act1:
                # Plot Drawings
                if st.button("🔄 Generate/Update Drawings", type="primary"):
                    try:
                        st.write("**Longitudinal Section:**")
                        fig_long = section_plotter.plot_longitudinal_section_detailed(spans, sup_df, final_design_res, params['h'], final_design_res[0]['cover'])
                        st.pyplot(fig_long, use_container_width=True)
                        
                        st.write("**Cross Section (Typical Span 1):**")
                        # (Sample Cross Section plot for Span 1)
                        res1 = final_design_res[0]
                        fig_sec = section_plotter.plot_section(
                            params['b'], params['h'], res1['cover'], res1['top_db'], res1['bot_db'],
                            res1['neg']['n'], res1['pos']['n'], f"RB{res1['stir_db']}@{res1['shear']['s']}",
                            fc, fy, "SECTION A-A (Span 1)"
                        )
                        st.pyplot(fig_sec, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error plotting: {e}")
            
            with col_act2:
                # Download Report
                st.download_button(
                    label="📄 Download Calculation Report (.txt)",
                    data=full_cal_report,
                    file_name=f"Design_Report_{project_name}.txt",
                    mime="text/plain"
                )
                
                # Show Report Preview
                with st.expander("View Report Preview"):
                    st.text(full_cal_report)

    except Exception as e:
        st.error(f"❌ Calculation Error: {e}")
        st.warning("Please check your input loads or support conditions.")
        st.exception(e)  





