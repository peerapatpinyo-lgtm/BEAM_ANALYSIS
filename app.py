import streamlit as st
import pandas as pd
import numpy as np
import io
import time

# --- 1. IMPORT CUSTOM MODULES ---
# ต้องมีไฟล์เหล่านี้ในโฟลเดอร์เดียวกัน
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

# Custom CSS
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
    """คำนวณปริมาณเหล็กเสริมที่ต้องการ (As required) USD Concept"""
    if Mu_kNm == 0: return 0.0, 0.0, False
    Mu = abs(Mu_kNm) * 1e6 # N-mm
    phi = 0.9 
    
    m = fy / (0.85 * fc)
    Rn = Mu / (phi * b_mm * d_eff_mm**2)
    
    rho = 0.0
    is_error = False
    
    try:
        term = 1 - (2 * m * Rn) / fy
        if term < 0:
            rho = 0.0 
            is_error = True # Section too small
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
    """คำนวณ Moment Capacity (Phi Mn) และ Strain"""
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
        strain_t = 0.005
        
    phi = 0.9 

    Mn = Ast * fy * (d_eff - a/2)
    phi_Mn = phi * Mn / 1e6 # kNm
    return phi_Mn, Ast, a, Mn, c, strain_t

def check_shear_details(Vu_kN, b, d, fc, fy, stir_db, spacing):
    """คำนวณ Shear Capacity (Phi Vn)"""
    Vu = abs(Vu_kN) * 1000 # N
    
    # Vc
    Vc = 0.17 * np.sqrt(fc) * b * d
    phi = 0.85
    phi_Vc = phi * Vc
    
    # Vs
    Av = 2 * (np.pi * (stir_db/2)**2) 
    if spacing <= 0: spacing = 1000 
    
    Vs = (Av * fy * d) / spacing
    phi_Vs = phi * Vs
    phi_Vn = phi_Vc + phi_Vs
    
    status = "OK" if phi_Vn >= Vu else "FAIL"
    return status, phi_Vn/1000, phi_Vc/1000, phi_Vs/1000, Vc, Vs

# --- 4. MAIN APPLICATION ---

st.markdown('<div class="main-header">🏗️ RC Beam Analysis & Design Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Timoshenko / Finite Element Method</div>', unsafe_allow_html=True)
st.markdown("---")

# --- 4.1 SIDEBAR: INPUTS ---
with st.sidebar:
    st.header("📝 Project Information")
    project_name = st.text_input("Project Name", "Residential Building A")
    engineer_name = st.text_input("Engineer", "Eng. Somchai")
    st.markdown("---")
    
    # เรียก Input Handler
    params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()

if not stable:
    st.error("🚨 **Structure Error:** โครงสร้างไม่เสถียร (Unstable)! กรุณาตรวจสอบจุดรองรับ (Support)")
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

 # --- 4.3 LOAD CALCULATION PROCESS & SOLVER ---
    try:
        # Calculate Self-weight
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

        # ----------------------------------------------------------------------
        # RUN SOLVER
        # ----------------------------------------------------------------------
        with st.spinner('Running Analysis...'):
            x_eval, M, V, D, R = solver.solve_beam(spans, sup_df, calc_loads_df, params)
        
        # สร้าง DataFrame หลักตัวเดียว (Master DataFrame)
        # หมายเหตุ: solver ปกติจะ return หน่วย SI พื้นฐาน (N, mm, N-mm)
        master_df = pd.DataFrame({
            'x': x_eval,                 # m
            'M_Nmm': M,                  # N-mm
            'V_N': V,                    # N
            'D_m': D                     # m
        })

        # คำนวณหน่วย Engineering (kNm, kN, mm) เตรียมไว้เลย
        master_df['M_kNm'] = master_df['M_Nmm'] / 1000.0
        master_df['V_kN'] = master_df['V_N'] / 1000.0
        master_df['D_mm'] = master_df['D_m'] * 1000.0

        # DataFrame สำหรับแสดงผล (เปลี่ยนชื่อ Column ให้สวยงาม)
        res_df_display = master_df.copy()
        res_df_display.rename(columns={
            'x': 'x (m)',
            'M_Nmm': 'Moment (N-mm)',
            'M_kNm': 'Moment (kNm)',
            'V_N': 'Shear (N)',
            'V_kN': 'Shear (kN)',
            'D_mm': 'Deflection (mm)'
        }, inplace=True)

        # --- 5. TABS INTERFACE ---
        tab1, tab2 = st.tabs(["📊 1. Analysis Results", "📝 2. Concrete Design & Detailing"])
        
        # ================= TAB 1: ANALYSIS =================
        with tab1:
            st.subheader("📈 Force Diagrams")
            
            # 1. Plot Diagram (ส่ง master_df หรือ res_df_display ที่มีหน่วยครบถ้วน)
            # ต้องมั่นใจว่า design_view รองรับชื่อคอลัมน์ใหม่ หรือเราส่งแบบเดิม
            # เพื่อความชัวร์ เราสร้าง df แบบเดิมส่งให้ฟังก์ชันวาดกราฟ
            df_for_plot = pd.DataFrame({
                'x': x_eval,
                'moment': M, # N-mm
                'shear': V,  # N
                'deflection': D * 1000 # mm
            })
            
            if not df_for_plot.empty:
                st.plotly_chart(design_view.plot_analysis_results(df_for_plot, spans, sup_df, calc_loads_df, R), use_container_width=True)
            else:
                st.info("ℹ️ Please input data and click 'Analyze'")

            # 2. Key Metrics
            st.markdown("### 📌 Critical Values (Global)")
            if not master_df.empty:
                v_max_kN = master_df['V_kN'].abs().max()
                
                # Global Max/Min Moment
                g_max_m = master_df['M_kNm'].max()
                g_min_m = master_df['M_kNm'].min()

                # แยกคิดค่าบวกและลบ
                m_max_pos_kNm = g_max_m if g_max_m > 0 else 0.0
                m_max_neg_kNm = abs(g_min_m) if g_min_m < 0 else 0.0
                
                d_abs_max_mm = master_df['D_mm'].abs().max()
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Max Shear (Vu)", f"{v_max_kN:.2f} kN")
                m2.metric("Max Moment (+)", f"{m_max_pos_kNm:.2f} kNm")
                m3.metric("Max Moment (-)", f"{m_max_neg_kNm:.2f} kNm")
                m4.metric("Max Deflection", f"{d_abs_max_mm:.2f} mm")

            # 3. Support Reactions & Check
            st.markdown("### 📍 Support Reactions & Checks")
            
            if R:
                col_r1, col_r2 = st.columns([1, 2])
                with col_r1:
                    reaction_data = [{"Node": int(str(k).replace('R', '')), "Reaction (kN)": v/1000.0} for k, v in R.items()]
                    df_reac = pd.DataFrame(reaction_data).sort_values(by="Node")
                    st.dataframe(
                        df_reac.style.format({"Reaction (kN)": "{:.2f}"}).background_gradient(cmap="Blues", subset=["Reaction (kN)"]),
                        use_container_width=True, hide_index=True
                    )
                
                with col_r2:
                    st.write("**Equilibrium Check (ΣFy = 0):**")
                    sum_R_kN = sum(R.values()) / 1000.0
                    total_applied_kN = 0.0
                    if not calc_loads_df.empty:
                        for _, l in calc_loads_df.iterrows():
                            if l['type'] == 'P': 
                                total_applied_kN += l['mag'] / 1000.0
                            elif l['type'] == 'U':
                                dist = l.get('dist', 0)
                                total_applied_kN += (l['mag'] * dist) / 1000.0

                    diff = abs(sum_R_kN - total_applied_kN)
                    is_balanced = diff < 0.1 

                    if is_balanced:
                        st.success(f"✅ **Balanced** | Diff: {diff:.4f} kN")
                    else:
                        st.error(f"⚠️ **Unbalanced** | Diff: {diff:.4f} kN")

                    c1, c2 = st.columns(2)
                    c1.metric("Total Reactions (Up)", f"{sum_R_kN:.2f} kN")
                    c2.metric("Total Loads (Down)", f"{total_applied_kN:.2f} kN")

            # 4. Export
            if not res_df_display.empty:
                st.markdown("---")
                csv = res_df_display.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Analysis Results (CSV)",
                    data=csv,
                    file_name=f'Analysis_Results.csv',
                    mime='text/csv',
                    type='primary'
                )

        # ================= TAB 2: INTERACTIVE DESIGN =================
        with tab2:
            st.header(f"🏗️ Interactive RC Design ({tag})")
            
            if is_service:
                st.warning("⚠️ Warning: Service Load Mode (Factor=1.0). Please switch to Ultimate for design.")
            
            b_mm, h_mm = params['b'] * 1000, params['h'] * 1000
            fc, fy = params['fc'], params['fy']
            
            final_design_res = []
            offsets = [0] + list(np.cumsum(spans))
            
            full_cal_report = f"PROJECT: {project_name}\nENGINEER: {engineer_name}\n"
            full_cal_report += f"DATE: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            full_cal_report += "="*60 + "\n"
            full_cal_report += f"DESIGN PARAMETERS:\n  fc' = {fc} MPa\n  fy = {fy} MPa\n  Section: {b_mm:.0f}x{h_mm:.0f} mm\n"
            full_cal_report += f"  Load Factors: DL={f_dl}, LL={f_ll}\n"
            full_cal_report += "="*60 + "\n\n"

            # --- SPAN LOOP ---
            for i in range(n_spans):
                s_len = spans[i]
                s_start, s_end = offsets[i], offsets[i+1]
                
                # ดึงข้อมูลจาก Master DataFrame โดยใช้คอลัมน์ kNm ที่คำนวณไว้แล้ว
                # ใช้ buffer เล็กน้อย (+/- 1e-6) เพื่อกัน Floating point error
                span_data = master_df[(master_df['x'] >= s_start - 1e-6) & (master_df['x'] <= s_end + 1e-6)]
                
                # --- CALCULATION LOGIC ---
                if not span_data.empty:
                    # Positive Moment (Design Bottom Steel)
                    # หาค่าสูงสุดใน span นี้
                    raw_max_kNm = span_data['M_kNm'].max()
                    mu_pos = max(0.0, raw_max_kNm) # ถ้าค่า max เป็นลบ (คานยื่น) ให้ถือว่าเป็น 0 สำหรับเหล็กล่าง
                    
                    # Negative Moment (Design Top Steel)
                    # หาค่าต่ำสุด (ที่เป็นลบมากที่สุด) แล้วแปลงเป็น Absolute
                    raw_min_kNm = span_data['M_kNm'].min()
                    mu_neg = abs(raw_min_kNm) if raw_min_kNm < 0 else 0.0
                    
                    vu_max = span_data['V_kN'].abs().max()
                else:
                    mu_pos, mu_neg, vu_max = 0, 0, 0

                # --- DEBUG CHECKER ---
                # ส่วนนี้สำคัญ: ช่วยให้คุณเช็คว่าค่าที่ Code เห็น ตรงกับที่คุณคิดไหม
                with st.expander(f"🔍 Debug Data Check: Span {i+1}"):
                    st.write(f"**Range X:** {s_start:.2f} to {s_end:.2f} m")
                    st.write(f"**Raw Max kNm in Data:** {raw_max_kNm if 'raw_max_kNm' in locals() else 'No Data'}")
                    st.write(f"**Raw Min kNm in Data:** {raw_min_kNm if 'raw_min_kNm' in locals() else 'No Data'}")
                    st.write(f"👉 **Used for Design:** Mu(+) = {mu_pos:.2f}, Mu(-) = {mu_neg:.2f}")
                    if not span_data.empty:
                        st.dataframe(span_data[['x', 'M_kNm', 'V_kN']].describe())

                # --- REPORT WRITING ---
                full_cal_report += f"\n>> SPAN {i+1} (Length {s_len} m)\n"
                full_cal_report += f"   Design Forces: Mu(+)={mu_pos:.2f} kNm, Mu(-)={mu_neg:.2f} kNm, Vu={vu_max:.2f} kN\n"

                # --- UI DISPLAY ---
                with st.expander(f"📍 **Span {i+1}** (L={s_len} m) | Forces: $M_u^+$ {mu_pos:.2f} kNm, $M_u^-$ {mu_neg:.2f} kNm, $V_u$ {vu_max:.2f} kN", expanded=True):
                    
                    c_const, c_cov = st.columns([3, 1])
                    with c_const:
                        st.caption(f"Design Constants: fc'={fc}, fy={fy}, Size {b_mm:.0f}x{h_mm:.0f} mm")
                    with c_cov:
                        cover_mm = st.number_input(f"Covering (mm)", value=25.0, step=5.0, key=f"cov_{i}")

                    # 1. Bottom Steel (+Moment)
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
                        st.markdown(f"**Area**: $A_{{s,prov}} =$ :{clr_b}[**{as_prov_bot:.0f}**] **mm²** vs $A_{{req}} =$ **{as_req_bot:.0f}** **mm²**")
                        st.markdown(f"**Strength**: $\phi M_n =$ :{clr_b}[**{phi_Mn_bot:.2f}**] **kNm** $\ge M_u =$ **{mu_pos:.2f}** **kNm**")
                    
                    full_cal_report += f"   [Bottom] Prov: {bot_n}-DB{bot_db} (As={as_prov_bot:.0f}), phiMn={phi_Mn_bot:.2f} >= Mu={mu_pos:.2f} -> {icon_b}\n"

                    # 2. Top Steel (-Moment)
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
                        st.markdown(f"**Area**: $A_{{s,prov}} =$ :{clr_t}[**{as_prov_top:.0f}**] **mm²** vs $A_{{req}} =$ **{as_req_top:.0f}** **mm²**")
                        st.markdown(f"**Strength**: $\phi M_n =$ :{clr_t}[**{phi_Mn_top:.2f}**] **kNm** $\ge M_u =$ **{mu_neg:.2f}** **kNm**")

                    full_cal_report += f"   [Top]    Prov: {top_n}-DB{top_db} (As={as_prov_top:.0f}), phiMn={phi_Mn_top:.2f} >= Mu={mu_neg:.2f} -> {icon_t}\n"

                    # 3. Shear
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

            col_act1, col_act2 = st.columns(2)
            with col_act1:
                if st.button("🔄 Generate/Update Drawings", type="primary"):
                    try:
                        st.write("**Longitudinal Section:**")
                        fig_long = section_plotter.plot_longitudinal_section_detailed(spans, sup_df, final_design_res, params['h'], final_design_res[0]['cover'])
                        st.pyplot(fig_long, use_container_width=True)
                        
                        st.write("**Cross Section (Typical Span 1):**")
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
                st.download_button(
                    label="📄 Download Calculation Report (.txt)",
                    data=full_cal_report,
                    file_name=f"Design_Report_{project_name}.txt",
                    mime="text/plain"
                )
                
                with st.expander("View Report Preview"):
                    st.text(full_cal_report)

    except Exception as e:
        st.error(f"❌ Calculation Error: {e}")
        st.warning("Please check your input loads or support conditions.")
        st.exception(e)

