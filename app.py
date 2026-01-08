import streamlit as st
import pandas as pd
import numpy as np
import time

# --- 1. IMPORT CUSTOM MODULES ---
# ตรวจสอบว่ามีไฟล์เหล่านี้อยู่ในโฟลเดอร์เดียวกัน
try:
    import input_handler
    import solver
    import design_view
    import section_plotter
except ImportError as e:
    st.error(f"❌ Missing Module: {e}. Please ensure input_handler.py, solver.py, design_view.py, and section_plotter.py are in the same directory.")
    st.stop()

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
    if Mu_kNm == 0: return 0.0, 0.0, False
    Mu = abs(Mu_kNm) * 1e6 
    phi = 0.9 
    
    m = fy / (0.85 * fc)
    Rn = Mu / (phi * b_mm * d_eff_mm**2)
    
    rho = 0.0
    is_error = False
    
    try:
        term = 1 - (2 * m * Rn) / fy
        if term < 0:
            rho = 0.0 
            is_error = True 
        else:
            rho = (1/m) * (1 - np.sqrt(term))
    except:
        rho = 0.0
        is_error = True
    
    as_req = rho * b_mm * d_eff_mm
    
    if as_req > 0 and not is_error: 
        as_min1 = (0.25 * np.sqrt(fc) / fy) * b_mm * d_eff_mm
        as_min2 = (1.4 / fy) * b_mm * d_eff_mm
        as_min = max(as_min1, as_min2)
        return max(as_req, as_min), rho, False
    
    return as_req, rho, is_error

def get_phi_Mn_details(n, db, d_eff, b, fc, fy):
    Ast = n * (np.pi * (db/2)**2)
    if Ast == 0: return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    a = (Ast * fy) / (0.85 * fc * b)
    beta1 = 0.85 if fc <= 30 else max(0.65, 0.85 - 0.05 * (fc - 30) / 7)
    c = a / beta1
    
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
    Vu = abs(Vu_kN) * 1000 # N
    
    Vc = 0.17 * np.sqrt(fc) * b * d
    phi = 0.85
    phi_Vc = phi * Vc
    
    Av = 2 * (np.pi * (stir_db/2)**2)
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
    
    params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()

# Check Stability
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

    # --- 4.3 LOAD CALCULATION PROCESS ---
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
        res_df = pd.DataFrame() # Initialize empty
        R = {}

        if st.button("🚀 Analyze & Design", type="primary"):
            with st.spinner('Running Analysis...'):
                # Call Solver
                x_eval, M, V, D, R = solver.solve_beam(spans, sup_df, calc_loads_df, params)
                
                # Create raw result DataFrame
                res_df = pd.DataFrame({
                    'x': x_eval,
                    'moment': M, 
                    'shear': V,    
                    'deflection': D * 1000 # Convert m to mm
                })

        # --- 0. PREPARE DATA FOR DISPLAY (SAFETY BLOCK) ---
        # สร้าง Dataframe สำหรับแสดงผลและคำนวณต่อ โดยไม่กระทบตัวแปรต้นฉบับ
        res_df_display = pd.DataFrame()
        
        if not res_df.empty:
            res_df_display = res_df.copy()
            
            # Create Display Columns (Standardizing)
            # 1. Rename columns to be User Friendly
            res_df_display.rename(columns={
                'x': 'x (m)',
                'moment': 'Moment (N-mm)',
                'shear': 'Shear (N)',
                'deflection': 'Deflection (mm)'
            }, inplace=True)

            # 2. Add Engineering Units for Design (kN, kNm)
            # หารด้วย 1e6 หรือ 1000 จากคอลัมน์ที่มีอยู่แล้วแน่นอน
            res_df_display['Moment (kNm)'] = res_df_display['Moment (N-mm)'] / 1e6
            res_df_display['Shear (kN)'] = res_df_display['Shear (N)'] / 1000
        
        # --- 5. TABS INTERFACE ---
        tab1, tab2 = st.tabs(["📊 1. Analysis Results", "📝 2. Concrete Design & Detailing"])
        
        # ================= TAB 1: ANALYSIS =================
        with tab1:
            st.subheader("📈 Force Diagrams")
            
            if not res_df_display.empty:
                # Plotting Function
                st.plotly_chart(design_view.plot_analysis_results(res_df, spans, sup_df, calc_loads_df, R), use_container_width=True)

                # Key Metrics
                st.markdown("### 📌 Critical Values")
                v_max_kN = res_df_display['Shear (kN)'].abs().max()
                
                raw_max_m = res_df_display['Moment (kNm)'].max()
                raw_min_m = res_df_display['Moment (kNm)'].min()
                m_max_pos_kNm = raw_max_m if raw_max_m > 0 else 0.0
                m_max_neg_kNm = abs(raw_min_m) if raw_min_m < 0 else 0.0
                
                d_abs_max_mm = res_df_display['Deflection (mm)'].abs().max()
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Max Shear (Vu)", f"{v_max_kN:.2f} kN")
                m2.metric("Max Moment (+)", f"{m_max_pos_kNm:.2f} kNm")
                m3.metric("Max Moment (-)", f"{m_max_neg_kNm:.2f} kNm")
                m4.metric("Max Deflection", f"{d_abs_max_mm:.2f} mm")

                # Reactions
                st.markdown("### 📍 Support Reactions")
                col_r1, col_r2 = st.columns([1, 2])
                with col_r1:
                    reaction_data = [{"Node": int(str(k).replace('R', '')), "Reaction (kN)": v/1000.0} for k, v in R.items()]
                    df_reac = pd.DataFrame(reaction_data).sort_values(by="Node")
                    st.dataframe(df_reac.style.format({"Reaction (kN)": "{:.2f}"}).background_gradient(cmap="Blues"), use_container_width=True, hide_index=True)
                
                with col_r2:
                    st.write("**Equilibrium Check (ΣFy = 0):**")
                    sum_R_kN = sum(R.values()) / 1000.0
                    total_applied_kN = 0.0
                    if not calc_loads_df.empty:
                        for _, l in calc_loads_df.iterrows():
                            if l['type'] == 'P': total_applied_kN += l['mag']
                            elif l['type'] == 'U': total_applied_kN += (l['mag'] * l['dist'])
                    
                    diff = abs(sum_R_kN - total_applied_kN/1000.0) # Load in calc_loads_df is N, need /1000
                    if diff < 0.1:
                        st.success(f"✅ Balanced | Diff: {diff:.3f} kN")
                    else:
                        st.error(f"⚠️ Unbalanced | Diff: {diff:.3f} kN")

                # Export
                csv = res_df_display.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results (CSV)", csv, "results.csv", "text/csv")
            
            else:
                st.info("👈 Please adjust inputs and click 'Analyze & Design' to start.")

        # ================= TAB 2: INTERACTIVE DESIGN =================
        with tab2:
            st.header(f"🏗️ Interactive RC Design ({tag})")
            
            if res_df_display.empty:
                st.warning("Please run Analysis first.")
            else:
                # Design Constants
                b_mm, h_mm = params['b'] * 1000, params['h'] * 1000
                fc, fy = params['fc'], params['fy']
                
                final_design_res = []
                offsets = [0] + list(np.cumsum(spans))
                
                full_cal_report = f"PROJECT: {project_name}\nENGINEER: {engineer_name}\nDATE: {time.strftime('%Y-%m-%d')}\n"
                
                # --- SPAN LOOP ---
                for i in range(n_spans):
                    s_len = spans[i]
                    s_start, s_end = offsets[i], offsets[i+1]
                    
                    # Filter Data for this Span
                    span_data = res_df_display[(res_df_display['x (m)'] >= s_start - 1e-4) & (res_df_display['x (m)'] <= s_end + 1e-4)]
                    
                    if not span_data.empty:
                        raw_max = span_data['Moment (kNm)'].max()
                        mu_pos = max(0, raw_max) 
                        
                        raw_min = span_data['Moment (kNm)'].min()
                        mu_neg = abs(raw_min) if raw_min < 0 else 0
                        
                        vu_max = span_data['Shear (kN)'].abs().max()
                    else:
                        mu_pos, mu_neg, vu_max = 0, 0, 0

                    full_cal_report += f"\n>> SPAN {i+1} (Length {s_len} m) | Mu+={mu_pos:.2f}, Mu-={mu_neg:.2f}, Vu={vu_max:.2f}\n"

                    with st.expander(f"📍 **Span {i+1}** (L={s_len} m) | Forces: $M_u^+$ {mu_pos:.2f} kNm, $M_u^-$ {mu_neg:.2f} kNm", expanded=True):
                        
                        c_const, c_cov = st.columns([3, 1])
                        with c_const: st.caption(f"Section: {b_mm:.0f}x{h_mm:.0f} mm | fc'={fc}, fy={fy}")
                        with c_cov: cover_mm = st.number_input(f"Cover (mm)", value=25.0, step=5.0, key=f"cov_{i}")

                        # 1. Bottom Steel
                        st.markdown("##### 1. Bottom Reinforcement ($+M_u$)")
                        d_eff_bot_est = h_mm - cover_mm - 9 - 10 
                        as_req_bot, rho_bot, err_bot = get_as_req(mu_pos, d_eff_bot_est, fc, fy, b_mm)
                        
                        c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
                        with c1: st.metric("Req As", f"{as_req_bot:.0f} mm²")
                        with c2: bot_db = st.selectbox("DB", [12, 16, 20, 25, 28], index=1, key=f"bdb_{i}")
                        with c3: bot_n = st.number_input("Qty", 2, 10, 2, key=f"bn_{i}")
                        
                        d_eff_bot_real = h_mm - cover_mm - 9 - (bot_db / 2)
                        phi_Mn_bot, as_prov_bot, _, _, _, _ = get_phi_Mn_details(bot_n, bot_db, d_eff_bot_real, b_mm, fc, fy)
                        pass_b = phi_Mn_bot >= mu_pos
                        
                        with c4: 
                            icon_b = "✅" if pass_b else "❌"
                            st.write(f"{icon_b} **Prov:** {as_prov_bot:.0f} mm² | **Cap:** {phi_Mn_bot:.2f} kNm")

                        # 2. Top Steel
                        st.markdown("##### 2. Top Reinforcement ($-M_u$)")
                        d_eff_top_est = h_mm - cover_mm - 9 - 10 
                        as_req_top, rho_top, err_top = get_as_req(mu_neg, d_eff_top_est, fc, fy, b_mm)
                        
                        c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
                        with c1: st.metric("Req As", f"{as_req_top:.0f} mm²")
                        with c2: top_db = st.selectbox("DB", [12, 16, 20, 25, 28], index=1, key=f"tdb_{i}")
                        with c3: top_n = st.number_input("Qty", 2, 10, 2, key=f"tn_{i}")
                        
                        d_eff_top_real = h_mm - cover_mm - 9 - (top_db / 2)
                        phi_Mn_top, as_prov_top, _, _, _, _ = get_phi_Mn_details(top_n, top_db, d_eff_top_real, b_mm, fc, fy)
                        pass_t = phi_Mn_top >= mu_neg
                        
                        with c4:
                            icon_t = "✅" if pass_t else "❌"
                            st.write(f"{icon_t} **Prov:** {as_prov_top:.0f} mm² | **Cap:** {phi_Mn_top:.2f} kNm")

                        # 3. Shear
                        st.markdown("##### 3. Shear ($V_u$)")
                        c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
                        with c1: st.metric("Vu", f"{vu_max:.2f} kN")
                        with c2: stir_db = st.selectbox("Stirrup", [6, 9, 12], index=0, key=f"sdb_{i}")
                        with c3: stir_s = st.number_input("Spacing", 50, 300, 150, 10, key=f"ss_{i}")
                        
                        d_shear = d_eff_bot_real 
                        status_v, phi_Vn, phi_Vc, phi_Vs, _, _ = check_shear_details(vu_max, b_mm, d_shear, fc, fy, stir_db, stir_s)
                        
                        with c4:
                            icon_v = "✅" if status_v == "OK" else "❌"
                            st.write(f"{icon_v} **Cap:** {phi_Vn:.2f} kN")

                        # Store Data
                        final_design_res.append({
                            'span': i+1, 'cover': cover_mm,
                            'top_db': top_db, 'bot_db': bot_db, 'stir_db': stir_db,
                            'pos': {'n': bot_n, 'area': as_prov_bot, 'status': pass_b},
                            'neg': {'n': top_n, 'area': as_prov_top, 'status': pass_t},
                            'shear': {'s': stir_s, 'status': status_v}
                        })
                
                # --- SUMMARY ---
                st.markdown("---")
                st.subheader("📋 Summary Table")
                summary_data = []
                for item in final_design_res:
                    summary_data.append({
                        "Span": item['span'],
                        "Bot Steel": f"{item['pos']['n']}-DB{item['bot_db']}",
                        "Top Steel": f"{item['neg']['n']}-DB{item['top_db']}",
                        "Stirrup": f"RB{item['stir_db']}@{item['shear']['s']}",
                        "Status": "✅ PASS" if (item['pos']['status'] and item['neg']['status'] and item['shear']['status'] == "OK") else "❌ FAIL"
                    })
                st.table(pd.DataFrame(summary_data))
                
                # Drawing & Report
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    if st.button("🔄 Generate Drawings"):
                         # เรียก section_plotter (สมมติว่า function รับค่าถูกต้อง)
                         try:
                             fig = section_plotter.plot_longitudinal_section_detailed(spans, sup_df, final_design_res, params['h'], 25)
                             st.pyplot(fig, use_container_width=True)
                         except Exception as e:
                             st.error(f"Cannot plot: {e}")
                
                with col_d2:
                    st.download_button("📄 Download Report", full_cal_report, "Design_Report.txt")

    except Exception as e:
        st.error(f"❌ Application Error: {e}")
        st.caption("Try checking your inputs (Loads/Supports) or refreshing the page.")
