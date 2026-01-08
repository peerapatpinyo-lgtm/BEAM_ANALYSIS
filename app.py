# app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import time

# --- 1. IMPORT CUSTOM MODULES ---
import input_handler
import solver
import design_view
import section_plotter
import reporter

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

def prepare_load_dataframe(raw_loads_df, n_spans, spans, params, f_dl, f_ll):
    """Helper function to prepare load dataframe for solver"""
    # 1. Self-weight
    w_sw_base_kN = params['b'] * params['h'] * 24.0      
    w_sw_factored_kN = w_sw_base_kN * f_dl
    
    span_total_udl_N = {i: w_sw_factored_kN * 1000.0 for i in range(n_spans)} 
    combined_loads_list = []
    
    # 2. User Loads
    if not raw_loads_df.empty:
        for _, row in raw_loads_df.iterrows():
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
                        'd_start': d_start, 'dist': 0.0
                    })
                elif l_type == 'U':
                    if d_start <= 0.01 and dist >= (spans[s_idx] - 0.01):
                        span_total_udl_N[s_idx] += mag_factored_N
                    else:
                        combined_loads_list.append({
                            'span_index': s_idx, 'type': 'U', 'mag': mag_factored_N, 
                            'd_start': d_start, 'dist': dist
                        })
            except Exception: continue
    
    # Add Self-weight + Full Span UDLs
    for i in range(n_spans):
        if span_total_udl_N[i] > 0:
            combined_loads_list.append({
                'span_index': i, 'type': 'U', 'mag': span_total_udl_N[i], 
                'd_start': 0.0, 'dist': spans[i]
            })
            
    return pd.DataFrame(combined_loads_list)

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
        with st.spinner('Running Analysis...'):
            # =================================================================
            # RUN 1: ULTIMATE LOAD ANALYSIS (For Strength Design)
            # =================================================================
            calc_loads_ult = prepare_load_dataframe(loads_df, n_spans, spans, params, f_dl, f_ll)
            x_ult, M_ult, V_ult, D_ult, R_ult = solver.solve_beam(spans, sup_df, calc_loads_ult, params)
            
            # =================================================================
            # RUN 2: SERVICE LOAD ANALYSIS (For Deflection Check)
            # =================================================================
            # Force factors to 1.0 for Serviceability Limit State
            calc_loads_svc = prepare_load_dataframe(loads_df, n_spans, spans, params, 1.0, 1.0)
            x_svc, M_svc, V_svc, D_svc, R_svc = solver.solve_beam(spans, sup_df, calc_loads_svc, params)

        # --- PREPARE DATA FOR DISPLAY (Based on User Selection) ---
        # If user selected Service Mode, show Service results in graphs.
        # If Ultimate Mode, show Ultimate results.
        if is_service:
            x_plot, M_plot, V_plot, D_plot, R_plot = x_svc, M_svc, V_svc, D_svc, R_svc
            display_loads = calc_loads_svc
        else:
            x_plot, M_plot, V_plot, D_plot, R_plot = x_ult, M_ult, V_ult, D_ult, R_ult
            display_loads = calc_loads_ult

        # Master DataFrame for Plotting (Current Mode)
        master_df = pd.DataFrame({
            'x': x_plot,
            'M_Nmm': M_plot,
            'V_N': V_plot,
            'D_m': D_plot
        })
        master_df['M_kNm'] = master_df['M_Nmm'] / 1000.0
        master_df['V_kN'] = master_df['V_N'] / 1000.0
        master_df['D_mm'] = master_df['D_m'] * 1000.0
        
        # --- 5. TABS INTERFACE ---
        tab1, tab2, tab3 = st.tabs(["📊 1. Analysis Results", "📝 2. Concrete Design & Detailing", "📘 3. Detailed Calculation Report"])
        
        final_design_res = []

        # ================= TAB 1: ANALYSIS =================
        with tab1:
            st.subheader(f"📈 Force Diagrams ({tag} Load)")
            
            df_for_plot = pd.DataFrame({
                'x': x_plot,
                'moment': M_plot, # N-mm
                'shear': V_plot,  # N
                'deflection': D_plot * 1000 # mm
            })
            
            if not df_for_plot.empty:
                st.plotly_chart(design_view.plot_analysis_results(df_for_plot, spans, sup_df, display_loads, R_plot), use_container_width=True)
            
            # Key Metrics
            v_max = master_df['V_kN'].abs().max()
            m_max = master_df['M_kNm'].max()
            m_min = master_df['M_kNm'].min()
            d_max = master_df['D_mm'].abs().max()
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Max Shear", f"{v_max:.2f} kN")
            c2.metric("Max Moment (+)", f"{max(0, m_max):.2f} kNm")
            c3.metric("Max Moment (-)", f"{abs(min(0, m_min)):.2f} kNm")
            c4.metric("Max Deflection", f"{d_max:.2f} mm")

        # ================= TAB 2: INTERACTIVE DESIGN =================
        with tab2:
            st.header(f"🏗️ Interactive RC Design")
            if is_service:
                st.warning("⚠️ You are in Service Mode. Design should be based on Ultimate Loads.")
            
            b_mm, h_mm = params['b'] * 1000, params['h'] * 1000
            fc, fy = params['fc'], params['fy']
            offsets = [0] + list(np.cumsum(spans))
            
            # --- SPAN LOOP ---
            for i in range(n_spans):
                s_len = spans[i]
                s_start, s_end = offsets[i], offsets[i+1]
                
                # 1. Get ULTIMATE Forces for Strength Design
                mask_ult = (x_ult >= s_start - 1e-6) & (x_ult <= s_end + 1e-6)
                if any(mask_ult):
                    mu_pos = max(0.0, (M_ult[mask_ult] / 1000.0).max())
                    mu_neg = abs(min(0.0, (M_ult[mask_ult] / 1000.0).min()))
                    vu_max = abs((V_ult[mask_ult] / 1000.0)).max()
                else:
                    mu_pos, mu_neg, vu_max = 0, 0, 0
                
                # 2. Get SERVICE Forces for Deflection Check
                mask_svc = (x_svc >= s_start - 1e-6) & (x_svc <= s_end + 1e-6)
                if any(mask_svc):
                    ma_pos_svc = max(0.0, (M_svc[mask_svc] / 1000.0).max())
                    delta_svc_mm = abs((D_svc[mask_svc] * 1000.0)).max()
                else:
                    ma_pos_svc, delta_svc_mm = 0, 0

                # --- UI DISPLAY ---
                with st.expander(f"📍 **Span {i+1}** (L={s_len} m) | Strength Design Forces", expanded=True):
                    
                    c_const, c_cov = st.columns([3, 1])
                    with c_const:
                        st.caption(f"Design Constants: fc'={fc}, fy={fy}, Size {b_mm:.0f}x{h_mm:.0f} mm")
                    with c_cov:
                        cover_mm = st.number_input(f"Covering (mm)", value=25.0, step=5.0, key=f"cov_{i}")

                    # 1. Bottom Steel (+Moment)
                    st.markdown("##### 1. Bottom Reinforcement (Mid-Span)")
                    d_eff_bot_est = h_mm - cover_mm - 20
                    as_req_bot, _, _ = get_as_req(mu_pos, d_eff_bot_est, fc, fy, b_mm)
                    
                    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
                    with c1: st.markdown(f"**Req $A_s$:**\n`{as_req_bot:.0f}` mm²")
                    with c2: bot_db = st.selectbox("DB", [12, 16, 20, 25, 28], index=1, key=f"bdb_{i}")
                    with c3: bot_n = st.number_input("Qty", 2, 10, 2, key=f"bn_{i}")
                    
                    d_eff_bot_real = h_mm - cover_mm - 9 - (bot_db / 2)
                    phi_Mn_bot, as_prov_bot, _, _, _, _ = get_phi_Mn_details(bot_n, bot_db, d_eff_bot_real, b_mm, fc, fy)
                    pass_b = phi_Mn_bot >= mu_pos
                    
                    with c4: 
                        clr_b = "green" if pass_b else "red"
                        st.markdown(f"$\phi M_n$: :{clr_b}[**{phi_Mn_bot:.2f}**] kNm vs $M_u$: **{mu_pos:.2f}**")
                    
                    # 2. Top Steel (-Moment)
                    st.markdown("##### 2. Top Reinforcement (Supports)")
                    d_eff_top_est = h_mm - cover_mm - 20
                    as_req_top, _, _ = get_as_req(mu_neg, d_eff_top_est, fc, fy, b_mm)
                    
                    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
                    with c1: st.markdown(f"**Req $A_s$:**\n`{as_req_top:.0f}` mm²")
                    with c2: top_db = st.selectbox("DB", [12, 16, 20, 25, 28], index=1, key=f"tdb_{i}")
                    with c3: top_n = st.number_input("Qty", 2, 10, 2, key=f"tn_{i}")
                    
                    d_eff_top_real = h_mm - cover_mm - 9 - (top_db / 2)
                    phi_Mn_top, as_prov_top, _, _, _, _ = get_phi_Mn_details(top_n, top_db, d_eff_top_real, b_mm, fc, fy)
                    pass_t = phi_Mn_top >= mu_neg
                    
                    with c4:
                        clr_t = "green" if pass_t else "red"
                        st.markdown(f"$\phi M_n$: :{clr_t}[**{phi_Mn_top:.2f}**] kNm vs $M_u$: **{mu_neg:.2f}**")

                    # 3. Shear
                    st.markdown("##### 3. Shear Reinforcement")
                    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
                    with c1: st.markdown(f"**Design $V_u$:**\n`{vu_max:.2f}` kN")
                    with c2: stir_db = st.selectbox("Stirrup", [6, 9, 12], index=0, key=f"sdb_{i}")
                    with c3: stir_s = st.number_input("Spacing (mm)", 50, 300, 150, 10, key=f"ss_{i}")
                    
                    d_shear = d_eff_bot_real 
                    status_v, phi_Vn, phi_Vc, phi_Vs, _, _ = check_shear_details(vu_max, b_mm, d_shear, fc, fy, stir_db, stir_s)
                    
                    with c4:
                        clr_v = "green" if status_v == "OK" else "red"
                        st.markdown(f"$\phi V_n$: :{clr_v}[**{phi_Vn:.1f}**] kN")
                    
                    # Store results for Report
                    final_design_res.append({
                        'span_id': i,
                        'L': s_len,
                        'Mu_pos': mu_pos,
                        'Mu_neg': mu_neg,
                        'Vu_max': vu_max,
                        'cover': cover_mm,
                        'top_db': top_db, 'bot_db': bot_db, 'stir_db': stir_db,
                        'pos': {'n': bot_n, 'area': as_prov_bot, 'status': pass_b},
                        'neg': {'n': top_n, 'area': as_prov_top, 'status': pass_t},
                        'shear': {'s': stir_s, 'status': status_v},
                        # Service Load Results for Report
                        'Ma_pos_svc': ma_pos_svc,
                        'delta_svc_mm': delta_svc_mm
                    })

            # --- SUMMARY & REPORT ---
            st.markdown("---")
            st.subheader("📋 Design Summary")
            
            summary_data = []
            for item in final_design_res:
                summary_data.append({
                    "Span": item['span_id'] + 1,
                    "Bottom": f"{item['pos']['n']}-DB{item['bot_db']}",
                    "Top": f"{item['neg']['n']}-DB{item['top_db']}",
                    "Stirrup": f"RB{item['stir_db']}@{item['shear']['s']}",
                    "Status": "✅ Pass" if (item['pos']['status'] and item['neg']['status'] and item['shear']['status'] == "OK") else "❌ Fail"
                })
            st.table(pd.DataFrame(summary_data))
            
            # Drawings Button
            if st.button("🔄 Generate Drawings", type="primary"):
                try:
                    st.write("**Longitudinal Section:**")
                    fig_long = section_plotter.plot_longitudinal_section_detailed(spans, sup_df, final_design_res, params['h'], final_design_res[0]['cover'])
                    st.pyplot(fig_long, use_container_width=True)
                except Exception as e:
                    st.error(f"Drawing Error: {e}")

        # ================= TAB 3: DETAILED REPORT =================
        with tab3:
            st.header("📝 Detailed Calculation Reports")
            st.markdown(f"**Project:** {project_name} | **Engineer:** {engineer_name}")
            
            if not final_design_res:
                st.warning("⚠️ Please complete the design in Tab 2 first.")
            else:
                for i, res in enumerate(final_design_res):
                    with st.expander(f"📘 Calculation Sheet: Span {i+1}", expanded=False):
                        reporter.render_calculation_report(
                            span_idx=i,
                            span_len=res['L'],
                            b=params['b'],
                            h=params['h'],
                            fc=params['fc'],
                            fy=params['fy'],
                            Mu_pos=res['Mu_pos'],
                            Mu_neg=res['Mu_neg'],
                            Vu=res['Vu_max'],
                            res_data=res,
                            # Pass Service Results here
                            Ma_pos=res['Ma_pos_svc'],
                            delta_analysis_mm=res['delta_svc_mm']
                        )

    except Exception as e:
        st.error(f"❌ Application Error: {e}")
        st.exception(e)
