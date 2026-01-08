import streamlit as st
import pandas as pd
import numpy as np
import io

# --- 1. IMPORT CUSTOM MODULES ---
# ตรวจสอบให้แน่ใจว่าไฟล์ modules เหล่านี้อยู่ใน folder เดียวกัน
import input_handler
import solver
import design_view
import section_plotter

# --- HELPER FUNCTIONS: REAL-TIME CALCULATION ---

def get_as_req(Mu_kNm, d_eff_mm, fc, fy, b_mm):
    """
    คำนวณ As required โดยประมาณ
    Returns: as_req, rho_calc, is_error
    """
    if Mu_kNm == 0: return 0.0, 0.0, False
    Mu = abs(Mu_kNm) * 1e6 # N-mm
    phi = 0.9
    
    # คำนวณ Rho
    m = fy / (0.85 * fc)
    Rn = Mu / (phi * b_mm * d_eff_mm**2)
    
    rho = 0.0
    is_error = False
    
    try:
        # Check if Rn is too high (Section too small / Over reinforced limit check)
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
    """
    คำนวณ Capacity และส่งค่าตัวแปรย่อยกลับมาเพื่อทำ Report
    Returns: phi_Mn, Ast, a, Mn_raw, c_depth, strain_t
    """
    Ast = n * (np.pi * (db/2)**2)
    if Ast == 0: return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Whitney Stress Block
    a = (Ast * fy) / (0.85 * fc * b)
    beta1 = 0.85 if fc <= 30 else max(0.65, 0.85 - 0.05 * (fc - 30) / 7)
    c = a / beta1
    
    # Strain Check (Optional logic for precision)
    dt = d_eff # Assume single layer for simple calc
    if c > 0:
        strain_t = 0.003 * (dt - c) / c
    else:
        strain_t = 0.005 # Safe default
        
    # Phi factor adjustment (Spiral / Other) - Here we assume tension controlled for simplicity or fix at 0.9
    # For strict ACI: if strain_t >= 0.005, phi=0.9. If between 0.002 and 0.005, linear transition.
    phi = 0.9 
    if strain_t < 0.005:
         # Simplified warning, but keeping 0.9 for standard beam design unless very deep
         pass 

    Mn = Ast * fy * (d_eff - a/2)
    phi_Mn = phi * Mn / 1e6 # kNm
    return phi_Mn, Ast, a, Mn, c, strain_t

def check_shear_details(Vu_kN, b, d, fc, fy, stir_db, spacing):
    """
    คำนวณ Capacity แรงเฉือน และส่งค่าตัวแปรย่อยกลับมาเพื่อทำ Report
    Returns: status, phi_Vn, phi_Vc, phi_Vs, Vc_raw, Vs_raw
    """
    Vu = abs(Vu_kN) * 1000 # N
    Vc = 0.17 * np.sqrt(fc) * b * d
    phi = 0.85
    phi_Vc = phi * Vc
    
    Av = 2 * (np.pi * (stir_db/2)**2) # 2 legs
    if spacing <= 0: spacing = 1000 # prevent div by zero
    
    Vs = (Av * fy * d) / spacing
    phi_Vs = phi * Vs
    phi_Vn = phi_Vc + phi_Vs
    
    status = "OK" if phi_Vn >= Vu else "FAIL"
    return status, phi_Vn/1000, phi_Vc/1000, phi_Vs/1000, Vc, Vs

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(page_title="Beam Analysis & Design Pro", layout="wide", page_icon="🏗️")
st.title("🏗️ RC Beam Analysis & Design Pro (Timoshenko)")

# --- 3. SIDEBAR INPUTS ---
params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()

if not stable:
    st.error("🚨 Structure is unstable! Please check supports (Must have at least 3 reaction components).")
else:
    # --- 4. ANALYSIS SETTINGS & LOAD FACTORS ---
    st.markdown("### ⚙️ Analysis Settings & Load Factors")
    
    mode_select = st.radio(
        "Select Analysis Mode:",
        ["Service Load (1.0DL + 1.0LL)", "Ultimate / Custom Factors"],
        horizontal=True
    )
    
    col_fac1, col_fac2, _ = st.columns([1, 1, 2])
    
    is_service = False
    if mode_select.startswith("Service"):
        f_dl, f_ll = 1.0, 1.0
        with col_fac1:
            st.number_input("Dead Load Factor (DL)", value=1.0, disabled=True, key="fdl_serv")
        with col_fac2:
            st.number_input("Live Load Factor (LL)", value=1.0, disabled=True, key="fll_serv")
        tag = "Service"
        st.info("ℹ️ Using **Service Load** (Factors = 1.0) for Deflection & Serviceability checks.")
        is_service = True
    else:
        with col_fac1:
            f_dl = st.number_input("Dead Load Factor (DL)", value=1.4, step=0.1, format="%.2f", key="fdl_ult")
        with col_fac2:
            f_ll = st.number_input("Live Load Factor (LL)", value=1.7, step=0.1, format="%.2f", key="fll_ult")
        tag = "Ultimate"
        st.warning(f"⚡ Using **Factored Load**: {f_dl} DL + {f_ll} LL for Strength Design.")

    # --- 5. LOAD CALCULATIONS & COMBINATIONS ---
    try:
        # 5.1 Self-Weight Calculation
        w_sw_base_kN = params['b'] * params['h'] * 24.0     
        w_sw_factored_kN = w_sw_base_kN * f_dl
        
        # 5.2 Initialize Total UDL per span
        span_total_udl_N = {i: w_sw_factored_kN * 1000.0 for i in range(n_spans)} 
        combined_loads_list = []
        
        # 5.3 Process User-Defined Loads
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
                            'span_index': s_idx,
                            'type': 'P',
                            'mag': mag_factored_N, 
                            'd_start': d_start,
                            'dist': 0.0,
                            'desc': f'User Point ({row["case"]})'
                        })
                    elif l_type == 'U':
                        if d_start <= 0.01 and dist >= (spans[s_idx] - 0.01):
                            span_total_udl_N[s_idx] += mag_factored_N
                        else:
                            combined_loads_list.append({
                                'span_index': s_idx,
                                'type': 'U',
                                'mag': mag_factored_N, 
                                'd_start': d_start,
                                'dist': dist,
                                'desc': f'User Partial UDL ({row["case"]})'
                            })
                except Exception:
                    continue
        
        # 5.4 Add merged UDL
        for i in range(n_spans):
            if span_total_udl_N[i] > 0:
                combined_loads_list.append({
                    'span_index': i,
                    'type': 'U',
                    'mag': span_total_udl_N[i], 
                    'd_start': 0.0,
                    'dist': spans[i],
                    'desc': 'Total Combined UDL (Incl. SW)'
                })
        
        calc_loads_df = pd.DataFrame(combined_loads_list)

        # --- 6. BEAM SOLVER ---
        x_eval, M, V, D, R = solver.solve_beam(spans, sup_df, calc_loads_df, params)
        
        res_df = pd.DataFrame({
            'x': x_eval,
            'moment': M, 
            'shear': V,   
            'deflection': D * 1000 
        })
        
        # --- 7. TABS FOR RESULTS & REPORTING ---
        tab1, tab2 = st.tabs(["📊 1. Analysis Results & Checks", "📝 2. Interactive Design & Report"])
        
        with tab1:
            st.plotly_chart(design_view.plot_analysis_results(res_df, spans, sup_df, calc_loads_df, R), use_container_width=True)
            
            st.subheader("📌 Analysis Summary")
            v_max_kN = res_df['shear'].abs().max() / 1000.0
            m_max_pos_kNm = res_df['moment'].max() / 1000.0
            m_max_neg_kNm = res_df['moment'].min() / 1000.0
            d_abs_max_mm = res_df['deflection'].abs().max()
            
            c_res1, c_res2, c_res3, c_res4 = st.columns(4)
            c_res1.metric(f"Max Shear", f"{v_max_kN:.2f} kN")
            c_res2.metric("Max Moment (+)", f"{m_max_pos_kNm:.2f} kNm")
            c_res3.metric("Max Moment (-)", f"{abs(m_max_neg_kNm):.2f} kNm")
            c_res4.metric("Max Deflection", f"{d_abs_max_mm:.2f} mm")

            st.markdown("### 📍 Support Reactions")
            if R:
                reaction_data = [{"Node": int(str(k).replace('R', '')), "Reaction (kN)": v/1000.0} for k, v in R.items()]
                df_reac = pd.DataFrame(reaction_data).sort_values(by="Node")
                st.dataframe(df_reac.style.format({"Reaction (kN)": "{:.2f}"}), use_container_width=True, hide_index=True)

            with st.expander("✅ Equilibrium & Deflection Checks", expanded=True):
                ec1, ec2 = st.columns(2)
                with ec1:
                    st.markdown("**Static Equilibrium ($\Sigma F_y = 0$)**")
                    sum_R_kN = sum(R.values()) / 1000.0
                    total_applied_N = 0
                    for _, l in calc_loads_df.iterrows():
                        if l['type'] == 'P': total_applied_N += l['mag']
                        else: total_applied_N += (l['mag'] * l['dist'])
                    total_applied_kN = total_applied_N / 1000.0
                    
                    st.write(f"Total Reactions: **{sum_R_kN:.3f} kN**")
                    st.write(f"Total Applied Loads: **{total_applied_kN:.3f} kN**")
                    if abs(sum_R_kN - total_applied_kN) < 0.1:
                        st.success("Balance Check: PASS")
                    else:
                        st.error(f"Balance Check: FAIL (Diff: {abs(sum_R_kN - total_applied_kN):.4f} kN)")

        # ================= TAB 2: INTERACTIVE DESIGN & REPORT =================
        with tab2:
            st.header(f"⚙️ Interactive RC Design ({tag})")
            
            if is_service:
                st.warning("⚠️ Warning: Service Load Factors (1.0) are selected. Please switch to 'Ultimate' for proper strength design.")
            
            # --- Design Parameters ---
            b_mm = params['b'] * 1000
            h_mm = params['h'] * 1000
            fc = params['fc']
            fy = params['fy']
            
            final_design_res = []
            offsets = [0] + list(np.cumsum(spans))
            
            # String Accumulator สำหรับสร้าง Report ท้ายสุด
            full_cal_report = f"# 🏗️ RC Beam Design Calculation Report\n"
            full_cal_report += f"**Design Parameters:** fc' = {fc} MPa, fy = {fy} MPa, b = {b_mm} mm, h = {h_mm} mm\n"
            full_cal_report += f"**Load Factors:** DL={f_dl}, LL={f_ll}\n"
            full_cal_report += "---\n"

            # --- Loop Design for Each Span ---
            for i in range(n_spans):
                s_len = spans[i]
                s_start, s_end = offsets[i], offsets[i+1]
                
                # Get Forces for this span
                span_data = res_df[(res_df['x'] >= s_start - 1e-6) & (res_df['x'] <= s_end + 1e-6)]
                
                if not span_data.empty:
                    mu_pos = span_data['moment'].max() / 1000.0
                    mu_neg = abs(span_data['moment'].min()) / 1000.0
                    vu_max = span_data['shear'].abs().max() / 1000.0
                else:
                    mu_pos, mu_neg, vu_max = 0, 0, 0

                # Header for Span
                full_cal_report += f"\n## Span {i+1}: Length {s_len} m\n"
                full_cal_report += f"**Analysis Forces:** Mu(+) = {mu_pos:.2f} kNm, Mu(-) = {mu_neg:.2f} kNm, Vu = {vu_max:.2f} kN\n"

                with st.expander(f"📌 **Span {i+1}: Length {s_len} m** (Interactive Design)", expanded=True):
                    
                    # --- 0. Covering ---
                    c_cov1, c_cov2, _ = st.columns([2, 1, 2])
                    with c_cov1:
                         st.info(f"Forces: $M_u^+ = {mu_pos:.1f}$ kNm, $M_u^- = {mu_neg:.1f}$ kNm, $V_u = {vu_max:.1f}$ kN")
                    with c_cov2:
                        cover_mm = st.number_input(f"Covering (mm)", value=25.0, step=5.0, key=f"cov_{i}")

                    # ==========================================
                    # 1. POSITIVE MOMENT DESIGN (Bottom Steel)
                    # ==========================================
                    st.markdown("---")
                    st.markdown("### 1. Bottom Reinforcement (Mid-Span)")
                    
                    # 1.1 Calculate Req
                    d_eff_bot_est = h_mm - cover_mm - 9 - 10 
                    as_req_bot, rho_bot, err_bot = get_as_req(mu_pos, d_eff_bot_est, fc, fy, b_mm)
                    
                    if err_bot:
                        st.error("🚨 Section Size might be too small (Over-reinforced). Please increase Depth/Width.")
                        full_cal_report += "**Error: Section too small for Positive Moment.**\n"
                    
                    # 1.2 Interactive Selection (INPUTS HERE)
                    col_b_calc, col_b_sel = st.columns([1, 1])
                    with col_b_calc:
                        st.markdown(f"**Required Reinforcement:**")
                        st.markdown(f"- $M_u^+ = {mu_pos:.2f}$ kNm")
                        st.markdown(f"- $A_{{s,req}} \\approx \\mathbf{{{as_req_bot:.0f}}}$ $mm^2$")
                    
                    with col_b_sel:
                        st.markdown("**Select Rebar:**")
                        c_sel1, c_sel2 = st.columns(2)
                        with c_sel1:
                            bot_db = st.selectbox(f"DB (Bottom)", [12, 16, 20, 25, 28], index=1, key=f"bdb_{i}")
                        with c_sel2:
                            bot_n = st.number_input(f"Qty (Bottom)", min_value=2, value=2, step=1, key=f"bn_{i}")
                    
                    # 1.3 Verify & Show Report Line
                    d_eff_bot_real = h_mm - cover_mm - 9 - (bot_db / 2)
                    phi_Mn_bot, as_prov_bot, a_bot, Mn_bot_val, c_bot, st_bot = get_phi_Mn_details(bot_n, bot_db, d_eff_bot_real, b_mm, fc, fy)
                    status_b = "✅ OK" if phi_Mn_bot >= mu_pos else "❌ FAIL"
                    
                    # Show check result inline
                    st.markdown(f"**Verification:** Provide {bot_n}-DB{bot_db} ($A_s = {as_prov_bot:.0f} mm^2$)")
                    if status_b == "✅ OK":
                        st.success(f"Capacity $\\phi M_n = {phi_Mn_bot:.2f}$ kNm $\\ge M_u$ ({status_b})")
                    else:
                        st.error(f"Capacity $\\phi M_n = {phi_Mn_bot:.2f}$ kNm $< M_u$ ({status_b})")

                    # Append to Full Report
                    full_cal_report += f"\n**1. Positive Moment Design (Mid-Span)**\n"
                    full_cal_report += f"- Required As = {as_req_bot:.2f} mm² (based on estimated d)\n"
                    full_cal_report += f"- **Select:** {bot_n}-DB{bot_db} (As Provided = {as_prov_bot:.2f} mm²)\n"
                    full_cal_report += f"- **Check Capacity:**\n"
                    full_cal_report += f"  - d = {h_mm} - {cover_mm} - 9 - {bot_db/2} = {d_eff_bot_real:.2f} mm\n"
                    full_cal_report += f"  - a = ({as_prov_bot:.0f} * {fy}) / (0.85 * {fc} * {b_mm}) = {a_bot:.2f} mm\n"
                    full_cal_report += f"  - Mn = As * fy * (d - a/2) = {Mn_bot_val/1e6:.2f} kNm\n"
                    full_cal_report += f"  - **phi Mn = 0.9 * Mn = {phi_Mn_bot:.2f} kNm**\n"
                    full_cal_report += f"  - Conclusion: {phi_Mn_bot:.2f} >= {mu_pos:.2f} -> **{status_b}**\n"

                    # ==========================================
                    # 2. NEGATIVE MOMENT DESIGN (Top Steel)
                    # ==========================================
                    st.markdown("---")
                    st.markdown("### 2. Top Reinforcement (Supports)")
                    
                    # 2.1 Calculate Req
                    d_eff_top_est = h_mm - cover_mm - 9 - 10 
                    as_req_top, rho_top, err_top = get_as_req(mu_neg, d_eff_top_est, fc, fy, b_mm)
                    
                    # 2.2 Interactive Selection
                    col_t_calc, col_t_sel = st.columns([1, 1])
                    with col_t_calc:
                        st.markdown(f"**Required Reinforcement:**")
                        st.markdown(f"- $M_u^- = {mu_neg:.2f}$ kNm")
                        st.markdown(f"- $A_{{s,req}} \\approx \\mathbf{{{as_req_top:.0f}}}$ $mm^2$")
                        
                    with col_t_sel:
                        st.markdown("**Select Rebar:**")
                        c_sel3, c_sel4 = st.columns(2)
                        with c_sel3:
                            top_db = st.selectbox(f"DB (Top)", [12, 16, 20, 25, 28], index=1, key=f"tdb_{i}")
                        with c_sel4:
                            top_n = st.number_input(f"Qty (Top)", min_value=2, value=2, step=1, key=f"tn_{i}")

                    # 2.3 Verify & Show Report Line
                    d_eff_top_real = h_mm - cover_mm - 9 - (top_db / 2)
                    phi_Mn_top, as_prov_top, a_top, Mn_top_val, c_top, st_top = get_phi_Mn_details(top_n, top_db, d_eff_top_real, b_mm, fc, fy)
                    status_t = "✅ OK" if phi_Mn_top >= mu_neg else "❌ FAIL"
                    
                    st.markdown(f"**Verification:** Provide {top_n}-DB{top_db} ($A_s = {as_prov_top:.0f} mm^2$)")
                    if status_t == "✅ OK":
                        st.success(f"Capacity $\\phi M_n = {phi_Mn_top:.2f}$ kNm $\\ge M_u$ ({status_t})")
                    else:
                        st.error(f"Capacity $\\phi M_n = {phi_Mn_top:.2f}$ kNm $< M_u$ ({status_t})")

                    # Append to Full Report
                    full_cal_report += f"\n**2. Negative Moment Design (Support)**\n"
                    full_cal_report += f"- Required As = {as_req_top:.2f} mm²\n"
                    full_cal_report += f"- **Select:** {top_n}-DB{top_db} (As Provided = {as_prov_top:.2f} mm²)\n"
                    full_cal_report += f"- **Check Capacity:**\n"
                    full_cal_report += f"  - d = {d_eff_top_real:.2f} mm\n"
                    full_cal_report += f"  - a = {a_top:.2f} mm\n"
                    full_cal_report += f"  - **phi Mn = {phi_Mn_top:.2f} kNm**\n"
                    full_cal_report += f"  - Conclusion: {phi_Mn_top:.2f} >= {mu_neg:.2f} -> **{status_t}**\n"

                    # ==========================================
                    # 3. SHEAR DESIGN (Stirrups)
                    # ==========================================
                    st.markdown("---")
                    st.markdown("### 3. Shear Reinforcement (Stirrups)")
                    
                    # 3.1 Show Loads
                    st.markdown(f"**Design Force:** $V_u = {vu_max:.2f}$ kN")
                    
                    # 3.2 Interactive Selection
                    col_s1, col_s2, col_s3 = st.columns([1, 1, 2])
                    with col_s1:
                        stir_db = st.selectbox(f"Stirrup DB", [6, 9, 12], index=0, key=f"sdb_{i}")
                    with col_s2:
                        stir_s = st.number_input(f"Spacing (mm)", value=150, step=10, key=f"ss_{i}")
                    
                    # 3.3 Verify
                    d_shear = d_eff_bot_real 
                    status_v, phi_Vn, phi_Vc, phi_Vs, Vc_raw, Vs_raw = check_shear_details(vu_max, b_mm, d_shear, fc, fy, stir_db, stir_s)
                    
                    with col_s3:
                        st.markdown(f"**Check:** $\\phi V_c = {phi_Vc:.1f}$ kN, $\\phi V_s = {phi_Vs:.1f}$ kN")
                        if status_v == "OK":
                            st.success(f"Total $\\phi V_n = {phi_Vn:.1f}$ kN ({status_v})")
                        else:
                            st.error(f"Total $\\phi V_n = {phi_Vn:.1f}$ kN ({status_v})")

                    # Append to Full Report
                    full_cal_report += f"\n**3. Shear Design**\n"
                    full_cal_report += f"- Vu = {vu_max:.2f} kN\n"
                    full_cal_report += f"- **Select:** RB{stir_db} @ {stir_s} mm (2 legs)\n"
                    full_cal_report += f"- **Check Capacity:**\n"
                    full_cal_report += f"  - phi Vc = 0.85 * 0.17 * sqrt(fc) * b * d = {phi_Vc:.2f} kN\n"
                    full_cal_report += f"  - phi Vs = 0.85 * (Av * fy * d) / s = {phi_Vs:.2f} kN\n"
                    full_cal_report += f"  - **phi Vn = {phi_Vn:.2f} kN**\n"
                    full_cal_report += f"  - Conclusion: {phi_Vn:.2f} >= {vu_max:.2f} -> **{status_v}**\n"
                    full_cal_report += "--------------------------------------------------\n"

                    # Collect Data for Plotting
                    final_design_res.append({
                        'span': i+1,
                        'cover': cover_mm,
                        'top_db': top_db,
                        'bot_db': bot_db,
                        'stir_db': stir_db,
                        'pos': {'n': bot_n, 'area': as_prov_bot},
                        'neg': {'n': top_n, 'area': as_prov_top},
                        'shear': {'s': stir_s}
                    })

            # --- PLOTTING SECTION ---
            st.markdown("---")
            st.header("📊 Final Drawings")
            
            if st.button("Update / Refresh Drawings", type="primary"):
                # 1. LONGITUDINAL SECTION
                st.subheader("1. Longitudinal Section")
                try:
                    fig_long = section_plotter.plot_longitudinal_section_detailed(
                        spans, sup_df, final_design_res, params['h'], final_design_res[0]['cover']
                    )
                    st.pyplot(fig_long, use_container_width=True)
                except Exception as e:
                    st.error(f"Error plotting longitudinal: {e}")

                # 2. CROSS SECTIONS
                st.subheader("2. Cross Section Details")
                span_tabs = st.tabs([f"Span {i+1}" for i in range(n_spans)])
                
                for i, tab in enumerate(span_tabs):
                    with tab:
                        res = final_design_res[i]
                        c_det1, c_det2 = st.columns(2)
                        
                        with c_det1:
                            st.markdown(f"**Section A-A (Mid-span {i+1})**")
                            fig_a = section_plotter.plot_section(
                                params['b'], params['h'], res['cover'], 
                                res['top_db'], res['bot_db'], 
                                2, 
                                res['pos']['n'], 
                                f"RB{res['stir_db']}@{int(res['shear']['s'])}", 
                                params['fc'], params['fy'], f"SECTION A-A (Span {i+1})"
                            )
                            st.pyplot(fig_a, use_container_width=True)
                        
                        with c_det2:
                            st.markdown(f"**Section B-B (Support {i+1})**")
                            fig_b = section_plotter.plot_section(
                                params['b'], params['h'], res['cover'], 
                                res['top_db'], res['bot_db'], 
                                res['neg']['n'], 
                                2, 
                                f"RB{res['stir_db']}@{int(res['shear']['s'])}", 
                                params['fc'], params['fy'], f"SECTION B-B (Span {i+1})"
                            )
                            st.pyplot(fig_b, use_container_width=True)

            # --- DISPLAY CALCULATION REPORT ---
            st.markdown("---")
            st.header("📄 Detailed Calculation Report (รายการคำนวณแบบละเอียด)")
            st.info("รายการคำนวณด้านล่างนี้ อัพเดทตามสิ่งที่คุณเลือกด้านบนโดยอัตโนมัติ")
            with st.expander("คลิกเพื่อดูรายการคำนวณ (Click to expand)", expanded=False):
                st.code(full_cal_report, language='markdown')

    except Exception as e:
        st.error(f"❌ Analysis Error: {e}")
        st.exception(e)
