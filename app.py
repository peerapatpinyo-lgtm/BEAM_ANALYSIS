import streamlit as st
import pandas as pd
import numpy as np
import io

# --- 1. IMPORT CUSTOM MODULES ---
import input_handler
import solver
# import rc_design  <-- ไม่ต้องใช้แล้ว เพราะเราคำนวณสดในหน้าเว็บ
import design_view
import section_plotter

# --- HELPER FUNCTIONS: REAL-TIME CALCULATION ---
def get_as_req(Mu_kNm, d_eff_mm, fc, fy, b_mm):
    """คำนวณ As required โดยประมาณ"""
    if Mu_kNm == 0: return 0.0
    Mu = abs(Mu_kNm) * 1e6 # N-mm
    phi = 0.9
    
    # คำนวณ Rho
    m = fy / (0.85 * fc)
    Rn = Mu / (phi * b_mm * d_eff_mm**2)
    try:
        # Check if Rn is too high (Section too small)
        term = 1 - (2 * m * Rn) / fy
        if term < 0:
            rho = 0.0 # Fail / Complex number
        else:
            rho = (1/m) * (1 - np.sqrt(term))
    except:
        rho = 0.0 # Error case
    
    as_req = rho * b_mm * d_eff_mm
    
    # Min Reinforcement Check
    if as_req > 0: # Check min only if moment exists
        as_min1 = (0.25 * np.sqrt(fc) / fy) * b_mm * d_eff_mm
        as_min2 = (1.4 / fy) * b_mm * d_eff_mm
        as_min = max(as_min1, as_min2)
        return max(as_req, as_min)
    
    return 0.0

def get_phi_Mn(n, db, d_eff, b, fc, fy):
    """คำนวณ Capacity รับโมเมนต์จริง (phi Mn)"""
    Ast = n * (np.pi * (db/2)**2)
    if Ast == 0: return 0.0, 0.0
    
    a = (Ast * fy) / (0.85 * fc * b)
    Mn = Ast * fy * (d_eff - a/2)
    phi_Mn = 0.9 * Mn / 1e6 # kNm
    return phi_Mn, Ast

def check_shear(Vu_kN, b, d, fc, fy, stir_db, spacing):
    """คำนวณ Capacity แรงเฉือน (phi Vn)"""
    Vu = abs(Vu_kN) * 1000 # N
    Vc = 0.17 * np.sqrt(fc) * b * d
    phi = 0.85
    phi_Vc = phi * Vc
    
    Av = 2 * (np.pi * (stir_db/2)**2) # 2 legs
    if spacing <= 0: spacing = 1000 # prevent div by zero
    
    Vs = (Av * fy * d) / spacing
    phi_Vn = phi_Vc + (phi * Vs)
    
    status = "OK" if phi_Vn >= Vu else "FAIL"
    return status, phi_Vn/1000, phi_Vc/1000

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(page_title="Beam Analysis & Design Pro", layout="wide", page_icon="🏗️")
st.title("🏗️ RC Beam Analysis & Design Pro (Timoshenko)")

# --- 3. SIDEBAR INPUTS ---
# Get parameters and beam configuration from Sidebar via input_handler
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
        # 5.1 Self-Weight Calculation (Unit Weight = 24 kN/m³)
        w_sw_base_kN = params['b'] * params['h'] * 24.0     
        w_sw_factored_kN = w_sw_base_kN * f_dl
        
        # 5.2 Initialize Total UDL per span (Newton (N/m))
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
                        # Merge full-span UDL with Self-Weight to reduce matrix complexity
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
        
        # 5.4 Add merged UDL to main list
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
        
        # Convert results to DataFrame
        res_df = pd.DataFrame({
            'x': x_eval,
            'moment': M, 
            'shear': V,   
            'deflection': D * 1000 # m to mm
        })
        
        # --- 7. TABS FOR RESULTS & REPORTING ---
        tab1, tab2 = st.tabs(["📊 1. Analysis Results & Checks", "📝 2. Interactive Design"])
        
        with tab1:
            # 7.1 Plot Analysis Diagrams (BMD, SFD, Deflection)
            st.plotly_chart(design_view.plot_analysis_results(res_df, spans, sup_df, calc_loads_df, R), use_container_width=True)
            
            # 7.2 Summary Metrics
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

            # 7.3 Support Reactions Table
            st.markdown("### 📍 Support Reactions")
            if R:
                reaction_data = [{"Node": int(str(k).replace('R', '')), "Reaction (kN)": v/1000.0} for k, v in R.items()]
                df_reac = pd.DataFrame(reaction_data).sort_values(by="Node")
                st.dataframe(df_reac.style.format({"Reaction (kN)": "{:.2f}"}), use_container_width=True, hide_index=True)

            # 7.5 Static Equilibrium
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

        # ================= TAB 2: INTERACTIVE DESIGN =================
        with tab2:
            st.header(f"⚙️ Interactive RC Design ({tag})")
            
            if is_service:
                st.warning("⚠️ Warning: Service Load Factors (1.0) are selected. Please switch to 'Ultimate' for proper strength design.")
            
            # เตรียมตัวแปร
            b_mm = params['b'] * 1000
            h_mm = params['h'] * 1000
            fc = params['fc']
            fy = params['fy']
            
            final_design_res = []
            offsets = [0] + list(np.cumsum(spans))
            
            # Loop ทีละ Span เพื่อให้ User กรอกข้อมูล
            for i in range(n_spans):
                s_len = spans[i]
                s_start, s_end = offsets[i], offsets[i+1]
                
                # Filter forces
                span_data = res_df[(res_df['x'] >= s_start - 1e-6) & (res_df['x'] <= s_end + 1e-6)]
                
                if not span_data.empty:
                    mu_pos = span_data['moment'].max() / 1000.0
                    mu_neg = abs(span_data['moment'].min()) / 1000.0
                    vu_max = span_data['shear'].abs().max() / 1000.0
                else:
                    mu_pos, mu_neg, vu_max = 0, 0, 0

                # --- UI for Span i ---
                with st.expander(f"📌 **Span {i+1}: Length {s_len} m** (Design Input)", expanded=True):
                    
                    # 1. Covering Input
                    c_cov1, c_cov2, _ = st.columns([1, 1, 3])
                    with c_cov1:
                        st.markdown(f"**Forces:** Mu(+)={mu_pos:.1f}, Mu(-)={mu_neg:.1f}, Vu={vu_max:.1f}")
                    with c_cov2:
                        cover_mm = st.number_input(f"Covering (mm)", value=25.0, step=5.0, key=f"cov_{i}")
                    
                    st.markdown("---")

                    # 2. Bottom Steel (Positive Moment)
                    st.markdown("##### 1. Bottom Bars (Mid-span)")
                    # Approx d_eff for estimation
                    d_eff_bot_est = h_mm - cover_mm - 9 - 10 
                    as_req_bot = get_as_req(mu_pos, d_eff_bot_est, fc, fy, b_mm)
                    
                    cb1, cb2, cb3, cb4 = st.columns([2, 1.5, 1.5, 2])
                    with cb1:
                        st.info(f"Req: **{as_req_bot:.0f}** mm²")
                    with cb2:
                        bot_db = st.selectbox(f"DB", [12, 16, 20, 25, 28], index=1, key=f"bdb_{i}")
                    with cb3:
                        bot_n = st.number_input(f"Qty", min_value=2, value=2, step=1, key=f"bn_{i}")
                    with cb4:
                        # Update d_eff based on actual selection (assuming 9mm stirrup for cal)
                        d_eff_bot_real = h_mm - cover_mm - 9 - (bot_db / 2)
                        phi_Mn_bot, as_prov_bot = get_phi_Mn(bot_n, bot_db, d_eff_bot_real, b_mm, fc, fy)
                        
                        status_b = "✅ OK" if phi_Mn_bot >= mu_pos else "❌ FAIL"
                        if status_b == "✅ OK":
                            st.success(f"{status_b} (Cap={phi_Mn_bot:.1f} kNm)")
                        else:
                            st.error(f"{status_b} (Cap={phi_Mn_bot:.1f} kNm)")

                    # 3. Top Steel (Negative Moment)
                    st.markdown("##### 2. Top Bars (Supports)")
                    # Approx d_eff for estimation
                    d_eff_top_est = h_mm - cover_mm - 9 - 10 
                    as_req_top = get_as_req(mu_neg, d_eff_top_est, fc, fy, b_mm)
                    
                    ct1, ct2, ct3, ct4 = st.columns([2, 1.5, 1.5, 2])
                    with ct1:
                        st.info(f"Req: **{as_req_top:.0f}** mm²")
                    with ct2:
                        top_db = st.selectbox(f"DB", [12, 16, 20, 25, 28], index=1, key=f"tdb_{i}")
                    with ct3:
                        top_n = st.number_input(f"Qty", min_value=2, value=2, step=1, key=f"tn_{i}")
                    with ct4:
                        # Update d_eff based on actual selection
                        d_eff_top_real = h_mm - cover_mm - 9 - (top_db / 2)
                        phi_Mn_top, as_prov_top = get_phi_Mn(top_n, top_db, d_eff_top_real, b_mm, fc, fy)
                        
                        status_t = "✅ OK" if phi_Mn_top >= mu_neg else "❌ FAIL"
                        if status_t == "✅ OK":
                            st.success(f"{status_t} (Cap={phi_Mn_top:.1f} kNm)")
                        else:
                            st.error(f"{status_t} (Cap={phi_Mn_top:.1f} kNm)")

                    # 4. Shear (Stirrups)
                    st.markdown("##### 3. Stirrups (Shear)")
                    cs1, cs2, cs3, cs4 = st.columns([2, 1.5, 1.5, 2])
                    with cs1:
                        st.warning(f"Vu: **{vu_max:.1f}** kN")
                    with cs2:
                        stir_db = st.selectbox(f"RB/DB", [6, 9, 12], index=0, key=f"sdb_{i}")
                    with cs3:
                        stir_s = st.number_input(f"Spacing (mm)", value=150, step=10, key=f"ss_{i}")
                    with cs4:
                        d_shear = d_eff_bot_real # Use calculated effective depth
                        status_v, phi_Vn, phi_Vc = check_shear(vu_max, b_mm, d_shear, fc, fy, stir_db, stir_s)
                        if status_v == "OK":
                            st.success(f"✅ OK (Cap={phi_Vn:.1f})")
                        else:
                            st.error(f"❌ FAIL (Cap={phi_Vn:.1f})")

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
                        
                        # Section A-A: Mid Span (Show Bottom Bars dominant)
                        with c_det1:
                            st.markdown(f"**Section A-A (Mid-span {i+1})**")
                            fig_a = section_plotter.plot_section(
                                params['b'], params['h'], res['cover'], 
                                res['top_db'], res['bot_db'], 
                                2, # Top bars (Hanger)
                                res['pos']['n'], # Bottom bars (Actual)
                                f"RB{res['stir_db']}@{int(res['shear']['s'])}", 
                                params['fc'], params['fy'], f"SECTION A-A (Span {i+1})"
                            )
                            st.pyplot(fig_a, use_container_width=True)
                        
                        # Section B-B: Support (Show Top Bars dominant)
                        with c_det2:
                            st.markdown(f"**Section B-B (Support {i+1})**")
                            fig_b = section_plotter.plot_section(
                                params['b'], params['h'], res['cover'], 
                                res['top_db'], res['bot_db'], 
                                res['neg']['n'], # Top bars (Actual)
                                2, # Bottom bars (Hanger)
                                f"RB{res['stir_db']}@{int(res['shear']['s'])}", 
                                params['fc'], params['fy'], f"SECTION B-B (Span {i+1})"
                            )
                            st.pyplot(fig_b, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Analysis Error: {e}")
        st.exception(e)
