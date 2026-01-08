import streamlit as st
import pandas as pd
import numpy as np

# --- 1. IMPORT CUSTOM MODULES ---
import input_handler
import solver
import rc_design
import design_view
import section_plotter

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(page_title="Beam Analysis & Design Pro", layout="wide")
st.title("🏗️ RC Beam Analysis & Design Pro (Timoshenko)")

# --- 3. SIDEBAR INPUTS ---
# รับค่า Parameters และรูปพรรณคานจาก Sidebar
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

    # --- 5. LOAD CALCULATIONS & COMBINATIONS (FIXING POINT LOAD ERROR) ---
    try:
        # 5.1 Self-Weight Calculation (Unit Weight = 24 kN/m³)
        w_sw_base_kN = params['b'] * params['h'] * 24.0   
        w_sw_factored_kN = w_sw_base_kN * f_dl
        
        # 5.2 Initialize Total UDL per span (Newton (N/m))
        span_total_udl_N = {i: w_sw_factored_kN * 1000.0 for i in range(n_spans)} 
        combined_loads_list = []
        
        # 5.3 Process User-Defined Loads (Check Units Carefully)
        if not loads_df.empty:
            for _, row in loads_df.iterrows():
                try:
                    s_idx = int(row['span_index'])
                    if s_idx >= n_spans: continue 
                    
                    l_type = row['type']
                    # [FIX] ดึงค่า kN มาแล้วคูณ 1000 แค่ครั้งเดียวเพื่อเป็น Newton
                    mag_base_kN = float(row['mag']) 
                    mag_factored_N = mag_base_kN * f_ll * 1000.0 
                    dist = float(row['dist']) 
                    
                    # หากเป็น UDL เต็มช่วงคาน ให้ยุบรวมเข้ากับ SW เพื่อลดขนาด Matrix ใน Solver
                    if l_type == 'U' and dist >= (spans[s_idx] - 0.01):
                        span_total_udl_N[s_idx] += mag_factored_N
                    else:
                        combined_loads_list.append({
                            'span_index': s_idx,
                            'type': l_type,
                            'mag': mag_factored_N, 
                            'dist': dist,
                            'desc': 'User (Partial/Point)'
                        })
                except Exception:
                    continue
        
        # 5.4 รวม UDL ที่ยุบแล้วเข้าสู่ List หลัก
        for i in range(n_spans):
            if span_total_udl_N[i] > 0:
                combined_loads_list.append({
                    'span_index': i,
                    'type': 'U',
                    'mag': span_total_udl_N[i], 
                    'dist': spans[i],
                    'desc': 'Total Combined UDL (Incl. SW)'
                })
        
        calc_loads_df = pd.DataFrame(combined_loads_list)

        # --- 6. BEAM SOLVER ---
        # Solver จะรับค่าหน่วย SI (N, m) เสมอ
        x_eval, M, V, D, R = solver.solve_beam(spans, sup_df, calc_loads_df, params)
        
        # แปลงผลลัพธ์ใส่ DataFrame
        res_df = pd.DataFrame({
            'x': x_eval,
            'moment': M, 
            'shear': V,  
            'deflection': D * 1000 # m to mm
        })
        
        # --- 7. TABS FOR RESULTS & REPORTING ---
        tab1, tab2 = st.tabs(["📊 1. Analysis Results & Checks", "📝 2. RC Design & Report"])
        
        with tab1:
            # 7.1 Plot Analysis Diagrams
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

            # 7.3 Support Reactions
            st.markdown("### 📍 Support Reactions")
            if R:
                reaction_data = [{"Node": int(str(k).replace('R', '')), "Reaction (kN)": v/1000.0} for k, v in R.items()]
                df_reac = pd.DataFrame(reaction_data).sort_values(by="Node")
                st.dataframe(df_reac.style.format({"Reaction (kN)": "{:.2f}"}), use_container_width=True, hide_index=True)

            # 7.4 Detailed Load Reports
            with st.expander("🧮 Detailed Load Calculation Report", expanded=True):
                st.markdown("#### A. Self-Weight Calculation")
                sw_report = []
                for i in range(n_spans):
                    sw_report.append({
                        "Span": i+1,
                        "Dimensions": f"{params['b']}m x {params['h']}m",
                        "Formula": f"b*h * 24 kN/m³ * {f_dl}",
                        "Factored Result": f"{w_sw_factored_kN:.2f} kN/m"
                    })
                st.table(pd.DataFrame(sw_report))

                st.markdown("#### B. Load Combination Breakdown (Internal Solver Units)")
                st.dataframe(calc_loads_df[['span_index', 'type', 'mag', 'dist', 'desc']], use_container_width=True)

            # 7.5 Static Equilibrium & Serviceability Checks
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
                with ec2:
                    st.markdown("**Deflection Limit Check**")
                    limit_mm = (max(spans) * 1000) / 240.0
                    st.write(f"Max Deflection: {d_abs_max_mm:.2f} mm")
                    st.write(f"Allowable (L/240): {limit_mm:.2f} mm")
                    if d_abs_max_mm <= limit_mm: st.success("Deflection: PASS")
                    else: st.error("Deflection: FAIL")

        # ================= TAB 2: RC DESIGN =================
        with tab2:
            if is_service:
                st.warning("⚠️ Warning: Strength Design requires 'Ultimate Load' factors.")
            
            st.header(f"Reinforced Concrete Design ({tag})")
            design_res = []
            db_main = 16 
            offsets = [0] + list(np.cumsum(spans))

            for i in range(n_spans):
                s_start, s_end = offsets[i], offsets[i+1]
                span_data = res_df[(res_df['x'] >= s_start - 1e-6) & (res_df['x'] <= s_end + 1e-6)]
                
                if not span_data.empty:
                    mu_pos = span_data['moment'].max() / 1000.0
                    mu_neg = abs(span_data['moment'].min()) / 1000.0
                    vu_max = span_data['shear'].abs().max() / 1000.0
                    d_eff = params['h'] - 0.05
                    
                    As_pos, _, _, steps_pos = rc_design.design_beam_flexure(mu_pos, params['b'], d_eff, params['fc'], params['fy'])
                    As_neg, _, _, steps_neg = rc_design.design_beam_flexure(mu_neg, params['b'], d_eff, params['fc'], params['fy'])
                    s_req, _, steps_shear = rc_design.check_shear(vu_max, params['b'], d_eff, params['fc'], params['fy'])
                    
                    def n_bars(area): 
                        return max(2, int(np.ceil(area / (np.pi * (db_main / 2)**2))))
                    
                    design_res.append({
                        'span': i+1, 'db': db_main,
                        'pos': {'n': n_bars(As_pos)}, 
                        'neg': {'n': n_bars(As_neg)}, 
                        'shear': {'s': s_req}
                    })
                    
                    with st.expander(f"📘 Detailed Design Calculation: Span {i+1}", expanded=False):
                        st.markdown(f"**Flexural Design (Mu+ = {mu_pos:.2f}, Mu- = {mu_neg:.2f} kNm)**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Bottom Steel:")
                            for s in steps_pos: st.latex(s)
                        with col2:
                            st.write("Top Steel:")
                            for s in steps_neg: st.latex(s)
                        st.markdown("**Shear Design**")
                        for s in steps_shear: st.latex(s)

            st.markdown("---")
            st.subheader("🛠️ Detailing Preview")
            if design_res:
                col_det1, col_det2 = st.columns([1, 2])
                with col_det1:
                    fig_sec = section_plotter.plot_section(params['b'], params['h'], 40, db_main, design_res[0]['neg']['n'], design_res[0]['pos']['n'], "RB6", params['fc'], params['fy'])
                    st.pyplot(fig_sec)
                with col_det2:
                    fig_long = section_plotter.plot_longitudinal_section_detailed(spans, sup_df, design_res, params['h'], 40)
                    st.pyplot(fig_long)

    except Exception as e:
        st.error(f"❌ Calculation Error: {e}")
        st.exception(e)

# --- END OF SCRIPT (250+ Lines maintained for logic completeness) ---
