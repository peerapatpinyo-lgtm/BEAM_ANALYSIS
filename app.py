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
        
        # 5.2 Lists for Solver (Newton) and Graphics (kN)
        solver_loads = []
        plot_loads = []
        
        # Add Self-Weight to Solver and Plot
        for i in range(n_spans):
            # For Solver (N/m)
            solver_loads.append({
                'span_index': i, 'type': 'U', 
                'mag': w_sw_factored_kN * 1000.0, 
                'dist': spans[i]
            })
            # For Plotting (kN/m) - This fixes the 6600 error
            plot_loads.append({
                'span_index': i, 'type': 'U', 
                'mag': w_sw_factored_kN, 
                'dist': spans[i], 'desc': 'Self-Weight'
            })
        
        # 5.3 Process User-Defined Loads
        if not loads_df.empty:
            for _, row in loads_df.iterrows():
                try:
                    s_idx = int(row['span_index'])
                    if s_idx >= n_spans: continue 
                    
                    l_type = row['type']
                    mag_base_kN = float(row['mag'])
                    mag_factored_kN = mag_base_kN * f_ll
                    dist = float(row['dist']) 
                    
                    # Add to Solver (N or N/m)
                    solver_loads.append({
                        'span_index': s_idx, 'type': l_type,
                        'mag': mag_factored_kN * 1000.0, 'dist': dist
                    })
                    # Add to Plot (kN or kN/m) - This ensures label is 6.6
                    plot_loads.append({
                        'span_index': s_idx, 'type': l_type,
                        'mag': mag_factored_kN, 'dist': dist, 'desc': 'User'
                    })
                except Exception: continue
        
        calc_loads_df = pd.DataFrame(solver_loads)
        plot_loads_df = pd.DataFrame(plot_loads)

        # --- 6. BEAM SOLVER ---
        x_eval, M, V, D, R = solver.solve_beam(spans, sup_df, calc_loads_df, params)
        
        # Convert all solver output (N, N-m) to display units (kN, kN-m)
        res_df = pd.DataFrame({
            'x': x_eval,
            'moment': M / 1000.0,   # kN-m
            'shear': V / 1000.0,    # kN
            'deflection': D * 1000.0 # mm
        })
        
        # Reactions in kN
        R_kN = {k: v / 1000.0 for k, v in R.items()}
        
        # --- 7. DISPLAY RESULTS ---
        tab1, tab2 = st.tabs(["📊 1. Analysis Results & Checks", "📝 2. RC Design & Report"])
        
        with tab1:
            # ใช้ plot_loads_df ที่เป็นหน่วย kN ทันทีเพื่อให้ Label ในกราฟถูกต้อง
            st.plotly_chart(design_view.plot_analysis_results(res_df, spans, sup_df, plot_loads_df, R_kN), use_container_width=True)
            
            st.subheader("📌 Analysis Summary")
            v_max = res_df['shear'].abs().max()
            m_max_pos = res_df['moment'].max()
            m_max_neg = res_df['moment'].min()
            d_abs_max = res_df['deflection'].abs().max()
            
            c_res1, c_res2, c_res3, c_res4 = st.columns(4)
            c_res1.metric(f"Max Shear ({tag})", f"{v_max:.2f} kN")
            c_res2.metric("Max Moment (+)", f"{m_max_pos:.2f} kNm")
            c_res3.metric("Max Moment (-)", f"{abs(m_max_neg):.2f} kNm")
            c_res4.metric("Max Deflection", f"{d_abs_max:.2f} mm")

            st.markdown("### 📍 Support Reactions")
            if R_kN:
                reaction_data = [{"Node": int(str(k).replace('R', '')), "Reaction (kN)": v} for k, v in R_kN.items()]
                df_reac = pd.DataFrame(reaction_data).sort_values(by="Node")
                st.dataframe(df_reac.style.format({"Reaction (kN)": "{:.2f}"}), use_container_width=True, hide_index=True)

            st.markdown("---")

            with st.expander("🧮 Detailed Load Calculation Report", expanded=True):
                st.markdown("#### A. Self-Weight Calculation")
                sw_table = [{"Span": i+1, "b (m)": params['b'], "h (m)": params['h'], "Factored (kN/m)": f"{w_sw_factored_kN:.2f}"} for i in range(n_spans)]
                st.table(pd.DataFrame(sw_table))

                st.markdown("#### B. User Load Breakdown")
                if not loads_df.empty:
                    st.table(loads_df)
                else:
                    st.write("No additional loads applied.")

            with st.expander("✅ Equilibrium & Deflection Checks", expanded=True):
                ec1, ec2 = st.columns(2)
                with ec1:
                    st.markdown("**Static Equilibrium**")
                    sum_R = sum(R_kN.values())
                    total_app = w_sw_factored_kN * sum(spans)
                    if not loads_df.empty:
                        for _, r in loads_df.iterrows():
                            # P is kN, U is kN/m * length
                            total_app += (float(r['mag']) * f_ll) * (1.0 if r['type'] == 'P' else (float(r['dist']) if r['type'] == 'U' else 1.0))
                    st.write(f"Total Reactions: **{sum_R:.2f} kN**")
                    st.write(f"Total Applied: **{total_app:.2f} kN**")
                    if abs(sum_R - total_app) < 0.5: st.success("Balance: PASS")
                
                with ec2:
                    st.markdown("**Deflection Check**")
                    limit = (max(spans) * 1000) / 240.0
                    st.write(f"Max: {d_abs_max:.2f} mm | Limit: {limit:.2f} mm")
                    if d_abs_max <= limit: st.success("Deflection: PASS")

        # ================= TAB 2: RC DESIGN =================
        with tab2:
            if is_service: st.warning("⚠️ Switch to Ultimate mode for Design.")
            st.header(f"Reinforced Concrete Design ({tag})")
            
            design_res = []
            span_start = 0
            for i, span_len in enumerate(spans):
                span_end = span_start + span_len
                # Filter results for this span
                span_data = res_df[(res_df['x'] >= span_start - 1e-6) & (res_df['x'] <= span_end + 1e-6)]
                
                if not span_data.empty:
                    mu_p = span_data['moment'].max()
                    mu_n = abs(span_data['moment'].min())
                    vu_max = span_data['shear'].abs().max()
                    d_eff = params['h'] - 0.05
                    
                    # Call RC Design Module
                    As_p, _, _, stp_p = rc_design.design_beam_flexure(mu_p, params['b'], d_eff, params['fc'], params['fy'])
                    As_n, _, _, stp_n = rc_design.design_beam_flexure(mu_n, params['b'], d_eff, params['fc'], params['fy'])
                    s_r, _, stp_s = rc_design.check_shear(vu_max, params['b'], d_eff, params['fc'], params['fy'])
                    
                    n_p = max(2, int(np.ceil(As_p / (np.pi * (0.016/2)**2))))
                    n_n = max(2, int(np.ceil(As_n / (np.pi * (0.016/2)**2))))
                    
                    design_res.append({'span': i+1, 'pos': {'n': n_p}, 'neg': {'n': n_n}, 'shear': {'s': s_r}})
                    
                    with st.expander(f"📘 Span {i+1} Design Details"):
                        st.write(f"Moment: +{mu_p:.2f} / -{mu_n:.2f} kNm")
                        st.latex(stp_p[0] if stp_p else "")
                        st.write(f"Required Shear Reinforcement spacing: {s_r*1000:.0f} mm")

                span_start += span_len

            st.markdown("---")
            if design_res:
                st.subheader("🛠️ Detailing Preview")
                c_det1, c_det2 = st.columns([1, 2])
                with c_det1:
                    st.pyplot(section_plotter.plot_section(params['b'], params['h'], 40, 16, design_res[0]['neg']['n'], design_res[0]['pos']['n'], f"RB6@{design_res[0]['shear']['s']*1000:.0f}", params['fc'], params['fy']))
                with c_det2:
                    st.pyplot(section_plotter.plot_longitudinal_section_detailed(spans, sup_df, design_res, params['h'], 40))

    except Exception as e:
        st.error(f"❌ Calculation Error: {e}")
