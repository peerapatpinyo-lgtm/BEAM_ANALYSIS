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
st.set_page_config(page_title="Beam Analysis & Design Pro", layout="wide", page_icon="🏗️")

st.title("🏗️ RC Beam Analysis & Design Pro (Timoshenko)")

# --- 3. SIDEBAR INPUTS ---
# Calling your existing input_handler logic
try:
    params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()
except Exception as e:
    st.error(f"Error in input_handler: {e}")
    st.stop()

if not stable:
    st.error("🚨 **Structure is unstable!** Please check supports. A stable beam requires at least 3 reaction components.")
else:
    # --- 4. LOAD FACTORS & ANALYSIS SETTINGS ---
    st.markdown("### ⚙️ Analysis Settings & Load Factors")
    
    mode_select = st.radio(
        "Select Analysis Mode:",
        ["Service Load (1.0DL + 1.0LL)", "Ultimate / Custom Factors"],
        horizontal=True
    )
    
    col_f1, col_f2, _ = st.columns([1, 1, 2])
    
    if mode_select.startswith("Service"):
        f_dl, f_ll = 1.0, 1.0
        with col_f1: st.number_input("Dead Load Factor (DL)", value=1.0, disabled=True, key="d1")
        with col_f2: st.number_input("Live Load Factor (LL)", value=1.0, disabled=True, key="d2")
        tag = "Service"
        is_service = True
    else:
        with col_f1: f_dl = st.number_input("Dead Load Factor (DL)", value=1.4, step=0.1, format="%.2f", key="u1")
        with col_f2: f_ll = st.number_input("Live Load Factor (LL)", value=1.7, step=0.1, format="%.2f", key="u2")
        tag = "Ultimate"
        is_service = False

    # --- 5. LOAD CALCULATIONS & COMBINATIONS ---
    try:
        # 5.1 Self-Weight (Concrete = 24 kN/m³)
        # Formula: Area (m2) * 24 kN/m3
        w_sw_base_kN = params['b'] * params['h'] * 24.0  
        w_sw_factored_kN = w_sw_base_kN * f_dl
        
        # 5.2 Initialize Load List for Solver (Convert kN/m to N/m)
        span_total_udl_N = {i: w_sw_factored_kN * 1000.0 for i in range(n_spans)}
        point_and_partial_loads = []
        
        # 5.3 Process User Loads from Dataframe
        if loads_df is not None and not loads_df.empty:
            for _, row in loads_df.iterrows():
                s_idx = int(row['span_index'])
                if s_idx < n_spans:
                    mag_factored = row['mag'] * f_ll # Apply Live Load Factor
                    
                    # If it's a Full Uniform Load, add to the span bucket
                    if row['type'] == 'U' and row['dist'] >= (spans[s_idx] - 0.01):
                        span_total_udl_N[s_idx] += mag_factored
                    else:
                        point_and_partial_loads.append({
                            'span_index': s_idx, 'type': row['type'],
                            'mag': mag_factored, 'dist': row['dist'], 'desc': 'User Load'
                        })
        
        # Final combined list for the Solver
        final_calc_list = []
        for i in range(n_spans):
            final_calc_list.append({
                'span_index': i, 'type': 'U', 'mag': span_total_udl_N[i],
                'dist': spans[i], 'desc': 'Total UDL (SW + User)'
            })
        final_calc_list.extend(point_and_partial_loads)
        calc_loads_df = pd.DataFrame(final_calc_list)

        # --- 6. SOLVER EXECUTION ---
        x_eval, M, V, D, R = solver.solve_beam(spans, sup_df, calc_loads_df, params)
        
        res_df = pd.DataFrame({
            'x': x_eval, 'moment': M, 'shear': V, 'deflection': D * 1000 # to mm
        })

        # --- 7. TABS FOR RESULTS & DESIGN ---
        tab1, tab2 = st.tabs(["📊 1. Analysis Results", "📝 2. RC Design Report"])

        with tab1:
            # Graphical Output
            st.plotly_chart(design_view.plot_analysis_results(res_df, spans, sup_df, calc_loads_df, R), use_container_width=True)
            
            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"Max Shear ({tag})", f"{res_df['shear'].abs().max()/1000:.2f} kN")
            m2.metric("Max Moment (+)", f"{res_df['moment'].max()/1000:.2f} kNm")
            m3.metric("Max Moment (-)", f"{abs(res_df['moment'].min())/1000:.2f} kNm")
            m4.metric("Max Deflection", f"{res_df['deflection'].abs().max():.2f} mm")

            # --- DETAILED LOAD COMBINATION REPORT ---
            st.markdown("---")
            with st.expander("🧮 Detailed Load Calculation & Combinations", expanded=True):
                st.markdown("#### A. Self-Weight Calculation (DL)")
                st.latex(r"w_{sw} = b \times h \times 24.0 = " + f"{params['b']} \times {params['h']} \times 24 = {w_sw_base_kN:.2f} \, \text{kN/m}")
                
                st.markdown("#### B. Load Combination Breakdown (Total Design Load)")
                report_rows = []
                for i in range(n_spans):
                    # Add Self-Weight Row
                    report_rows.append({
                        "Span": i+1, "Source": "Self-Weight (DL)", "Base Value": f"{w_sw_base_kN:.2f} kN/m",
                        "Factor": f"x{f_dl:.2f}", "Factored Load": f"{w_sw_factored_kN:.2f} kN/m"
                    })
                    # Add User Load Rows (if any)
                    if loads_df is not None and not loads_df.empty:
                        s_user = loads_df[loads_df['span_index'] == i]
                        for _, r in s_user.iterrows():
                            u_base = r['mag']/1000.0
                            unit = "kN" if r['type'] == 'P' else "kN/m"
                            report_rows.append({
                                "Span": i+1, "Source": "User Load (LL)", "Base Value": f"{u_base:.2f} {unit}",
                                "Factor": f"x{f_ll:.2f}", "Factored Load": f"{u_base*f_ll:.2f} {unit}"
                            })
                st.table(pd.DataFrame(report_rows))

            # Equilibrium Check
            with st.expander("⚖️ Static Equilibrium Check"):
                sum_R = sum(R.values()) / 1000.0
                total_applied = (w_sw_factored_kN * sum(spans)) 
                if point_and_partial_loads:
                    for l in point_and_partial_loads:
                        val = l['mag']/1000.0
                        total_applied += val if l['type'] == 'P' else (val * l['dist'])
                
                st.write(f"Total Reactions (Up): **{sum_R:.2f} kN**")
                st.write(f"Total Applied (Down): **{total_applied:.2f} kN**")
                if abs(sum_R - total_applied) < 0.5: st.success("Equilibrium: OK")

        with tab2:
            st.header(f"Reinforced Concrete Design ({tag})")
            if is_service: st.warning("Please switch to 'Ultimate Factors' for safety design.")
            
            design_res = []
            current_x = 0
            for i, L in enumerate(spans):
                # Filter results for this span
                span_data = res_df[(res_df['x'] >= current_x) & (res_df['x'] <= current_x + L)]
                if not span_data.empty:
                    mu_pos = span_data['moment'].max()/1000
                    mu_neg = abs(span_data['moment'].min())/1000
                    vu_max = span_data['shear'].abs().max()/1000
                    d_eff = params['h'] - 0.05
                    
                    as_p, _, _, steps_p = rc_design.design_beam_flexure(mu_pos, params['b'], d_eff, params['fc'], params['fy'])
                    as_n, _, _, steps_n = rc_design.design_beam_flexure(mu_neg, params['b'], d_eff, params['fc'], params['fy'])
                    s_v, _, steps_v = rc_design.check_shear(vu_max, params['b'], d_eff, params['fc'], params['fy'])
                    
                    def n_bars(As): return max(2, int(np.ceil(As / (np.pi * 0.008**2))))
                    
                    design_res.append({'span': i+1, 'pos': {'n': n_bars(as_p)}, 'neg': {'n': n_bars(as_n)}, 'shear': {'s': s_v}})
                    
                    with st.expander(f"📘 Span {i+1} Calculation Details"):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.write("**Positive Flexure**")
                            for s in steps_p: st.latex(s)
                        with c2:
                            st.write("**Negative Flexure**")
                            for s in steps_n: st.latex(s)
                        st.write("**Shear Design**")
                        for s in steps_v: st.latex(s)
                current_x += L

            st.markdown("---")
            st.subheader("🛠️ Detailing Preview")
            if design_res:
                col_d1, col_d2 = st.columns([1, 2])
                with col_d1:
                    st.pyplot(section_plotter.plot_section(params['b'], params['h'], 40, 16, design_res[0]['neg']['n'], design_res[0]['pos']['n'], "Stirrup", params['fc'], params['fy']))
                with col_d2:
                    st.pyplot(section_plotter.plot_longitudinal_section_detailed(spans, sup_df, design_res, params['h'], 40))

    except Exception as e:
        st.error(f"Analysis Crash: {e}")
        st.info("Check if all input files (.py) are present and the Sidebar inputs are valid.")
