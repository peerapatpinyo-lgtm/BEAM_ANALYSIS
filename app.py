import streamlit as st
import pandas as pd
import numpy as np

# --- 1. Import custom modules ---
# ตรวจสอบให้แน่ใจว่าไฟล์เหล่านี้อยู่ในโฟลเดอร์เดียวกัน
import input_handler
import solver
import rc_design
import design_view
import section_plotter

# --- 2. Page Config ---
st.set_page_config(page_title="RC Beam Pro", layout="wide", page_icon="🏗️")

st.title("🏗️ RC Beam Analysis & Design Pro (Timoshenko)")

# --- 3. Sidebar Inputs ---
try:
    params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()
except Exception as e:
    st.error(f"Error in Sidebar Inputs: {e}")
    st.stop()

if not stable:
    st.error("🚨 **Structure is unstable!** Please check supports (Min. 3 reaction components required).")
else:
    # --- 4. Analysis Settings & Load Factors ---
    st.markdown("### ⚙️ Analysis Settings")
    mode_select = st.radio("Select Analysis Mode:", ["Service Load (1.0DL + 1.0LL)", "Ultimate / Custom Factors"], horizontal=True)
    
    col_fac1, col_fac2 = st.columns(2)
    if mode_select.startswith("Service"):
        f_dl, f_ll, tag, is_service = 1.0, 1.0, "Service", True
        col_fac1.number_input("DL Factor", value=1.0, disabled=True)
        col_fac2.number_input("LL Factor", value=1.0, disabled=True)
    else:
        f_dl = col_fac1.number_input("Dead Load Factor (DL)", value=1.4, step=0.1, key="fdl_u")
        f_ll = col_fac2.number_input("Live Load Factor (LL)", value=1.7, step=0.1, key="fll_u")
        tag, is_service = "Ultimate", False

    # --- 5. Load Processing ---
    try:
        # Self-Weight calculation
        w_sw_kN_m = params['b'] * params['h'] * 24.0 * f_dl
        
        combined_loads = []
        # Add Self-weight for each span
        for i in range(n_spans):
            combined_loads.append({
                'span_index': i, 'type': 'U', 'mag': w_sw_kN_m * 1000.0, 
                'dist': spans[i], 'desc': 'Self-Weight'
            })
        
        # Add User Loads
        if not loads_df.empty:
            for _, row in loads_df.iterrows():
                if row['span_index'] < n_spans:
                    combined_loads.append({
                        'span_index': int(row['span_index']),
                        'type': row['type'],
                        'mag': row['mag'] * f_ll,
                        'dist': row['dist'],
                        'desc': 'User Load'
                    })
        
        calc_loads_df = pd.DataFrame(combined_loads)

        # --- 6. Solver ---
        x_eval, M, V, D, R = solver.solve_beam(spans, sup_df, calc_loads_df, params)
        
        res_df = pd.DataFrame({
            'x': x_eval, 'moment': M, 'shear': V, 'deflection': D * 1000
        })

        # --- 7. Tabs Layout ---
        tab1, tab2 = st.tabs(["📊 Analysis Results", "📝 RC Design"])

        with tab1:
            st.plotly_chart(design_view.plot_analysis_results(res_df, spans, sup_df, calc_loads_df, R), use_container_width=True)
            
            # Metrics
            v_max = res_df['shear'].abs().max()/1000
            m_max_pos = res_df['moment'].max()/1000
            m_max_neg = res_df['moment'].min()/1000
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Max Shear", f"{v_max:.2f} kN")
            m2.metric("Max Moment (+)", f"{m_max_pos:.2f} kNm")
            m3.metric("Max Moment (-)", f"{abs(m_max_neg):.2f} kNm")

            # Equilibrium Check
            with st.expander("⚖️ Equilibrium Check", expanded=True):
                sum_R = sum(R.values()) / 1000.0
                # Simple sum of loads (Approximation for display)
                total_w = (calc_loads_df[calc_loads_df['type']=='U']['mag']/1000 * calc_loads_df[calc_loads_df['type']=='U']['dist']).sum()
                total_p = calc_loads_df[calc_loads_df['type']=='P']['mag'].sum() / 1000
                sum_L = total_w + total_p
                
                st.write(f"Total Reactions: {sum_R:.2f} kN | Total Loads: {sum_L:.2f} kN")
                if abs(sum_R - sum_L) < 1.0: st.success("Equilibrium Pass")
                else: st.warning("Check load definitions")

        with tab2:
            st.header(f"Reinforced Concrete Design ({tag})")
            design_res = []
            span_start = 0
            
            for i, s_len in enumerate(spans):
                span_end = span_start + s_len
                span_data = res_df[(res_df['x'] >= span_start) & (res_df['x'] <= span_end)]
                
                if not span_data.empty:
                    mu_p, mu_n = span_data['moment'].max()/1000, abs(span_data['moment'].min())/1000
                    vu = span_data['shear'].abs().max()/1000
                    d_eff = params['h'] - 0.05
                    
                    # Design logic from your module
                    as_p, _, _, steps_p = rc_design.design_beam_flexure(mu_p, params['b'], d_eff, params['fc'], params['fy'])
                    as_n, _, _, steps_n = rc_design.design_beam_flexure(mu_n, params['b'], d_eff, params['fc'], params['fy'])
                    s_shear, _, steps_v = rc_design.check_shear(vu, params['b'], d_eff, params['fc'], params['fy'])
                    
                    # Store results for plotting
                    n_p = max(2, int(np.ceil(as_p / (np.pi * 0.008**2))))
                    n_n = max(2, int(np.ceil(as_n / (np.pi * 0.008**2))))
                    design_res.append({'span': i+1, 'pos': {'n': n_p}, 'neg': {'n': n_n}, 'shear': {'s': s_shear}})
                    
                    with st.expander(f"Span {i+1} Details"):
                        st.write(f"As Req (Bot): {as_p*1e6:.0f} mm² | As Req (Top): {as_n*1e6:.0f} mm²")

                span_start += s_len
            
            # Drawing
            if design_res:
                st.subheader("Detailing Preview")
                fig_sec = section_plotter.plot_section(params['b'], params['h'], 40, 16, design_res[0]['neg']['n'], design_res[0]['pos']['n'], "Stirrup", params['fc'], params['fy'])
                st.pyplot(fig_sec)

    except Exception as e:
        st.error(f"Analysis Error: {e}")
