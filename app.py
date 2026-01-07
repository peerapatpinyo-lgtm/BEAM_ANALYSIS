import streamlit as st
import pandas as pd
import numpy as np

# Import custom modules
import input_handler
import solver
import rc_design
import design_view
import section_plotter

st.set_page_config(page_title="Beam Analysis & Design", layout="wide")
st.title("🏗️ RC Beam Analysis & Design Pro (Timoshenko)")

# --- Sidebar Inputs ---
params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()

if not stable:
    st.error("🚨 Structure is unstable! Please check supports (Must have at least 3 reaction components).")
else:
    # --- 1. Prepare & Combine Loads ---
    # 1.1 Calculate Self-Weight
    # Define BOTH units to prevent NameError in display later
    w_sw_kN = params['b'] * params['h'] * 24.0   # kN/m (for Display)
    w_sw_N_m = w_sw_kN * 1000.0                  # N/m  (for Calculation)
    
    # 1.2 Initialize bucket for Total UDL per span
    # Key = span_index, Value = Total Magnitude (N/m)
    span_total_udl = {i: w_sw_N_m for i in range(n_spans)}
    
    combined_loads_list = []
    
    # 1.3 Process User Loads
    if not loads_df.empty:
        for _, row in loads_df.iterrows():
            try:
                s_idx = int(row['span_index'])
                # Safe check: if span index exceeds current spans (e.g. after deleting a span)
                if s_idx >= n_spans: continue 
                
                l_type = row['type']
                mag = row['mag']   # N or N/m
                dist = row['dist'] # m
                
                current_span_len = spans[s_idx]
                
                # CHECK: If UDL covers FULL span -> MERGE
                if l_type == 'U' and dist >= (current_span_len - 0.01):
                    span_total_udl[s_idx] += mag
                else:
                    # Point Load or Partial UDL -> KEEP SEPARATE
                    combined_loads_list.append({
                        'span_index': s_idx,
                        'type': l_type,
                        'mag': mag,
                        'dist': dist,
                        'desc': 'Point/Partial Load'
                    })
            except Exception as e:
                st.warning(f"Skipping invalid load row: {e}")
    
    # 1.4 Add Merged UDLs to list
    for i in range(n_spans):
        total_mag = span_total_udl[i]
        if total_mag > 0:
            combined_loads_list.append({
                'span_index': i,
                'type': 'U',
                'mag': total_mag,
                'dist': spans[i], # Full span
                'desc': 'Total Combined UDL (SW + User)'
            })
            
    calc_loads_df = pd.DataFrame(combined_loads_list)

    # --- 2. Solve Beam ---
    x_eval, M, V, D, R = solver.solve_beam(spans, sup_df, calc_loads_df, params)
    
    res_df = pd.DataFrame({
        'x': x_eval,
        'moment': M,
        'shear': V,
        'deflection': D * 1000 # Convert m to mm
    })
    
    # --- 3. DISPLAY RESULTS (TABS) ---
    tab1, tab2 = st.tabs(["📊 1. Analysis Results & Checks", "📝 2. RC Design & Report"])
    
    # ================= TAB 1: DIAGRAMS & CHECKS =================
    with tab1:
        # --- PART 1: PLOT DIAGRAMS ---
        st.info(f"ℹ️ **Note:** 'Uniform Loads' in the graph now combine **Self-Weight** + **User UDL**.")
        fig = design_view.plot_analysis_results(res_df, spans, sup_df, calc_loads_df, R)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # --- PART 2: ANALYSIS SUMMARY ---
        st.subheader("📌 Analysis Summary (Max/Min Values)")
        
        v_max_pos = res_df['shear'].max()/1000
        v_max_neg = res_df['shear'].min()/1000
        m_max_pos = res_df['moment'].max()/1000
        m_max_neg = res_df['moment'].min()/1000
        d_abs_max = res_df['deflection'].abs().max()
        
        col_sum1, col_sum2, col_sum3 = st.columns(3)
        col_sum1.metric("Max Shear (V+)", f"{v_max_pos:.2f} kN")
        col_sum1.metric("Min Shear (V-)", f"{v_max_neg:.2f} kN")
        col_sum2.metric("Max Moment (+)", f"{m_max_pos:.2f} kNm")
        col_sum2.metric("Max Moment (-)", f"{m_max_neg:.2f} kNm")
        col_sum3.metric("Max Deflection", f"{d_abs_max:.2f} mm")
        
        # --- PART 3: ENGINEERING CHECKS ---
        with st.expander("✅ Engineering Checks (Equilibrium & Deflection)", expanded=True):
            ec1, ec2 = st.columns(2)
            
            # 1. Equilibrium Check
            with ec1:
                st.markdown("### ⚖️ Equilibrium Check (Sigma Fy = 0)")
                
                sum_R = sum(R.values()) / 1000.0 # kN
                
                # Sum Loads from calc_loads_df
                sum_Load = 0.0
                for _, l in calc_loads_df.iterrows():
                    force = l['mag']
                    if l['type'] == 'U':
                        force = l['mag'] * l['dist']
                    sum_Load += force
                
                sum_Load_kN = sum_Load / 1000.0
                diff = sum_R - sum_Load_kN 
                
                st.write(f"Total Applied Load (↓): **{sum_Load_kN:.2f} kN**")
                st.write(f"Total Reaction (↑): **{sum_R:.2f} kN**")
                
                if abs(diff) < 0.1:
                    st.success(f"✅ OK! Balance Error = {diff:.4f} kN")
                else:
                    st.error(f"❌ Unbalanced! Error = {diff:.4f} kN")

            # 2. Deflection Control
            with ec2:
                st.markdown("### 📉 Deflection Control")
                max_span_L = max(spans) * 1000 
                allowable_def = max_span_L / 240.0
                
                st.write(f"Max Deflection: **{d_abs_max:.2f} mm**")
                st.write(f"Allowable Limit (L/240): **{allowable_def:.2f} mm**")
                
                if d_abs_max <= allowable_def:
                    st.success(f"✅ PASS")
                else:
                    st.warning(f"⚠️ EXCEEDS LIMIT")

        # --- PART 4: LOAD DETAILS ---
        with st.expander("🧮 Load Combination Details", expanded=False):
            st.markdown("### How Loads are Combined:")
            # FIXED: w_sw_kN is now defined correctly at the top
            st.write(f"**1. Self-Weight (SW):** {w_sw_kN:.3f} kN/m (Calculated automatically)")
            
            st.write("**2. Load Breakdown per Span:**")
            
            breakdown_data = []
            for i in range(n_spans):
                row = calc_loads_df[(calc_loads_df['span_index'] == i) & (calc_loads_df['type'] == 'U')]
                if not row.empty:
                    # Be careful: row might have multiple entries if logic failed, but our logic ensures unique 'U' per span for Total
                    # Actually, if user puts partial UDL, it is separate. We only want the "Total Combined" one.
                    # Let's filter by description or just take the max one to be safe, or sum them (though our logic merged them).
                    
                    # Better logic: Find the one with 'Total Combined' desc
                    total_row = row[row['desc'].str.contains("Total Combined")]
                    
                    if not total_row.empty:
                        total_udl = total_row.iloc[0]['mag'] / 1000.0 # kN/m
                        user_part = total_udl - w_sw_kN
                        breakdown_data.append({
                            "Span": i+1,
                            "Self-Weight": f"{w_sw_kN:.3f}",
                            "User UDL": f"{user_part:.3f}",
                            "TOTAL UDL (kN/m)": f"**{total_udl:.3f}**"
                        })
            if breakdown_data:
                st.table(pd.DataFrame(breakdown_data))
            else:
                st.write("No Combined UDLs found (structure might be empty).")

    # ================= TAB 2: DESIGN & REPORT =================
    with tab2:
        st.header("Reinforced Concrete Design (WSD/SDM Concept)")
        st.markdown(f"**Material:** f'c = {params['fc']:.0f} MPa, fy = {params['fy']:.0f} MPa | **Section:** {params['b']*100:.0f}x{params['h']*100:.0f} cm")
        
        design_res = []
        span_start = 0
        for i, span_len in enumerate(spans):
            span_end = span_start + span_len
            
            mask = (res_df['x'] >= span_start) & (res_df['x'] <= span_end)
            span_data = res_df[mask]
            
            if span_data.empty: continue
            
            mu_pos = span_data['moment'].max() / 1000 
            mu_neg = abs(span_data['moment'].min()) / 1000
            vu_max = span_data['shear'].abs().max() / 1000
            
            d = params['h'] - 0.05
            
            # Design
            As_pos, rho_pos, _, steps_pos = rc_design.design_beam_flexure(mu_pos, params['b'], d, params['fc'], params['fy'])
            As_neg, rho_neg, _, steps_neg = rc_design.design_beam_flexure(mu_neg, params['b'], d, params['fc'], params['fy'])
            s_req, _, steps_shear = rc_design.check_shear(vu_max, params['b'], d, params['fc'], params['fy'])
            
            def get_bars(As): return max(2, int(np.ceil(As / (3.14159*(0.008)**2)))) 
            
            design_res.append({
                'span': i+1,
                'pos': {'As': As_pos, 'n': get_bars(As_pos)},
                'neg': {'As': As_neg, 'n': get_bars(As_neg)},
                'shear': {'s': s_req}
            })
            
            with st.expander(f"📘 Span {i+1} Design (Mu+={mu_pos:.2f}, Mu-={mu_neg:.2f})", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Bottom Steel**")
                    for s in steps_pos: st.latex(s)
                with c2:
                    st.markdown("**Top Steel**")
                    for s in steps_neg: st.latex(s)
                st.markdown("**Shear Design**")
                for s in steps_shear: st.latex(s)

            span_start += span_len

        st.markdown("---")
        st.subheader("3. Detailing Preview")
        c_det1, c_det2 = st.columns([1, 2])
        with c_det1:
            st.write("**Cross Section**")
            if design_res:
                fig_sec = section_plotter.plot_section(
                    params['b'], params['h'], 40, 16, 
                    design_res[0]['neg']['n'], design_res[0]['pos']['n'], 
                    "RB6@200", params['fc'], params['fy']
                )
                st.pyplot(fig_sec)
        with c_det2:
            st.write("**Longitudinal Section**")
            if design_res:
                fig_long = section_plotter.plot_longitudinal_section_detailed(spans, sup_df, design_res, params['h'], 40)
                st.pyplot(fig_long)
