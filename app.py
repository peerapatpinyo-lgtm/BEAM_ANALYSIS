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
    # --- 0. Analysis Settings (Load Factors) ---
    st.markdown("### ⚙️ Analysis Settings & Load Factors")
    
    # Select Mode
    mode_select = st.radio(
        "Select Analysis Mode:",
        ["Service Load (1.0DL + 1.0LL)", "Ultimate / Custom Factors"],
        horizontal=True
    )
    
    col_fac1, col_fac2, col_fac3 = st.columns([1, 1, 2])
    
    # Logic for Factors
    if mode_select.startswith("Service"):
        f_dl = 1.0
        f_ll = 1.0
        with col_fac1:
            st.number_input("Dead Load Factor (DL)", value=1.0, disabled=True, key="fdl_serv")
        with col_fac2:
            st.number_input("Live Load Factor (LL)", value=1.0, disabled=True, key="fll_serv")
        tag = "M/V (Service)"
        st.info("ℹ️ Using **Service Load** for Deflection Check.")
        
    else:
        # Ultimate / Custom Mode
        with col_fac1:
            f_dl = st.number_input("Dead Load Factor (DL)", value=1.4, step=0.1, format="%.2f", key="fdl_ult")
        with col_fac2:
            f_ll = st.number_input("Live Load Factor (LL)", value=1.7, step=0.1, format="%.2f", key="fll_ult")
        tag = "Mu/Vu (Ultimate)"
        st.warning(f"⚡ Using **Factored Load**: {f_dl} DL + {f_ll} LL")

    # --- 1. Prepare & Combine Loads ---
    # 1.1 Calculate Self-Weight (Dead Load)
    w_sw_base_kN = params['b'] * params['h'] * 24.0   # kN/m (Unfactored)
    w_sw_factored_kN = w_sw_base_kN * f_dl            # kN/m (Factored)
    w_sw_factored_N_m = w_sw_factored_kN * 1000.0     # N/m
    
    # 1.2 Initialize bucket for Total UDL per span
    span_total_udl = {i: w_sw_factored_N_m for i in range(n_spans)}
    
    combined_loads_list = []
    
    # 1.3 Process User Loads (Assumed as LIVE LOAD)
    if not loads_df.empty:
        for _, row in loads_df.iterrows():
            try:
                s_idx = int(row['span_index'])
                if s_idx >= n_spans: continue 
                
                l_type = row['type']
                mag_base = row['mag']   # Unfactored
                mag_factored = mag_base * f_ll # Apply Live Load Factor
                
                dist = row['dist'] 
                current_span_len = spans[s_idx]
                
                # CHECK: If UDL covers FULL span -> MERGE
                if l_type == 'U' and dist >= (current_span_len - 0.01):
                    span_total_udl[s_idx] += mag_factored
                else:
                    # Point Load or Partial UDL -> KEEP SEPARATE
                    combined_loads_list.append({
                        'span_index': s_idx,
                        'type': l_type,
                        'mag': mag_factored,
                        'dist': dist,
                        'desc': f'Point/Partial (LL x {f_ll})'
                    })
            except Exception as e:
                pass
    
    # 1.4 Add Merged UDLs to list
    for i in range(n_spans):
        total_mag = span_total_udl[i]
        if total_mag > 0:
            combined_loads_list.append({
                'span_index': i,
                'type': 'U',
                'mag': total_mag,
                'dist': spans[i], 
                'desc': 'Total Combined (Factored DL+LL)'
            })
            
    calc_loads_df = pd.DataFrame(combined_loads_list)

    # --- 2. Solve Beam ---
    x_eval, M, V, D, R = solver.solve_beam(spans, sup_df, calc_loads_df, params)
    
    res_df = pd.DataFrame({
        'x': x_eval,
        'moment': M,
        'shear': V,
        'deflection': D * 1000 
    })
    
    # --- 3. DISPLAY RESULTS (TABS) ---
    tab1, tab2 = st.tabs(["📊 1. Analysis Results & Checks", "📝 2. RC Design & Report"])
    
    # ================= TAB 1: DIAGRAMS & CHECKS =================
    with tab1:
        # --- PART 1: PLOT DIAGRAMS ---
        st.plotly_chart(design_view.plot_analysis_results(res_df, spans, sup_df, calc_loads_df, R), use_container_width=True)

        st.markdown("---")

        # --- PART 2: ANALYSIS SUMMARY & REACTIONS ---
        st.subheader("📌 Analysis Summary")
        
        # 2.1 Max/Min Values
        v_max = res_df['shear'].abs().max()/1000
        m_max_pos = res_df['moment'].max()/1000
        m_max_neg = res_df['moment'].min()/1000
        d_abs_max = res_df['deflection'].abs().max()
        
        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
        col_res1.metric(f"Max Shear", f"{v_max:.2f} kN")
        col_res2.metric(f"Max Moment (+)", f"{m_max_pos:.2f} kNm")
        col_res3.metric(f"Max Moment (-)", f"{m_max_neg:.2f} kNm")
        col_res4.metric("Max Deflection", f"{d_abs_max:.2f} mm")
        
        # 2.2 REACTION TABLE (FIXED KEYERROR)
        st.markdown("### 📍 Support Reactions")
        if R:
            reaction_data = []
            
            # Helper to safely find support type by location
            def get_sup_type_safe(node_idx):
                # 1. Calculate X position of this node
                node_x = sum(spans[:node_idx])
                
                # 2. Check standard column names for position
                # (Since we don't know if it's 'position', 'x', or 'location')
                target_col = None
                for col in ['position', 'x', 'location', 'dist']:
                    if col in sup_df.columns:
                        target_col = col
                        break
                
                if target_col:
                    # Find support at this X (allow small float error)
                    match = sup_df[np.abs(sup_df[target_col] - node_x) < 0.01]
                    if not match.empty:
                        return match.iloc[0]['type']
                
                # Fallback: Check if 'node_index' exists
                if 'node_index' in sup_df.columns:
                     match = sup_df[sup_df['node_index'] == node_idx]
                     if not match.empty: return match.iloc[0]['type']

                return "Support"

            total_reaction = 0
            for key, val in R.items():
                node_idx = int(key[1:]) # 'R0' -> 0
                val_kN = val / 1000.0
                total_reaction += val_kN
                
                reaction_data.append({
                    "Node": node_idx,
                    "Support Type": get_sup_type_safe(node_idx),
                    "Reaction Force (kN)": val_kN
                })
            
            # Sort by Node
            df_reac = pd.DataFrame(reaction_data).sort_values(by="Node")
            
            # Display Table
            st.dataframe(
                df_reac.style.format({"Reaction Force (kN)": "{:.2f}"}),
                use_container_width=True,
                hide_index=True
            )
            st.caption(f"**Total Reaction:** {total_reaction:.2f} kN")
        
        st.markdown("---")

        # --- PART 3: CALCULATION REPORT ---
        with st.expander("🧮 Load Combination Details (Report)", expanded=False):
            st.markdown("### 1. Load Factors")
            st.write(f"- Dead Load Factor (DL): **{f_dl:.2f}**")
            st.write(f"- Live Load Factor (LL): **{f_ll:.2f}**")
            st.write(f"**Note:** Self-Weight is treated as DL. User Inputs are treated as LL.")

            st.markdown("### 2. Self-Weight (DL)")
            st.latex(f"w_{{sw}} = {params['b']} \\times {params['h']} \\times 24 = {w_sw_base_kN:.3f} \\text{{ kN/m}}")
            st.latex(f"w_{{sw,factored}} = {w_sw_base_kN:.3f} \\times {f_dl} = \\mathbf{{{w_sw_factored_kN:.3f}}} \\text{{ kN/m}}")

            st.markdown("### 3. Load Breakdown by Span")
            breakdown_data = []
            for i in range(n_spans):
                # Calculate User part (LL)
                user_udl_base = 0.0
                if not loads_df.empty:
                     user_rows = loads_df[(loads_df['span_index'] == i) & (loads_df['type'] == 'U') & (loads_df['dist'] >= spans[i]-0.01)]
                     if not user_rows.empty:
                         user_udl_base = user_rows['mag'].sum() / 1000.0
                
                total_factored = (w_sw_base_kN * f_dl) + (user_udl_base * f_ll)
                
                breakdown_data.append({
                    "Span": i+1,
                    "SW (DL x Factor)": f"{w_sw_base_kN:.3f} x {f_dl}",
                    "User (LL x Factor)": f"{user_udl_base:.3f} x {f_ll}",
                    "TOTAL UDL (kN/m)": f"**{total_factored:.3f}**"
                })
            
            st.table(pd.DataFrame(breakdown_data))

    # ================= TAB 2: DESIGN & REPORT =================
    with tab2:
        if mode_select.startswith("Service"):
            st.warning("⚠️ **Warning:** You are in 'Service Load' mode (Factors = 1.0). RC Design typically requires Ultimate Load factors. Please switch mode in 'Analysis Settings'.")
        
        st.header(f"Reinforced Concrete Design ({tag})")
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
