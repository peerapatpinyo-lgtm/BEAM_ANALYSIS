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
    is_service = False
    if mode_select.startswith("Service"):
        f_dl = 1.0
        f_ll = 1.0
        with col_fac1:
            st.number_input("Dead Load Factor (DL)", value=1.0, disabled=True, key="fdl_serv")
        with col_fac2:
            st.number_input("Live Load Factor (LL)", value=1.0, disabled=True, key="fll_serv")
        tag = "M/V (Service)"
        st.info("ℹ️ Using **Service Load** for Deflection Check.")
        is_service = True
    else:
        with col_fac1:
            f_dl = st.number_input("Dead Load Factor (DL)", value=1.4, step=0.1, format="%.2f", key="fdl_ult")
        with col_fac2:
            f_ll = st.number_input("Live Load Factor (LL)", value=1.7, step=0.1, format="%.2f", key="fll_ult")
        tag = "Mu/Vu (Ultimate)"
        st.warning(f"⚡ Using **Factored Load**: {f_dl} DL + {f_ll} LL")

    # --- 1. Prepare & Combine Loads ---
    try:
        # 1.1 Calculate Self-Weight (Dead Load)
        w_sw_base_kN = params['b'] * params['h'] * 24.0   # kN/m
        w_sw_factored_kN = w_sw_base_kN * f_dl
        w_sw_factored_N_m = w_sw_factored_kN * 1000.0
        
        # 1.2 Initialize bucket for Total UDL per span
        span_total_udl = {i: w_sw_factored_N_m for i in range(n_spans)}
        
        combined_loads_list = []
        
        # 1.3 Process User Loads
        if not loads_df.empty:
            for _, row in loads_df.iterrows():
                try:
                    s_idx = int(row['span_index'])
                    if s_idx >= n_spans: continue 
                    
                    l_type = row['type']
                    mag_base = row['mag']   
                    mag_factored = mag_base * f_ll 
                    
                    dist = row['dist'] 
                    current_span_len = spans[s_idx]
                    
                    # CHECK: If UDL covers FULL span -> MERGE
                    if l_type == 'U' and dist >= (current_span_len - 0.01):
                        span_total_udl[s_idx] += mag_factored
                    else:
                        # Point Load or Partial UDL
                        combined_loads_list.append({
                            'span_index': s_idx,
                            'type': l_type,
                            'mag': mag_factored,
                            'dist': dist,
                            'desc': 'User (Partial/Point)'
                        })
                except Exception:
                    continue
        
        # 1.4 Add Merged UDLs
        for i in range(n_spans):
            total_mag = span_total_udl[i]
            if total_mag > 0:
                combined_loads_list.append({
                    'span_index': i,
                    'type': 'U',
                    'mag': total_mag,
                    'dist': spans[i], # For Solver: usually this is treated as Length if full span
                    'desc': 'Total Combined UDL'
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
        
        # --- 3. DISPLAY RESULTS ---
        tab1, tab2 = st.tabs(["📊 1. Analysis Results & Checks", "📝 2. RC Design & Report"])
        
        # ================= TAB 1: DIAGRAMS & CHECKS =================
        with tab1:
            # --- PART 1: PLOT DIAGRAMS ---
            st.plotly_chart(design_view.plot_analysis_results(res_df, spans, sup_df, calc_loads_df, R), use_container_width=True)
            
            st.info("💡 **Note on Shear Diagram (SFD):** The slightly inclined lines at point loads are due to plotting resolution. Theoretically, these should be vertical jumps.")

            st.markdown("---")

            # --- PART 2: ANALYSIS SUMMARY ---
            st.subheader("📌 Analysis Summary")
            
            v_max = res_df['shear'].abs().max()/1000
            m_max_pos = res_df['moment'].max()/1000
            m_max_neg = res_df['moment'].min()/1000
            d_abs_max = res_df['deflection'].abs().max()
            
            col_res1, col_res2, col_res3, col_res4 = st.columns(4)
            col_res1.metric(f"Max Shear ({tag})", f"{v_max:.2f} kN")
            col_res2.metric(f"Max Moment (+)", f"{m_max_pos:.2f} kNm")
            col_res3.metric(f"Max Moment (-)", f"{m_max_neg:.2f} kNm")
            col_res4.metric("Max Deflection", f"{d_abs_max:.2f} mm")
            
            # --- PART 3: REACTION TABLE (SAFE MODE) ---
            st.markdown("### 📍 Support Reactions")
            if R:
                try:
                    reaction_data = []
                    
                    # Safe function to get support type
                    def get_sup_type_safe(node_idx):
                        try:
                            # Calculate Node X
                            node_x = sum(spans[:node_idx])
                            
                            # Try to match by Position (X)
                            target_col = None
                            for col in sup_df.columns:
                                if str(col).lower() in ['position', 'x', 'location', 'dist']:
                                    target_col = col
                                    break
                            
                            if target_col:
                                match = sup_df[np.abs(sup_df[target_col] - node_x) < 0.05]
                                if not match.empty:
                                    return match.iloc[0]['type']
                            
                            # Fallback: Try match by Index if node_index not present
                            if node_idx < len(sup_df):
                                return sup_df.iloc[node_idx]['type']
                                
                        except Exception:
                            pass
                        return "Support"

                    total_reaction = 0
                    for key, val in R.items():
                        # Key format usually 'R0', 'R1'
                        node_idx = int(str(key).replace('R', ''))
                        val_kN = val / 1000.0
                        total_reaction += val_kN
                        
                        reaction_data.append({
                            "Node": node_idx,
                            "Type": get_sup_type_safe(node_idx),
                            "Reaction (kN)": val_kN
                        })
                    
                    df_reac = pd.DataFrame(reaction_data).sort_values(by="Node")
                    st.dataframe(df_reac.style.format({"Reaction (kN)": "{:.2f}"}), use_container_width=True, hide_index=True)
                
                except Exception as e:
                    st.warning(f"⚠️ Could not display reaction table details. (Error: {e})")
                    st.write(R) # Show raw dict as backup
                
            st.markdown("---")
            
            # --- PART 4: ENGINEERING CHECKS (SAFE MODE) ---
            with st.expander("✅ Engineering Checks (Equilibrium & Deflection)", expanded=True):
                ec1, ec2 = st.columns(2)
                
                # 1. Equilibrium Check
                with ec1:
                    st.markdown("### ⚖️ Equilibrium Check")
                    try:
                        sum_R_kN = sum(R.values()) / 1000.0
                        
                        # Estimate Applied Load Sum (Simplified)
                        sum_Load_kN = 0.0
                        
                        # Add SW
                        total_length = sum(spans)
                        sum_Load_kN += (w_sw_factored_kN * total_length)
                        
                        # Add User Loads
                        if not loads_df.empty:
                             for _, row in loads_df.iterrows():
                                 mag_k = row['mag'] / 1000.0 * f_ll
                                 if row['type'] == 'P':
                                     sum_Load_kN += mag_k
                                 elif row['type'] == 'U':
                                     # For User UDL, assume full span if dist is large, or partial
                                     # This is an approximation if user inputs complex partials
                                     l_len = spans[int(row['span_index'])] 
                                     if row['dist'] > 0: l_len -= row['dist'] # Simple assumption
                                     sum_Load_kN += (mag_k * l_len)

                        diff = sum_R_kN - sum_Load_kN 
                        
                        st.write(f"Total Reactions (↑): **{sum_R_kN:.2f} kN**")
                        st.caption(f"Note: Total Load (↓) is approx. {sum_Load_kN:.2f} kN")
                        
                        if abs(diff) < (sum_Load_kN * 0.05 + 1.0): # 5% tolerance + 1kN
                            st.success(f"✅ Equilibrium Pass")
                        else:
                            st.warning(f"⚠️ Balance Check: Diff = {diff:.2f} kN (May due to partial load calc)")
                            
                    except Exception as e:
                        st.error("Could not verify equilibrium.")

                # 2. Deflection Control
                with ec2:
                    st.markdown("### 📉 Deflection Control")
                    try:
                        if not is_service:
                            st.info("⚠️ Note: Using **Ultimate Load**. Deflection is overestimated.")
                        
                        max_span_L = max(spans) * 1000 
                        allowable_def = max_span_L / 240.0
                        
                        st.write(f"Max Deflection: **{d_abs_max:.2f} mm**")
                        st.write(f"Limit (L/240): **{allowable_def:.2f} mm**")
                        
                        if d_abs_max <= allowable_def:
                            st.success(f"✅ Pass")
                        else:
                            if is_service:
                                st.error(f"❌ Exceeds Limit")
                            else:
                                st.warning(f"⚠️ Exceeds Limit (Factored Load)")
                    except Exception:
                        st.error("Error checking deflection.")

            # --- PART 5: DETAILED LOAD TABLE (SAFE MODE) ---
            with st.expander("🧮 Load Breakdown Table", expanded=False):
                try:
                    detailed_loads = []
                    
                    # 1. Self Weight
                    for i in range(n_spans):
                        detailed_loads.append({
                            "Span": i+1,
                            "Source": "Self-Weight",
                            "Type": "DL",
                            "Base": f"{w_sw_base_kN:.2f} kN/m",
                            "Factor": f"x{f_dl}",
                            "Factored": f"**{w_sw_factored_kN:.2f}**"
                        })
                    
                    # 2. User Loads
                    if not loads_df.empty:
                        for _, row in loads_df.iterrows():
                            if row['span_index'] < n_spans:
                                l_t = "Point (P)" if row['type'] == 'P' else "Uniform (w)"
                                unit = "kN" if row['type'] == 'P' else "kN/m"
                                mag_b = row['mag'] / 1000.0
                                mag_f = mag_b * f_ll
                                
                                detailed_loads.append({
                                    "Span": int(row['span_index'])+1,
                                    "Source": "User Input",
                                    "Type": "LL",
                                    "Base": f"{mag_b:.2f} {unit}",
                                    "Factor": f"x{f_ll}",
                                    "Factored": f"**{mag_f:.2f}**"
                                })
                    
                    st.table(pd.DataFrame(detailed_loads))
                except Exception as e:
                    st.error(f"Error creating load table: {e}")

        # ================= TAB 2: DESIGN =================
        with tab2:
            try:
                if is_service:
                    st.warning("⚠️ **Warning:** You are in 'Service Load' mode. Switch to 'Ultimate Load' for Strength Design.")
                
                st.header(f"Reinforced Concrete Design ({tag})")
                
                design_res = []
                span_start = 0
                for i, span_len in enumerate(spans):
                    span_end = span_start + span_len
                    mask = (res_df['x'] >= span_start) & (res_df['x'] <= span_end)
                    span_data = res_df[mask]
                    
                    if not span_data.empty:
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
            except Exception as e:
                st.error(f"Design module encountered an error: {e}")

    except Exception as e:
        st.error(f"⚠️ Calculation Error: {e}")
        st.write("Please check your input values (e.g. Length = 0, or Loads on non-existent spans).")
