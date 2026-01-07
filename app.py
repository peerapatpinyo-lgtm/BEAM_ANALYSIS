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
    # --- 0. Analysis Settings (NEW) ---
    st.markdown("### ⚙️ Analysis Settings")
    col_mode1, col_mode2 = st.columns([1, 3])
    with col_mode1:
        load_case = st.radio(
            "Select Load Case:",
            ["Service Load (Unfactored)", "Ultimate Load (Factored)"],
            help="Service: 1.0DL + 1.0LL (for Deflection)\nUltimate: 1.4DL + 1.7LL (for Strength Design)"
        )
    
    # Define Factors based on selection
    if "Ultimate" in load_case:
        f_dl = 1.4
        f_ll = 1.7
        tag = "Mu/Vu"
        st.info(f"⚡ **Designing with Ultimate Load:** $1.4 DL + 1.7 LL$")
    else:
        f_dl = 1.0
        f_ll = 1.0
        tag = "M/V"
        st.success(f"👀 **Checking Service Load:** $1.0 DL + 1.0 LL$")

    # --- 1. Prepare & Combine Loads ---
    # 1.1 Calculate Self-Weight (Dead Load)
    # Base SW
    w_sw_base_kN = params['b'] * params['h'] * 24.0   # kN/m
    # Factored SW
    w_sw_factored_kN = w_sw_base_kN * f_dl
    w_sw_factored_N_m = w_sw_factored_kN * 1000.0
    
    # 1.2 Initialize bucket for Total UDL per span
    # Key = span_index, Value = Total Magnitude (N/m)
    span_total_udl = {i: w_sw_factored_N_m for i in range(n_spans)}
    
    combined_loads_list = []
    
    # 1.3 Process User Loads (Assumed as LIVE LOAD)
    if not loads_df.empty:
        for _, row in loads_df.iterrows():
            try:
                s_idx = int(row['span_index'])
                if s_idx >= n_spans: continue 
                
                l_type = row['type']
                mag_base = row['mag']   # N or N/m (Unfactored)
                mag_factored = mag_base * f_ll # Apply Live Load Factor
                
                dist = row['dist'] # m
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
                st.warning(f"Skipping invalid load row: {e}")
    
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
        st.caption(f"Diagrams showing **{load_case}**")
        fig = design_view.plot_analysis_results(res_df, spans, sup_df, calc_loads_df, R)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # --- PART 2: ANALYSIS SUMMARY ---
        st.subheader("📌 Analysis Summary (Max/Min Values)")
        
        v_max = res_df['shear'].abs().max()/1000
        m_max_pos = res_df['moment'].max()/1000
        m_max_neg = res_df['moment'].min()/1000
        d_abs_max = res_df['deflection'].abs().max()
        
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Max Shear ({tag})", f"{v_max:.2f} kN")
        c2.metric(f"Max Moment ({tag})", f"{m_max_pos:.2f} / {m_max_neg:.2f} kNm")
        c3.metric("Max Deflection", f"{d_abs_max:.2f} mm")
        
        # --- PART 3: CALCULATION REPORT (NEW) ---
        with st.expander("🧮 Load Combination Calculation Report", expanded=True):
            st.markdown("### 1. Load Factors Definition")
            st.latex(f"Factor_{{DL}} = {f_dl}, \\quad Factor_{{LL}} = {f_ll}")
            st.write(f"**Assumption:** Self-Weight is Dead Load (DL). User Inputs are Live Loads (LL).")

            st.markdown("### 2. Self-Weight Calculation (DL)")
            st.latex(f"w_{{sw}} = {params['b']:.2f} \\times {params['h']:.2f} \\times 24 = \\mathbf{{{w_sw_base_kN:.3f}}} \\text{{ kN/m}}")
            st.latex(f"w_{{sw,factored}} = {f_dl} \\times {w_sw_base_kN:.3f} = \\mathbf{{{w_sw_factored_kN:.3f}}} \\text{{ kN/m}}")

            st.markdown("### 3. Total Load per Span")
            
            breakdown_data = []
            for i in range(n_spans):
                # Calculate User part back from the total logic for display
                # Note: This is simplified for display of UDLs
                
                # Get User UDL (Unfactored) for this span
                user_udl_base = 0.0
                if not loads_df.empty:
                     # Filter strictly UDL full span
                     user_rows = loads_df[(loads_df['span_index'] == i) & (loads_df['type'] == 'U') & (loads_df['dist'] >= spans[i]-0.01)]
                     if not user_rows.empty:
                         user_udl_base = user_rows['mag'].sum() / 1000.0 # kN/m
                
                total_factored = (w_sw_base_kN * f_dl) + (user_udl_base * f_ll)
                
                breakdown_data.append({
                    "Span": i+1,
                    "SW (DL)": f"{w_sw_base_kN:.3f}",
                    "User (LL)": f"{user_udl_base:.3f}",
                    "Equation": f"({f_dl}×DL) + ({f_ll}×LL)",
                    "TOTAL (kN/m)": f"**{total_factored:.3f}**"
                })
            
            st.table(pd.DataFrame(breakdown_data))

    # ================= TAB 2: DESIGN & REPORT =================
    with tab2:
        if "Ultimate" not in load_case:
            st.warning("⚠️ **Warning:** You are currently in 'Service Load' mode. RC Design usually requires 'Ultimate Load'. Switch to Ultimate mode for standard strength design.")
        
        st.header("Reinforced Concrete Design")
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
