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
    # --- 1. Prepare Loads (User Loads + Self Weight) ---
    # Self-Weight (Concrete ~ 24 kN/m3)
    w_sw_kN = params['b'] * params['h'] * 24.0
    
    calc_loads_list = []
    
    # 1.1 Add Self-Weight
    for i in range(n_spans):
        calc_loads_list.append({
            'span_index': i,
            'type': 'U',
            'mag': w_sw_kN * 1000, # N/m
            'dist': spans[i],      # Full span
            'desc': 'Self-Weight'
        })
        
    # 1.2 Add User Loads
    if not loads_df.empty:
        for _, row in loads_df.iterrows():
            calc_loads_list.append({
                'span_index': row['span_index'],
                'type': row['type'],
                'mag': row['mag'], 
                'dist': row['dist'],
                'desc': 'User Load'
            })
            
    calc_loads_df = pd.DataFrame(calc_loads_list)

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
        # --- A. ANALYSIS SUMMARY (Max/Min) ---
        st.subheader("📌 Analysis Summary (Max/Min Values)")
        
        # Calculate Global Extremes
        v_max_pos = res_df['shear'].max()/1000
        v_max_neg = res_df['shear'].min()/1000
        m_max_pos = res_df['moment'].max()/1000
        m_max_neg = res_df['moment'].min()/1000
        d_max = res_df['deflection'].max() # Positive is DOWN in common FEA sign (if solver fixed) or UP?
        # Let's check magnitude
        d_abs_max = res_df['deflection'].abs().max()
        
        # Summary Columns
        col_sum1, col_sum2, col_sum3 = st.columns(3)
        col_sum1.metric("Max Shear (V+)", f"{v_max_pos:.2f} kN")
        col_sum1.metric("Min Shear (V-)", f"{v_max_neg:.2f} kN")
        
        col_sum2.metric("Max Moment (Sagging +)", f"{m_max_pos:.2f} kNm")
        col_sum2.metric("Max Moment (Hogging -)", f"{m_max_neg:.2f} kNm")
        
        col_sum3.metric("Max Deflection", f"{d_abs_max:.2f} mm")
        
        # --- B. EQUILIBRIUM & DEFLECTION CHECK ---
        with st.expander("✅ Engineering Checks (Equilibrium & Deflection)", expanded=True):
            ec1, ec2 = st.columns(2)
            
            # 1. Equilibrium Check (Sigma Fy = 0)
            with ec1:
                st.markdown("### ⚖️ Equilibrium Check ($\Sigma F_y = 0$)")
                
                # Sum Reactions
                sum_R = sum(R.values()) / 1000.0 # kN (Up is +)
                
                # Sum Loads (Need to calculate based on type)
                sum_Load = 0.0
                for _, l in calc_loads_df.iterrows():
                    # Load mag is positive in DF. 
                    # If Point: Load = mag
                    # If UDL: Load = mag * dist
                    force = l['mag']
                    if l['type'] == 'U':
                        force = l['mag'] * l['dist']
                    sum_Load += force
                
                sum_Load_kN = sum_Load / 1000.0
                diff = sum_R - sum_Load_kN # Should be near 0
                
                st.write(f"Total Applied Load ($\downarrow$): **{sum_Load_kN:.2f} kN**")
                st.write(f"Total Reaction ($\uparrow$): **{sum_R:.2f} kN**")
                
                if abs(diff) < 0.1:
                    st.success(f"✅ OK! Balance Error = {diff:.4f} kN")
                else:
                    st.error(f"❌ Unbalanced! Error = {diff:.4f} kN")

            # 2. Deflection Control Check
            with ec2:
                st.markdown("### 📉 Deflection Control")
                
                # Criteria: L/240 (Common for Total Load)
                max_span_L = max(spans) * 1000 # mm
                allowable_def = max_span_L / 240.0
                
                st.write(f"Max Deflection: **{d_abs_max:.2f} mm**")
                st.write(f"Allowable Limit ($L/240$): **{allowable_def:.2f} mm** (based on longest span)")
                
                if d_abs_max <= allowable_def:
                    st.success(f"✅ PASS ( < L/240 )")
                else:
                    st.warning(f"⚠️ EXCEEDS LIMIT (Consider increasing Depth 'h')")

        st.markdown("---")

        # --- C. CALCULATION DETAILS (Reactions) ---
        with st.expander("🧮 Reaction Calculation Details", expanded=False):
            st.markdown("### 1. Self-Weight")
            st.latex(f"w_{{sw}} = {params['b']:.2f} \\times {params['h']:.2f} \\times 24 = \\mathbf{{{w_sw_kN:.3f}}} \\text{{ kN/m}}")
            
            st.markdown("### 2. Reaction Forces ($R_y$)")
            st.markdown("Calculated from Global Stiffness Matrix $[K]$:")
            st.latex(r"\{R\} = [K]\{d\} - \{F_{equiv}\}")
            
            if R:
                r_data = []
                for node_idx in sorted([int(k[1:]) for k in R.keys()]):
                    key = f"R{node_idx}"
                    if key in R:
                        val = R[key] / 1000.0 # kN
                        r_data.append({"Node": node_idx, "Reaction (kN)": f"{val:.3f}"})
                st.table(pd.DataFrame(r_data))
        
        # --- D. PLOT DIAGRAMS ---
        st.info(f"ℹ️ **Note:** Analysis uses **Timoshenko Beam Theory** (Service Load).")
        fig = design_view.plot_analysis_results(res_df, spans, sup_df, calc_loads_df, R)
        st.plotly_chart(fig, use_container_width=True)

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
            
            # Design Functions
            As_pos, rho_pos, _, steps_pos = rc_design.design_beam_flexure(mu_pos, params['b'], d, params['fc'], params['fy'])
            As_neg, rho_neg, _, steps_neg = rc_design.design_beam_flexure(mu_neg, params['b'], d, params['fc'], params['fy'])
            s_req, _, steps_shear = rc_design.check_shear(vu_max, params['b'], d, params['fc'], params['fy'])
            
            def get_bars(As): return max(2, int(np.ceil(As / (3.14159*(0.008)**2)))) # DB16 approx area
            
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
