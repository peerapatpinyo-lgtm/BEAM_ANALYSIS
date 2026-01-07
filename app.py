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
    # Calculate Self-Weight (Concrete ~ 24 kN/m3)
    # w_sw (kN/m) = b(m) * h(m) * 24
    w_sw_kN = params['b'] * params['h'] * 24.0
    
    # Create Calculation Load DataFrame
    calc_loads_list = []
    
    # 1.1 Add Self-Weight for each span
    for i in range(n_spans):
        calc_loads_list.append({
            'span_index': i,
            'type': 'U',
            'mag': w_sw_kN * 1000, # Convert to N/m for solver
            'dist': spans[i],      # Full span
            'desc': 'Self-Weight'
        })
        
    # 1.2 Add User Loads
    if not loads_df.empty:
        for _, row in loads_df.iterrows():
            calc_loads_list.append({
                'span_index': row['span_index'],
                'type': row['type'],
                'mag': row['mag'], # Already in N (or N/m) from input_handler
                'dist': row['dist'],
                'desc': 'User Load'
            })
            
    calc_loads_df = pd.DataFrame(calc_loads_list)

    # --- 2. Solve Beam ---
    # Note: solver expects N, m units
    x_eval, M, V, D, R = solver.solve_beam(spans, sup_df, calc_loads_df, params)
    
    # Store results
    res_df = pd.DataFrame({
        'x': x_eval,
        'moment': M,
        'shear': V,
        'deflection': D * 1000 # Convert m to mm
    })
    
    # --- 3. DISPLAY RESULTS (TABS) ---
    tab1, tab2 = st.tabs(["📊 1. Analysis Results", "📝 2. RC Design & Report"])
    
    # ================= TAB 1: DIAGRAMS =================
    with tab1:
        # --- Calculation Breakdown ---
        with st.expander("🧮 Calculation Details", expanded=False):
            st.markdown("### 1. Theory & Parameters")
            st.markdown("""
            * **Method:** Finite Element Method (Direct Stiffness)
            * **Beam Theory:** **Timoshenko Beam** (Includes Shear Deformation)
            * **Shear Correction Factor ($k$):** 5/6 (Rectangular Section)
            """)
            st.latex(r"\Phi = \frac{12EI}{kGA L^2}")
            
            st.markdown("---")
            st.markdown("### 2. Self-Weight Calculation")
            st.markdown("Formula: $w_{sw} = b \\times h \\times \\gamma_{conc}$")
            
            st.latex(f"w_{{sw}} = {params['b']:.2f} \\text{{ m}} \\times {params['h']:.2f} \\text{{ m}} \\times 24 \\text{{ kN/m}}^3")
            
            # FIX: Used correct variable name 'w_sw_kN' here
            st.latex(f"w_{{sw}} = \\mathbf{{{w_sw_kN:.3f}}} \\text{{ kN/m}}")
            st.caption("*This load is automatically added as a Uniform Load.")

            if R:
                st.markdown("---")
                st.markdown("### 3. Reaction Results")
                r_data = []
                for node_idx in sorted([int(k[1:]) for k in R.keys()]):
                    key = f"R{node_idx}"
                    if key in R:
                        val = R[key] / 1000.0 # Convert N to kN
                        r_data.append({"Node": node_idx, "Reaction (kN)": f"{val:.3f}"})
                st.table(pd.DataFrame(r_data))

        st.info(f"ℹ️ **Note:** Analysis includes **Self-Weight** ({w_sw_kN:.2f} kN/m). Graphs show **Service Load** (Unfactored) results.")
        
        # Plot Diagrams
        fig = design_view.plot_analysis_results(res_df, spans, sup_df, calc_loads_df, R)
        st.plotly_chart(fig, use_container_width=True)

    # ================= TAB 2: DESIGN & REPORT =================
    with tab2:
        st.header("Reinforced Concrete Design (WSD/SDM Concept)")
        
        # Material Summary
        st.markdown(f"**Material:** f'c = {params['fc']:.0f} MPa, fy = {params['fy']:.0f} MPa | **Section:** {params['b']*100:.0f}x{params['h']*100:.0f} cm")
        
        design_res = []
        
        # Loop through spans for design
        span_start = 0
        for i, span_len in enumerate(spans):
            span_end = span_start + span_len
            
            # Get Max Forces in this span
            mask = (res_df['x'] >= span_start) & (res_df['x'] <= span_end)
            span_data = res_df[mask]
            
            if span_data.empty: continue
                
            mu_pos = span_data['moment'].max() / 1000 # kNm
            mu_neg = abs(span_data['moment'].min()) / 1000 # kNm
            vu_max = span_data['shear'].abs().max() / 1000 # kN
            
            # Effective depth
            cover = 40 # mm
            db_est = 16 # mm
            d = params['h'] - (cover + 10 + db_est/2)/1000
            
            # --- RC DESIGN CALCULATION ---
            # 1. Positive Moment
            As_pos, rho_pos, stat_pos, steps_pos = rc_design.design_beam_flexure(
                mu_pos, params['b'], d, params['fc'], params['fy']
            )
            
            # 2. Negative Moment
            As_neg, rho_neg, stat_neg, steps_neg = rc_design.design_beam_flexure(
                mu_neg, params['b'], d, params['fc'], params['fy']
            )
            
            # 3. Shear
            s_req, stat_shear, steps_shear = rc_design.check_shear(
                vu_max, params['b'], d, params['fc'], params['fy']
            )
            
            # Convert As to Rebar count
            def calc_n_bars(As_req, db):
                area_one = 3.14159 * (db/2)**2
                if As_req <= 0: return 2
                return max(2, int(np.ceil(As_req / area_one)))

            n_pos = calc_n_bars(As_pos, 16)
            n_neg = calc_n_bars(As_neg, 16)
            
            design_res.append({
                'span': i+1,
                'pos': {'As': As_pos, 'n': n_pos},
                'neg': {'As': As_neg, 'n': n_neg},
                'shear': {'s': s_req, 'status': stat_shear},
                'db': 16
            })
            
            # --- EXPANDER FOR NOTE ---
            with st.expander(f"📘 Span {i+1} Design Calculation (L={span_len}m)", expanded=False):
                col_calc1, col_calc2 = st.columns(2)
                
                with col_calc1:
                    st.markdown("### 🧱 Flexure Design (Bottom)")
                    st.caption(f"Mu(+) = {mu_pos:.2f} kNm")
                    for s in steps_pos: st.latex(s)
                    st.success(f"Use: {n_pos}-DB16")
                    
                with col_calc2:
                    st.markdown("### 🧱 Flexure Design (Top)")
                    st.caption(f"Mu(-) = {mu_neg:.2f} kNm")
                    for s in steps_neg: st.latex(s)
                    st.success(f"Use: {n_neg}-DB16")
                
                st.divider()
                st.markdown("### ✂️ Shear Design")
                st.caption(f"Vu = {vu_max:.2f} kN")
                for s in steps_shear: st.latex(s)
                
                stirrup_txt = f"RB6@{min(200, int(s_req if s_req else 200))}mm"
                if s_req == 0: stirrup_txt = "Section Fail"
                st.info(f"Stirrups: {stirrup_txt}")

            span_start += span_len

        # --- DETAILING SECTION ---
        st.markdown("---")
        st.subheader("3. Detailing Preview")
        
        c_det1, c_det2 = st.columns([1, 2])
        
        with c_det1:
            st.markdown("**Cross Section (Example Span 1)**")
            if design_res:
                fig_sec = section_plotter.plot_section(
                    params['b'], params['h'], 40, 16, 
                    design_res[0]['neg']['n'], design_res[0]['pos']['n'], 
                    "RB6@200", params['fc'], params['fy']
                )
                st.pyplot(fig_sec)
        
        with c_det2:
            st.markdown("**Longitudinal Section**")
            if design_res:
                fig_long = section_plotter.plot_longitudinal_section_detailed(spans, sup_df, design_res, params['h'], 40)
                st.pyplot(fig_long)
