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
    st.error("🚨 Structure is unstable! Please check supports.")
else:
    # --- 1. Prepare Loads (User Loads + Self Weight) ---
    w_sw_kN = params['b'] * params['h'] * 24.0 # kN/m
    
    calc_loads_list = []
    
    # 1.1 Add Self-Weight
    for i in range(n_spans):
        calc_loads_list.append({
            'span_index': i,
            'type': 'U',
            'mag': w_sw_kN * 1000, # N/m
            'dist': spans[i],
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
    # Solver now uses Timoshenko Theory automatically using params['b'] and params['h']
    x_eval, M, V, D, R = solver.solve_beam(spans, sup_df, calc_loads_df, params)
    
    res_df = pd.DataFrame({
        'x': x_eval,
        'moment': M,
        'shear': V,
        'deflection': D * 1000 # m to mm
    })
    
    # --- 3. DISPLAY RESULTS ---
    tab1, tab2 = st.tabs(["📊 1. Analysis Results", "📝 2. RC Design & Report"])
    
    # ================= TAB 1: DIAGRAMS =================
    with tab1:
        # --- Calculation Breakdown ---
        with st.expander("🧮 ดูวิธีคำนวณ (Calculation Details)", expanded=False):
            st.markdown("### 1. Theory & Parameters")
            st.markdown("""
            * **Method:** Finite Element Method (Direct Stiffness)
            * **Beam Theory:** **Timoshenko Beam** (Includes Shear Deformation)
            * **Shear Correction Factor ($k$):** 5/6 (Rectangular Section)
            """)
            st.latex(r"\Phi = \frac{12EI}{kGA L^2}")
            
            st.markdown("---")
            st.markdown("### 2. Self-Weight")
            st.latex(f"w_{{sw}} = {params['b']:.2f} \\times {params['h']:.2f} \\times 24 = \\mathbf{{{w_sw_val:.3f}}} \\text{{ kN/m}}")
            
            if R:
                st.markdown("---")
                st.markdown("**3. Reactions (Result)**")
                r_data = []
                for node_idx in sorted([int(k[1:]) for k in R.keys()]):
                    key = f"R{node_idx}"
                    if key in R:
                        val = R[key] / 1000.0
                        r_data.append({"Node": node_idx, "Reaction (kN)": f"{val:.3f}"})
                st.table(pd.DataFrame(r_data))

        st.info(f"Analysis includes **Self-Weight** ({w_sw_kN:.2f} kN/m). Results are **Service Load**.")
        
        # Plot
        fig = design_view.plot_analysis_results(res_df, spans, sup_df, calc_loads_df, R)
        st.plotly_chart(fig, use_container_width=True)

    # ================= TAB 2: DESIGN =================
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
            
            # Design Forces (Service -> Need Factors if doing SDM strictly)
            mu_pos = span_data['moment'].max() / 1000 
            mu_neg = abs(span_data['moment'].min()) / 1000
            vu_max = span_data['shear'].abs().max() / 1000
            
            d = params['h'] - 0.05 # Approx d
            
            # Call Design Functions (Assuming rc_design.py is unchanged)
            As_pos, rho_pos, _, steps_pos = rc_design.design_beam_flexure(mu_pos, params['b'], d, params['fc'], params['fy'])
            As_neg, rho_neg, _, steps_neg = rc_design.design_beam_flexure(mu_neg, params['b'], d, params['fc'], params['fy'])
            s_req, _, steps_shear = rc_design.check_shear(vu_max, params['b'], d, params['fc'], params['fy'])
            
            # Simple Bar Count
            def get_bars(As): return max(2, int(np.ceil(As / (3.14*(0.008)**2)))) # Approx using DB16 ref
            
            design_res.append({
                'span': i+1,
                'pos': {'As': As_pos, 'n': get_bars(As_pos)},
                'neg': {'As': As_neg, 'n': get_bars(As_neg)},
                'shear': {'s': s_req}
            })
            
            with st.expander(f"📘 Span {i+1} Design (Mu+={mu_pos:.2f}, Mu-={mu_neg:.2f})", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Bottom Steel**")
                    for s in steps_pos: st.latex(s)
                with col2:
                    st.write("**Top Steel**")
                    for s in steps_neg: st.latex(s)
                st.write("**Shear Stirrups**")
                for s in steps_shear: st.latex(s)

            span_start += span_len
            
        st.markdown("---")
        st.subheader("Detailing Preview")
        c1, c2 = st.columns([1,2])
        with c1:
            st.write("Section View (Span 1)")
            # Assuming plot_section is available
            fig_sec = section_plotter.plot_section(params['b'], params['h'], 40, 16, design_res[0]['neg']['n'], design_res[0]['pos']['n'], "RB6@200", params['fc'], params['fy'])
            st.pyplot(fig_sec)
        with c2:
            st.write("Longitudinal View")
            fig_long = section_plotter.plot_longitudinal_section_detailed(spans, sup_df, design_res, params['h'], 40)
            st.pyplot(fig_long)
