import streamlit as st
import pandas as pd
import numpy as np

# Import Modules
import input_handler
import solver
import rc_design
import design_view

st.set_page_config(page_title="Pro Beam Design", layout="wide", page_icon="🏗️")

def main():
    st.title("🏗️ Structural Beam Analysis & Design Pro")
    st.markdown("---")
    
    # 1. Inputs
    params = input_handler.render_sidebar()
    n_spans, spans, sup_df, stable = input_handler.render_model_inputs(params)
    raw_loads = input_handler.render_loads(n_spans, spans, params, sup_df)
    
    st.markdown("---")
    
    # 2. Analyze
    if st.button("🚀 Run Analysis & Design", type="primary", use_container_width=True):
        if not stable:
            st.error("Structure is unstable. Please add supports.")
            return
            
        # Create Factored Loads
        factored_loads = []
        if not raw_loads.empty:
            for _, l in raw_loads.iterrows():
                factor = params['gamma_dead'] if l['case'] == 'DL' else params['gamma_live']
                new_l = l.to_dict()
                new_l['mag'] *= factor
                factored_loads.append(new_l)
        
        loads_df = pd.DataFrame(factored_loads) if factored_loads else pd.DataFrame()
        
        # Solver
        beam = solver.BeamSolver(spans, sup_df, loads_df, params['E'], params['I'])
        df_res, reactions = beam.solve()
        
        if df_res is None:
            st.error("Singular Matrix: Structure unstable.")
            return
            
        # 3. Visualization
        design_view.draw_interactive_diagrams(
            df_res, reactions, spans, sup_df, 
            raw_loads.to_dict('records') if not raw_loads.empty else [],
        )
        design_view.render_result_tables(df_res, reactions, spans, "kg", "m")
        
        # 4. RC Design
        st.markdown("---")
        st.header("🧱 Concrete Design Results")
        
        cum_dist = [0] + list(np.cumsum(spans))
        
        for i in range(len(spans)):
            st.subheader(f"Span {i+1} Design")
            start, end = cum_dist[i], cum_dist[i+1]
            span_res = df_res[(df_res['x'] >= start) & (df_res['x'] <= end)]
            
            # Critical Moments
            m_pos = span_res['moment'].max()
            m_neg = span_res['moment'].min() # Often at supports
            v_max = span_res['shear'].abs().max()
            
            col1, col2, col3 = st.columns(3)
            
            # +M Design
            with col1:
                res = rc_design.calculate_flexure_sdm(m_pos, "Midspan (+M)", params['b'], params['h'], params['cover'], params)
                color = "green" if "OK" in res['Status'] else "red"
                st.markdown(f"**{res['Type']}**")
                st.info(f"{res['Bars']}")
                with st.expander("Calculation Log"):
                    for log in res['Log']: st.markdown(f"- {log}", unsafe_allow_html=True)
            
            # -M Design
            with col2:
                # Typically -M is at support, simplify by taking min of span (conservative if cantilever involved)
                res = rc_design.calculate_flexure_sdm(m_neg, "Support (-M)", params['b'], params['h'], params['cover'], params)
                st.markdown(f"**{res['Type']}**")
                st.warning(f"{res['Bars']}")
                with st.expander("Calculation Log"):
                     for log in res['Log']: st.markdown(f"- {log}", unsafe_allow_html=True)
                     
            # Shear Design
            with col3:
                st.markdown(f"**Shear (V_max = {v_max:.0f} kg)**")
                stir, logs = rc_design.calculate_shear_capacity(v_max, params['b'], params['h'], params['cover'], params)
                st.error(f"{stir}")
                with st.expander("Shear Log"):
                    for log in logs: st.write(f"- {log}")

if __name__ == "__main__":
    main()
