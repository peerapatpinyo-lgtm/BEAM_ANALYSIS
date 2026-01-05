import streamlit as st
import pandas as pd
import numpy as np

# Import Custom Modules
import input_handler
import solver
import rc_design
import design_view

# Page Config
st.set_page_config(page_title="Pro Beam Design", layout="wide", page_icon="🏗️")

def main():
    st.title("🏗️ Structural Beam Analysis Professional")
    st.markdown("---")
    
    # 1. Sidebar & Inputs
    params = input_handler.render_sidebar()
    n_spans, spans, sup_df, stable = input_handler.render_model_inputs(params)
    
    st.markdown("---")
    
    # 2. Load Inputs (Updated: No sup_df needed here anymore)
    # *** จุดที่แก้คือบรรทัดนี้ครับ ***
    raw_loads_df = input_handler.render_loads(n_spans, spans, params)
    
    st.markdown("---")
    
    # 3. Analysis Action
    if st.button("🚀 Run Analysis & Design", type="primary", use_container_width=True):
        
        if not stable:
            st.error("❌ Structure is unstable (Mechanism). Please add more supports.")
            return

        # 3.1 Load Factoring
        factored_loads = []
        if not raw_loads_df.empty:
            for _, l in raw_loads_df.iterrows():
                # Apply Load Factors
                factor = params['gamma_dead'] if l['case'] == 'DL' else params['gamma_live']
                
                new_l = l.to_dict()
                new_l['mag'] *= factor # Factor magnitude
                factored_loads.append(new_l)
        
        loads_df_factored = pd.DataFrame(factored_loads) if factored_loads else pd.DataFrame()
        
        # 3.2 Solver Execution
        beam_solver = solver.BeamSolver(spans, sup_df, loads_df_factored, params['E'], params['I'])
        
        try:
            df_res, reactions = beam_solver.solve()
            
            if df_res is None:
                st.error("⚠️ Error: Singular Matrix. Structure is unstable.")
                return

            # 3.3 Visualization (Pro Version)
            # Pass RAW loads (Service Loads) for visualization
            design_view.draw_interactive_diagrams(
                df_res, reactions, spans, sup_df, 
                raw_loads_df.to_dict('records') if not raw_loads_df.empty else []
            )
            
            # 3.4 Tables
            design_view.render_result_tables(df_res, reactions, spans, "kg", "m")
            
            # 3.5 RC Design Summary
            st.markdown("---")
            st.header("🧱 Reinforced Concrete Design Checks")
            
            cum_dist = [0] + list(np.cumsum(spans))
            cols = st.columns(len(spans))
            
            for i, span_col in enumerate(cols):
                with span_col:
                    st.subheader(f"Span {i+1}")
                    start, end = cum_dist[i], cum_dist[i+1]
                    span_res = df_res[(df_res['x'] >= start) & (df_res['x'] <= end)]
                    
                    if span_res.empty: continue

                    # Design Forces
                    m_pos = span_res['moment'].max()
                    m_neg = span_res['moment'].min() 
                    v_max = span_res['shear'].abs().max()
                    
                    # Call RC Design Module
                    res_pos = rc_design.calculate_flexure_sdm(m_pos, "Mid (+M)", params['b'], params['h'], params['cover'], params)
                    res_neg = rc_design.calculate_flexure_sdm(m_neg, "Sup (-M)", params['b'], params['h'], params['cover'], params)
                    stir, shear_logs = rc_design.calculate_shear_capacity(v_max, params['b'], params['h'], params['cover'], params)
                    
                    # Display Card
                    st.markdown(f"""
                    <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; border:1px solid #ddd; margin-bottom:10px">
                        <p style="margin:0; font-size:14px"><b>Bottom Bars (+):</b> <span style="color:blue">{res_pos['Bars']}</span></p>
                        <small style="color:gray">Mu: {res_pos['Mu']:.0f} kg-m</small>
                        <hr style="margin:5px 0">
                        <p style="margin:0; font-size:14px"><b>Top Bars (-):</b> <span style="color:red">{res_neg['Bars']}</span></p>
                        <small style="color:gray">Mu: {res_neg['Mu']:.0f} kg-m</small>
                        <hr style="margin:5px 0">
                        <p style="margin:0; font-size:14px"><b>Stirrups:</b> {stir}</p>
                        <small style="color:gray">Vu: {v_max:.0f} kg</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("Detailed Calcs"):
                         st.caption("Positive Moment Check:")
                         for l in res_pos['Log']: st.markdown(f"<small>{l}</small>", unsafe_allow_html=True)
                         st.markdown("---")
                         st.caption("Shear Check:")
                         for l in shear_logs: st.markdown(f"<small>{l}</small>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Analysis Failed: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
