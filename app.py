import streamlit as st
import pandas as pd
import numpy as np

# Custom modules
import input_handler
import solver
import rc_design
import design_view

st.set_page_config(page_title="Beam Analysis Pro", layout="wide")

def main():
    st.title("🏗️ Beam Analysis & Design (Custom Solver)")
    
    # 1. Inputs
    params = input_handler.render_sidebar()
    n_spans, spans, sup_df, stable = input_handler.render_model_inputs(params)
    raw_loads_df = input_handler.render_loads(n_spans, spans, params, sup_df)
    
    st.markdown("---")
    
    # 2. Process
    if st.button("🚀 Run Analysis", type="primary"):
        if not stable:
            st.error("Unstable Structure!")
            return
            
        # Factoring Loads
        factored_loads = []
        if not raw_loads_df.empty:
            for _, l in raw_loads_df.iterrows():
                f = params['gamma_dead'] if l['case'] == 'DL' else params['gamma_live']
                new_l = l.to_dict()
                new_l['mag'] *= f
                factored_loads.append(new_l)
        
        loads_df = pd.DataFrame(factored_loads) if factored_loads else None
        
        # Solve
        beam = solver.BeamSolver(spans, sup_df, loads_df, E=params['E'], I=params['I'])
        try:
            df_res, reactions = beam.solve()
            if df_res is None:
                st.error("Singular Matrix Error")
                return
                
            # Visuals
            design_view.draw_interactive_diagrams(
                df_res, reactions, spans, sup_df, 
                raw_loads_df.to_dict('records') if not raw_loads_df.empty else [],
                unit_force=params['u_force']
            )
            design_view.render_result_tables(df_res, reactions, spans, params['u_force'], params['u_len'])
            
            # Design Loop
            st.header("🧱 Reinforced Concrete Design")
            design_results = []
            cum_dist = [0] + list(np.cumsum(spans))
            
            for i in range(len(spans)):
                start, end = cum_dist[i], cum_dist[i+1]
                span_res = df_res[(df_res['x'] >= start) & (df_res['x'] <= end)]
                
                # Critical Values
                mu_pos = span_res['moment'].max()
                mu_neg = span_res['moment'].min()
                vu_max = span_res['shear'].abs().max()
                
                # Flexure Design
                if mu_pos > 1e-3:
                    design_results.append(rc_design.calculate_flexure_sdm(mu_pos, f"Span {i+1} Mid (+)", params['b'], params['h'], params['cover'], params))
                if abs(mu_neg) > 1e-3:
                    design_results.append(rc_design.calculate_flexure_sdm(mu_neg, f"Span {i+1} Sup (-)", params['b'], params['h'], params['cover'], params))
            
            # Show Cards
            cols = st.columns(3)
            for idx, res in enumerate(design_results):
                with cols[idx % 3]:
                    color = "green" if "OK" in res['Status'] else "red"
                    st.markdown(f"""
                    <div style="border:1px solid #ddd; padding:10px; border-radius:5px; margin-bottom:10px">
                        <h4>{res['Type']}</h4>
                        <p>Mu: {res['Mu']:.2f}</p>
                        <p><b>{res['Bars']}</b></p>
                        <p style="color:{color}">{res['Status']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
