import streamlit as st
import pandas as pd
import input_handler
import solver
import design_view

def main():
    st.set_page_config(page_title="Beam Analysis", layout="wide")
    
    # 1. Render Sidebar & Get Parameters (Includes E, I)
    params = input_handler.render_sidebar()
    
    col_input, col_result = st.columns([1, 2])
    
    with col_input:
        # 2. Render Model Inputs
        n_spans, spans, sup_df, stable = input_handler.render_model_inputs(params)
        
        # Save to session state to persist during interaction
        st.session_state['sup_df'] = sup_df
        
        # 3. Render Loads
        loads = input_handler.render_loads(n_spans, spans, params)
        st.session_state['loads'] = loads

    with col_result:
        if not stable:
            st.error("Structure is unstable! Please check supports.")
        else:
            if st.button("🚀 Analyze Beam", type="primary", use_container_width=True):
                # 4. SOLVE (Pass E and I here!)
                beam_solver = solver.BeamSolver(
                    spans, 
                    st.session_state['sup_df'], 
                    st.session_state['loads'],
                    E=params['E'], # รับค่าจาก input_handler
                    I=params['I']  # รับค่าจาก input_handler
                )
                
                df_res, reac = beam_solver.solve()
                
                # Save results
                st.session_state['result_df'] = df_res
                st.session_state['reactions'] = reac
                
            # 5. Display Results
            if 'result_df' in st.session_state and not st.session_state['result_df'].empty:
                design_view.draw_interactive_diagrams(
                    st.session_state['result_df'], 
                    spans, 
                    st.session_state['sup_df'], 
                    st.session_state['loads'], 
                    params['u_force'], 
                    params['u_len']
                )
                
                design_view.render_result_tables(
                    st.session_state['result_df'],
                    st.session_state['reactions'],
                    spans
                )

if __name__ == "__main__":
    main()
