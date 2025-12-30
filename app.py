import streamlit as st
import pandas as pd
import input_handler
import solver
import design_view

# Page Config
st.set_page_config(page_title="Beam Analysis Pro", layout="wide", page_icon="🏗️")

def main():
    st.title("🏗️ Structural Beam Analysis Professional")
    st.markdown("---")

    # 1. Sidebar Settings
    params = input_handler.render_sidebar()

    # 2. Main Inputs (Model)
    # รับค่า sup_df ออกมาด้วย
    n_spans, spans, sup_df, stable = input_handler.render_model_inputs(params)
    
    st.markdown("---")

    # 3. Loads Input
    # *** แก้ไขจุดนี้: ส่ง sup_df เข้าไปให้ฟังก์ชันด้วย ***
    loads = input_handler.render_loads(n_spans, spans, params, sup_df)

    st.markdown("---")

    # 4. Calculation & Solver
    if st.button("🚀 Run Analysis", type="primary"):
        if not stable:
            st.error("❌ Structure is Unstable! Please add more supports (e.g., at least 2 Pins/Rollers or 1 Fixed).")
            return
            
        # Initialize Solver
        beam_solver = solver.BeamSolver(spans, sup_df, loads, E=params['E'], I=params['I'])
        
        # Solve
        try:
            df_results, reactions = beam_solver.solve()
            
            # 5. Visualization
            design_view.draw_interactive_diagrams(df_results, spans, sup_df, loads, params['u_force'], params['u_len'])
            
            # 6. Result Tables
            design_view.render_result_tables(df_results, reactions, spans)
            
        except Exception as e:
            st.error(f"Analysis Failed: {str(e)}")
            st.code(e)

if __name__ == "__main__":
    main()
