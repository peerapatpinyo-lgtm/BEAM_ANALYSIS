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

    # 2. Model Inputs
    # รับค่า sup_df กลับมาด้วย
    n_spans, spans, sup_df, stable = input_handler.render_model_inputs(params)
    
    st.markdown("---")

    # 3. Loads Input
    # ส่ง sup_df เข้าไปเพื่อเช็ค Warning Moment @ Pin
    loads = input_handler.render_loads(n_spans, spans, params, sup_df)

    st.markdown("---")

    # 4. Calculation & Solver
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        if not stable:
            st.error("❌ Structure is Unstable! Please check support conditions (Need at least 2 Pins/Rollers or 1 Fixed).")
            return
            
        # Initialize Solver
        beam_solver = solver.BeamSolver(spans, sup_df, loads, E=params['E'], I=params['I'])
        
        # Solve
        try:
            df_results, reactions = beam_solver.solve()
            
            # --- 5. Visualization (แก้ไขลำดับตัวแปรตรงนี้ครับ) ---
            # ลำดับที่ถูกต้อง: df, reac, spans, sup_df, loads, units...
            design_view.draw_interactive_diagrams(
                df_results, 
                reactions,       # ใส่ reactions เข้ามาเป็นตัวที่ 2
                spans, 
                sup_df, 
                loads, 
                params['u_force'], 
                params['u_len']
            )
            
            # --- 6. Result Tables ---
            design_view.render_result_tables(
                df_results, 
                reactions, 
                spans, 
                params['u_force'], 
                params['u_len']
            )
            
        except Exception as e:
            st.error(f"Analysis Failed: {str(e)}")
            # st.exception(e) # Uncomment for debug

if __name__ == "__main__":
    main()
