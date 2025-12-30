import streamlit as st
import pandas as pd
# import beam_analysis  <-- ลบอันเก่าทิ้ง
import solver  # <-- ใช้อันใหม่ที่เพิ่งสร้าง
import input_handler
import design_view

# ... (ส่วน Setup CSS และ Header เหมือนเดิม) ...

def main():
    st.title("🏗️ RC Beam Analysis & Design (Custom Engine)")
    
    # ... (ส่วน Sidebar และ Input เหมือนเดิม) ...
    
    with col_input:
        n, spans, sup_df, stable = input_handler.render_model_inputs(params)
        loads = input_handler.render_loads(n, spans, params)
        
        st.markdown("###")
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True, disabled=not stable):
            with st.spinner("Analyzing with Custom Matrix Engine..."):
                try:
                    # เรียกใช้ solver ตัวใหม่
                    engine = solver.BeamSolver(spans, sup_df, loads)
                    df_res, reactions = engine.solve()
                    
                    if df_res is not None:
                        st.session_state['analysis_done'] = True
                        st.session_state['df_res'] = df_res
                        st.session_state['spans'] = spans
                        st.session_state['sup_df'] = sup_df
                        st.session_state['loads'] = loads
                    else:
                        st.error("Calculation Failed")
                except Exception as e:
                    st.error(f"Solver Error: {e}")

    # ... (ส่วนแสดงผลด้านล่าง เหมือนเดิม) ...

if __name__ == "__main__":
    main()
