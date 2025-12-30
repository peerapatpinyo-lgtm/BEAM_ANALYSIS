import streamlit as st
import pandas as pd
import solver
import input_handler
import design_view 

# Config หน้าจอ
st.set_page_config(page_title="Beam Analysis", layout="wide")

# CSS: ปรับให้ดู Clean ขึ้น ลดความหนาของ Element ต่างๆ
st.markdown("""
<style>
    /* ใช้ฟอนต์ที่อ่านง่าย */
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Sarabun', sans-serif; }

    /* ปรับปุ่มให้ดู modern */
    .stButton button {
        width: 100%; font-weight: 600; border-radius: 8px; padding-top: 10px; padding-bottom: 10px;
    }
    /* ลด padding ด้านบน */
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

def main():
    # Header แบบเรียบๆ
    st.title("🏗️ โปรแกรมวิเคราะห์คานต่อเนื่อง")
    st.caption("Linear Elastic Analysis | Finite Element Method")
    st.divider() # ใช้เส้นแบ่งบางๆ แทนขอบหนาๆ

    col_input, col_output = st.columns([35, 65], gap="large")

    # === PANEL ซ้าย (Input) - เอา container(border=True) ออก ===
    with col_input:
        # ใช้ Header ย่อยแทนกรอบ
        params = input_handler.render_sidebar()
        n, spans, sup_df, stable = input_handler.render_model_inputs(params)
        
        st.markdown("###") # เว้นบรรทัด
        loads = input_handler.render_loads(n, spans, params)
        
        st.markdown("###")
        # ปุ่มคำนวณ
        run_btn = st.button("⚡ คำนวณ (Calculate)", type="primary", disabled=not stable)
        
        if not stable:
            st.warning("⚠️ กรุณาตรวจสอบจุดรองรับ (โครงสร้างไม่เสถียร)")

    # === PANEL ขวา (Output) ===
    with col_output:
        if run_btn or st.session_state.get('analysis_done'):
            if run_btn:
                try:
                    engine = solver.BeamSolver(spans, sup_df, loads)
                    df_res, reactions = engine.solve()
                    st.session_state.update({'analysis_done': True, 'df_res': df_res, 'reactions': reactions, 'spans': spans, 'sup_df': sup_df, 'loads': loads})
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()

            df = st.session_state['df_res']
            reac = st.session_state['reactions']
            spans = st.session_state['spans']

            # แสดงผล (พื้นหลังกราฟจะขาวสะอาดแล้ว)
            design_view.draw_interactive_diagrams(
                df, spans, st.session_state['sup_df'], 
                st.session_state['loads'], params['u_force'], params['u_len']
            )
            
            st.divider()
            design_view.render_result_tables(df, reac, spans)
            
        else:
            # หน้าจอเริ่มต้นแบบ Clean
            st.info("👈 กรอกข้อมูลที่แถบซ้ายมือ แล้วกดปุ่ม 'คำนวณ'")

if __name__ == "__main__":
    main()
