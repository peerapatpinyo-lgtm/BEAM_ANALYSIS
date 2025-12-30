import streamlit as st
import pandas as pd
import solver           # ใช้ Engine ที่เขียนเอง (ไม่ต้องลง library เพิ่ม)
import input_handler
import design_view

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="RC Beam Pro", layout="wide", page_icon="🏗️")

# Custom CSS
st.markdown("""
<style>
    .stApp { font-family: 'Sarabun', sans-serif; background-color: #F8F9FA; }
    h1, h2, h3 { color: #1565C0; }
    .stButton>button { background-color: #1565C0; color: white; border-radius: 6px; height: 3em; }
    div[data-testid="stExpander"] { background-color: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🏗️ RC Beam Analysis & Design (Custom Engine)")
    
    # 2. Sidebar Settings
    params = input_handler.render_sidebar()
    
    # 3. กำหนด Layout (ส่วนที่ขาดหายไปคราวก่อน)
    col_input, col_preview = st.columns([1, 1.5])
    
    # 4. ส่วน Input (ซ้าย)
    with col_input:
        n, spans, sup_df, stable = input_handler.render_model_inputs(params)
        loads = input_handler.render_loads(n, spans, params)
        
        st.markdown("###")
        # ปุ่มคำนวณ
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True, disabled=not stable):
            with st.spinner("Analyzing structure..."):
                try:
                    # เรียกใช้ Solver ตัวใหม่ (Matrix Method)
                    engine = solver.BeamSolver(spans, sup_df, loads)
                    df_res, reactions = engine.solve()
                    
                    if df_res is not None:
                        # บันทึกผลลง Session State
                        st.session_state['analysis_done'] = True
                        st.session_state['df_res'] = df_res
                        st.session_state['reactions'] = reactions
                        st.session_state['spans'] = spans
                        st.session_state['sup_df'] = sup_df
                        st.session_state['loads'] = loads
                    else:
                        st.error("Structure Unstable or Calculation Failed!")
                except Exception as e:
                    st.error(f"Solver Error: {e}")

    # 5. ส่วนแสดงผล (ขวา)
    if st.session_state.get('analysis_done'):
        df = st.session_state['df_res']
        
        with col_preview:
            st.subheader("📊 Analysis Results")
            # วาดกราฟ
            design_view.draw_diagrams(df, st.session_state['spans'], st.session_state['sup_df'], 
                                      st.session_state['loads'], params['u_force'], params['u_len'])
        
        # --- ส่วน Design (ด้านล่าง) ---
        st.markdown("---")
        st.header("✨ Section Design & Detailing")
        st.info("👇 Adjust beam sizes here. Reinforcement will update automatically.")
        
        # สร้าง Input สำหรับขนาดหน้าตัดแต่ละช่วง
        n_spans = len(st.session_state['spans'])
        cols = st.columns(n_spans)
        span_props = []
        
        for i in range(n_spans):
            with cols[i]:
                st.markdown(f"**Span {i+1}**")
                with st.container(border=True):
                    # Default ค่าเริ่มต้น
                    b = st.number_input(f"Width b (cm)", value=25.0, step=5.0, key=f"des_b_{i}")
                    h = st.number_input(f"Depth h (cm)", value=50.0, step=5.0, key=f"des_h_{i}")
                    cv = st.number_input(f"Cover (cm)", value=3.0, step=0.5, key=f"des_c_{i}")
                    span_props.append({"b": b, "h": h, "cv": cv})

        # เรียกฟังก์ชัน Design และแสดงตารางสรุป
        design_view.render_design_results(df, params, st.session_state['spans'], span_props, st.session_state['sup_df'])
        
    else:
        # ข้อความต้อนรับตอนยังไม่กดปุ่ม
        with col_preview:
            st.info("👈 Please define geometry and loads, then click 'Run Analysis'.")

if __name__ == "__main__":
    main()
