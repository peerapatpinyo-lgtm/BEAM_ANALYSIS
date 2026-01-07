import streamlit as st
import pandas as pd
import numpy as np

# ... (ส่วน render_sidebar_params และ render_model_inputs_main เหมือนเดิม ไม่ต้องแก้) ...
# Copy ฟังก์ชันด้านล่างนี้ไปทับ render_sidebar_params และ render_model_inputs_main ของเดิมได้เลย หรือคงของเดิมไว้ถ้ามันทำงานถูกแล้ว
# แต่เพื่อความชัวร์ ผมใส่ให้ครบไฟล์ครับ

def render_sidebar_params():
    st.subheader("1. Material Properties")
    E_gpa = st.number_input("Elastic Modulus (E) [GPa]", value=30.0, step=1.0)
    
    st.subheader("2. Beam Section")
    c1, c2 = st.columns(2)
    with c1:
        b = st.number_input("Width (b) [m]", value=0.25, step=0.05)
    with c2:
        h = st.number_input("Depth (h) [m]", value=0.50, step=0.05)
        
    I = (b * h**3) / 12
    return {"E": E_gpa * 1e9, "b": b, "h": h, "I": I}

def render_model_inputs_main(params):
    st.header("1. Model Geometry")
    col_span, col_viz = st.columns([1, 2])
    
    with col_span:
        st.markdown("##### Span Configuration")
        n_spans = st.number_input("Number of Spans", 1, 10, 2)
        spans = []
        with st.container():
            for i in range(n_spans):
                s = st.number_input(f"Span {i+1} Length (m)", 0.1, 50.0, 5.0, key=f"span_{i}")
                spans.append(s)

    with col_viz:
        st.markdown("##### Support Conditions")
        cum_dist = [0] + list(np.cumsum(spans))
        n_nodes = len(cum_dist)
        default_types = ["Pin"] + ["Roller"] * (n_nodes - 1)
        
        sup_data = {"id": range(n_nodes), "x": cum_dist, "type": default_types}
        df_sup_init = pd.DataFrame(sup_data)

        edited_sup = st.data_editor(
            df_sup_init,
            column_config={
                "id": "Node",
                "x": st.column_config.NumberColumn("Position (m)", disabled=True, format="%.2f"),
                "type": st.column_config.SelectboxColumn("Type", options=["Pin", "Roller", "Fixed", "Free"], required=True)
            },
            hide_index=True, use_container_width=True, key="sup_editor"
        )
        
    stable = True
    types = edited_sup['type'].tolist()
    if types.count('Fixed') == 0 and types.count('Pin') == 0 and types.count('Roller') < 2: stable = False
    if types.count('Free') == len(types): stable = False

    return n_spans, spans, edited_sup, stable

# --- [ส่วนที่แก้หลัก] Load Input ---
def render_loads_main(n_spans, spans, params, sup_df):
    st.header("2. Loads Definition")
    
    # Init Session State
    if "load_list" not in st.session_state:
        st.session_state.load_list = []
        
    with st.expander("➕ Add New Load", expanded=True):
        # ใช้ Form เพื่อป้องกันการ Refresh หน้าจอก่อนกดปุ่ม
        with st.form("load_form", clear_on_submit=False):
            c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
            with c1:
                l_type = st.selectbox("Load Type", ["Point Load (P)", "Uniform Load (U)"])
            with c2:
                span_idx_raw = st.selectbox("On Span", range(n_spans), format_func=lambda x: f"Span {x+1}")
            
            # Dynamic inputs inside form are tricky, we use conservative defaults
            # แต่เนื่องจาก Form ไม่ Interactive ทันที เราต้องดึงค่า span_idx ออกมาใช้นอก Form หรือยอมรับค่า Default
            # เพื่อความง่ายและเสถียร: เราจะใช้ logic การคำนวณตอน submit
            
            with c3:
                pos_val = st.number_input("Position / Start (m)", min_value=0.0, value=0.0)
            with c4:
                mag_val = st.number_input("Magnitude (kN, kN/m)", value=10.0)
                
            # Extra fields
            c5, c6 = st.columns(2)
            with c5:
                # ถ้าเป็น Uniform ให้ใส่ความยาว
                dist_val = st.number_input("Length (for UDL only)", min_value=0.0, value=0.0)
            with c6:
                case_val = st.selectbox("Load Case", ["DL", "LL"])

            submitted = st.form_submit_button("Add Load")
            
            if submitted:
                # Validation Logic
                current_span_len = spans[span_idx_raw]
                
                # Adjust Dist based on type
                final_dist = dist_val if "Uniform" in l_type else 0.0
                if "Uniform" in l_type and final_dist == 0.0:
                    final_dist = current_span_len - pos_val # Auto-fill rest of span
                
                new_load = {
                    "id": len(st.session_state.load_list),
                    "type": "P" if "Point" in l_type else "U",
                    "span_index": span_idx_raw,
                    "x": pos_val,
                    "mag": mag_val * 1000,
                    "dist": final_dist,
                    "case": case_val
                }
                st.session_state.load_list.append(new_load)
                st.rerun()

    if st.session_state.load_list:
        st.markdown("##### Current Loads List")
        df = pd.DataFrame(st.session_state.load_list)
        df_show = df.copy()
        df_show['Span'] = df_show['span_index'].apply(lambda x: f"Span {x+1}")
        df_show['Magnitude'] = df_show['mag'] / 1000.0
        df_show['Position'] = df_show.apply(lambda r: f"x={r['x']:.2f}" if r['type'] == 'P' else f"{r['x']:.2f} - {r['x']+r['dist']:.2f} m", axis=1)
        
        st.dataframe(
            df_show[['type', 'Span', 'Magnitude', 'Position', 'case']], 
            use_container_width=True, hide_index=True
        )
        
        if st.button("🗑️ Clear All Loads"):
            st.session_state.load_list = []
            st.rerun()
            
        return pd.DataFrame(st.session_state.load_list)
    return None
