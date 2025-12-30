import streamlit as st
import pandas as pd

def render_sidebar():
    st.sidebar.markdown("## ⚙️ Settings")
    with st.sidebar.expander("📏 Units & Standards", expanded=True):
        unit = st.radio("System", ["Metric (kg, cm)", "SI (kN, m)"])
    
    with st.sidebar.expander("🧱 Materials", expanded=True):
        if "Metric" in unit:
            fc = st.number_input("f'c (ksc)", value=240, step=10)
            fy = st.number_input("fy Main (ksc)", value=4000, step=100)
            fys = st.number_input("fy Stirrup (ksc)", value=2400, step=100)
            u_force, u_len = "kg", "m"
        else:
            fc = st.number_input("f'c (MPa)", value=25, step=5)
            fy = st.number_input("fy Main (MPa)", value=400, step=10)
            fys = st.number_input("fy Stirrup (MPa)", value=240, step=10)
            u_force, u_len = "kN", "m"
            
        st.markdown("---")
        c1, c2 = st.columns(2)
        db_main = c1.selectbox("Main DB", [12, 16, 20, 25, 28], index=1)
        db_stir = c2.selectbox("Stirrup RB", [6, 9, 10, 12], index=0)
    
    return {
        "unit": unit, "fc": fc, "fy": fy, "fys": fys,
        "db_main": db_main, "db_stirrup": db_stir,
        "u_force": u_force, "u_len": u_len
    }

def render_model_inputs(params):
    st.subheader("1️⃣ Geometry & Supports")
    
    # 1. จำนวนช่วงคาน
    n_spans = st.number_input("Number of Spans", min_value=1, max_value=10, value=2)
    spans = []
    supports = []
    
    # 2. เริ่มต้น Node แรก (Left Support)
    st.markdown(f"**📍 Start Node (Node 1)**")
    c1, c2 = st.columns([1, 3])
    with c1:
        sup_start = st.selectbox("Support Type", ["Pin", "Roller", "Fixed", "None"], key="sup_start")
    supports.append({"id": 0, "type": sup_start})
    
    # 3. ลูปแต่ละช่วงคาน + Node ถัดไป
    for i in range(int(n_spans)):
        st.markdown(f"---")
        st.markdown(f"**🚧 Span {i+1} ➔ Node {i+2}**")
        
        c_len, c_sup = st.columns([1, 1])
        with c_len:
            l = st.number_input(f"Span Length L{i+1} (m)", min_value=0.5, value=5.0, step=0.5, key=f"len_{i}")
        with c_sup:
            s = st.selectbox(f"Support @ Node {i+2} (Right)", ["Pin", "Roller", "Fixed", "None"], index=1, key=f"sup_{i}")
            
        spans.append(l)
        supports.append({"id": i+1, "type": s})

    return int(n_spans), spans, pd.DataFrame(supports), True

def render_loads(n_spans, spans, params):
    st.markdown("---")
    st.subheader("2️⃣ Loads Setup")
    u_f = params['u_force']
    
    if "loads" not in st.session_state: st.session_state.loads = []
    
    with st.expander("➕ Add Loads", expanded=True):
        c1, c2, c3 = st.columns([1, 1, 1])
        l_type = c1.selectbox("Type", ["Uniform Load", "Point Load"])
        span_idx = c2.selectbox("Span No.", range(1, int(n_spans)+1)) - 1
        
        if "Uniform" in l_type:
            val = c3.number_input(f"Magnitude w ({u_f}/m)", value=1000.0, step=100.0)
            if st.button("Add Uniform", use_container_width=True):
                st.session_state.loads.append({"type": "U", "span_idx": span_idx, "w": val})
        else:
            val = c3.number_input(f"Magnitude P ({u_f})", value=2000.0, step=100.0)
            loc = c3.number_input(f"Location x (m)", value=spans[span_idx]/2, max_value=float(spans[span_idx]))
            if st.button("Add Point", use_container_width=True):
                st.session_state.loads.append({"type": "P", "span_idx": span_idx, "P": val, "x": loc})

    # Show Loads
    if st.session_state.loads:
        for i, l in enumerate(st.session_state.loads):
            txt = f"Span {l['span_idx']+1}: "
            txt += f"Uniform w={l['w']}" if l['type']=='U' else f"Point P={l['P']} @ x={l['x']}m"
            col_txt, col_del = st.columns([4, 1])
            col_txt.info(txt)
            if col_del.button("❌", key=f"del_load_{i}"):
                st.session_state.loads.pop(i)
                st.rerun()
                
    return st.session_state.loads
