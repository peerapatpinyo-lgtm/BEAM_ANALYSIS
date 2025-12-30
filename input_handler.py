import streamlit as st
import pandas as pd

def render_sidebar():
    st.sidebar.markdown("## ⚙️ Settings")
    
    # 1. Unit System
    with st.sidebar.expander("📏 Units & Standards", expanded=True):
        unit = st.radio("System", ["Metric (kg, cm)", "SI (kN, m)"])
    
    # 2. Material Properties
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
    # ใช้ Container เพื่อจัดกลุ่ม Input ให้สวยงาม
    with st.container():
        st.subheader("1️⃣ Geometry & Supports")
        
        c_main, c_dummy = st.columns([2, 1])
        with c_main:
            n_spans = st.number_input("Number of Spans", min_value=1, max_value=10, value=2)

        spans = []
        supports = []
        
        # Table Header-like layout
        cols = st.columns([0.5, 1.5, 1.5]) 
        cols[0].markdown("##### No.")
        cols[1].markdown("##### Length (m)")
        cols[2].markdown("##### Support (Right)")

        # First Support
        supports.append({"id": 0, "type": "Pin"}) # Default start with Pin for simplicity or make editable if needed
        
        for i in range(int(n_spans)):
            c = st.columns([0.5, 1.5, 1.5])
            c[0].markdown(f"**Span {i+1}**")
            l = c[1].number_input(f"L{i+1}", min_value=0.5, value=5.0, step=0.5, key=f"len_{i}", label_visibility="collapsed")
            s = c[2].selectbox(f"S{i+1}", ["Pin", "Roller", "Fixed", "None"], index=1, key=f"sup_{i}", label_visibility="collapsed")
            
            spans.append(l)
            supports.append({"id": i+1, "type": s})
            
        # Leftmost support options (Minimal UI)
        st.caption("Start Node (Left): Pin Support (Default)")

    st.markdown("---")

    with st.container():
        st.subheader("2️⃣ Loads Setup")
        u_f = params['u_force']
        
        if "loads" not in st.session_state: st.session_state.loads = []
        
        # Add Load UI
        with st.expander("➕ Add Loads", expanded=True):
            c1, c2, c3 = st.columns([1, 1, 1])
            l_type = c1.selectbox("Type", ["Uniform Load", "Point Load"])
            span_idx = c2.selectbox("Span No.", range(1, int(n_spans)+1)) - 1
            
            if "Uniform" in l_type:
                val = c3.number_input(f"Magnitude ({u_f}/m)", value=1000.0, step=100.0)
                if st.button("Add Uniform Load", use_container_width=True):
                    st.session_state.loads.append({"type": "U", "span_idx": span_idx, "w": val})
            else:
                val = c3.number_input(f"Magnitude ({u_f})", value=2000.0, step=100.0)
                loc = c3.number_input(f"Location x (m)", value=spans[span_idx]/2, max_value=float(spans[span_idx]))
                if st.button("Add Point Load", use_container_width=True):
                    st.session_state.loads.append({"type": "P", "span_idx": span_idx, "P": val, "x": loc})

        # Load List (Chips style)
        if st.session_state.loads:
            st.markdown("###### Current Loads:")
            for i, l in enumerate(st.session_state.loads):
                txt = f"Span {l['span_idx']+1}: "
                txt += f"Uniform {l['w']}" if l['type']=='U' else f"Point {l['P']} @ {l['x']}m"
                
                col_txt, col_del = st.columns([4, 1])
                col_txt.info(txt)
                if col_del.button("❌", key=f"del_load_{i}"):
                    st.session_state.loads.pop(i)
                    st.rerun()
        else:
            st.warning("No loads defined yet.")

    return int(n_spans), spans, pd.DataFrame(supports), True
