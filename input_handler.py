import streamlit as st
import pandas as pd

def render_sidebar():
    st.sidebar.header("⚙️ Design Standards")
    unit = st.sidebar.radio("Units", ["Metric (kg, cm)", "SI (kN, m)"])
    
    st.sidebar.markdown("---")
    st.sidebar.header("🧱 Material Properties")
    
    if "Metric" in unit:
        fc = st.sidebar.number_input("f'c (ksc)", value=240, step=10)
        fy = st.sidebar.number_input("fy (Main) (ksc)", value=4000, step=100)
        fys = st.sidebar.number_input("fy (Stirrup) (ksc)", value=2400, step=100)
        u_force, u_len = "kg", "m"
    else:
        fc = st.sidebar.number_input("f'c (MPa)", value=25, step=5)
        fy = st.sidebar.number_input("fy (Main) (MPa)", value=400, step=10)
        fys = st.sidebar.number_input("fy (Stirrup) (MPa)", value=240, step=10)
        u_force, u_len = "kN", "m"
        
    db_main = st.sidebar.selectbox("Main Bar DB", [12, 16, 20, 25, 28, 32], index=1)
    db_stir = st.sidebar.selectbox("Stirrup Bar RB/DB", [6, 9, 10, 12], index=0)
    
    return {
        "unit": unit, "fc": fc, "fy": fy, "fys": fys,
        "db_main": db_main, "db_stirrup": db_stir,
        "u_force": u_force, "u_len": u_len
    }

def render_geometry():
    st.subheader("1️⃣ Beam Geometry & Supports")
    n_spans = st.number_input("Number of Spans", min_value=1, max_value=10, value=2)
    
    spans = []
    supports = []
    
    # Header row
    c1, c2 = st.columns([1, 1])
    c1.markdown("**Span Length (m)**")
    c2.markdown("**Support Type (Right Side)**")
    
    # First support (Leftmost)
    st.markdown("**Leftmost Support (Start)**")
    first_sup = st.selectbox("Type", ["Pin", "Roller", "Fixed", "None"], key="sup_0")
    supports.append({"id": 0, "type": first_sup})
    
    for i in range(int(n_spans)):
        c1, c2 = st.columns([1, 1])
        l = c1.number_input(f"L{i+1}", min_value=0.1, value=5.0, key=f"span_len_{i}")
        s = c2.selectbox(f"S{i+1}", ["Pin", "Roller", "Fixed", "None"], index=1, key=f"sup_{i+1}")
        
        spans.append(l)
        supports.append({"id": i+1, "type": s})
        
    return int(n_spans), spans, pd.DataFrame(supports), True

def render_loads(n_spans, spans, params):
    st.subheader("2️⃣ Defined Loads")
    u_f = params['u_force']
    
    if "loads" not in st.session_state:
        st.session_state.loads = []

    with st.expander("➕ Add New Load", expanded=True):
        c1, c2, c3, c4 = st.columns([1, 1, 1, 0.5])
        l_type = c1.selectbox("Type", ["Uniform (U)", "Point (P)"])
        span_idx = c2.selectbox("Span Index", range(1, n_spans+1)) - 1
        
        val = 0
        loc = 0
        
        if "Uniform" in l_type:
            val = c3.number_input(f"w ({u_f}/m)", value=1000.0)
            if c4.button("Add", key="add_u"):
                st.session_state.loads.append({"type": "U", "span_idx": span_idx, "w": val})
        else:
            val = c3.number_input(f"P ({u_f})", value=1000.0)
            loc = c3.number_input(f"Distance x from left (m)", min_value=0.0, max_value=spans[span_idx], value=spans[span_idx]/2)
            if c4.button("Add", key="add_p"):
                st.session_state.loads.append({"type": "P", "span_idx": span_idx, "P": val, "x": loc})

    # Display list
    if st.session_state.loads:
        for i, l in enumerate(st.session_state.loads):
            txt = f"Span {l['span_idx']+1}: "
            if l['type'] == 'U': txt += f"Uniform w={l['w']} {u_f}/m"
            else: txt += f"Point P={l['P']} {u_f} @ x={l['x']}m"
            
            c_txt, c_del = st.columns([4, 1])
            c_txt.text(txt)
            if c_del.button("❌", key=f"del_{i}"):
                st.session_state.loads.pop(i)
                st.rerun()
                
    return st.session_state.loads

# --- NEW FUNCTION: Moved Beam Size Here ---
def render_section_inputs(n_spans):
    st.subheader("3️⃣ Section Design Properties")
    st.info("Define the beam cross-section size for each span (used for Analysis stiffness & Design).")
    
    span_props = []
    cols = st.columns(n_spans)
    
    for i in range(n_spans):
        with cols[i]:
            st.markdown(f"**Span {i+1}**")
            b = st.number_input(f"b (cm)", min_value=10.0, value=30.0, key=f"b_{i}")
            h = st.number_input(f"h (cm)", min_value=20.0, value=60.0, key=f"h_{i}")
            cv = st.number_input(f"Cover (cm)", min_value=2.0, value=3.0, key=f"cv_{i}")
            span_props.append({"b": b, "h": h, "cv": cv})
            
    return span_props
