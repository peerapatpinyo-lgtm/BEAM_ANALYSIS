import streamlit as st
import pandas as pd

def render_sidebar():
    st.markdown("### 🧱 Material Properties")
    fc = st.number_input("Concrete fc' (ksc)", 240.0, step=10.0)
    fy = st.number_input("Rebar fy (ksc)", 4000.0, step=100.0)
    return {"fc": fc, "fy": fy, "u_len": "m", "u_force": "kg"}

def render_model_inputs(params):
    n_spans = st.number_input("Number of Spans", 1, 10, 2)
    
    st.markdown("**Span Lengths (m)**")
    cols = st.columns(n_spans)
    spans = []
    for i in range(n_spans):
        val = cols[i].number_input(f"L{i+1}", 0.5, value=4.0, step=0.5, key=f"s_{i}")
        spans.append(val)
        
    st.markdown("**Supports**")
    n_nodes = n_spans + 1
    sup_cols = st.columns(n_nodes)
    sup_data = []
    for i in range(n_nodes):
        # Default smart selection
        idx = 1 if i > 0 and i < n_nodes-1 else (0 if i==0 else 1) 
        stype = sup_cols[i].selectbox(f"S{i+1}", ["Pin", "Fixed", "Roller", "None"], index=idx, key=f"sup_{i}")
        if stype != "None": sup_data.append({"id": i, "type": stype})
            
    return n_spans, spans, pd.DataFrame(sup_data), (len(sup_data) >= 2)

def render_loads(n_spans, spans, params):
    if 'loads' not in st.session_state: st.session_state['loads'] = []
    
    # Improved Load Input Layout
    with st.container(border=True):
        st.markdown("###### ➕ Add New Load")
        
        # Row 1: Location
        c1, c2 = st.columns([1, 1])
        span_idx = c1.selectbox("Apply to Span #", range(1, n_spans+1)) - 1
        l_type = c2.selectbox("Load Type", ["Uniform Load (U)", "Point Load (P)"])
        
        # Row 2: Magnitude & Position
        c3, c4, c5 = st.columns([2, 2, 1])
        mag = c3.number_input("Magnitude (kg or kg/m)", value=1000.0, step=100.0)
        
        loc = 0.0
        if "Point" in l_type:
            loc = c4.number_input(f"Distance x from left (m)", 0.0, spans[span_idx], spans[span_idx]/2)
        else:
            c4.info("Applied to full span")
            
        if c5.button("Add", use_container_width=True):
            st.session_state['loads'].append({
                "span_idx": span_idx,
                "type": "U" if "Uniform" in l_type else "P",
                "w": mag if "Uniform" in l_type else 0,
                "P": mag if "Point" in l_type else 0,
                "x": loc
            })
            
    # List Loads
    if st.session_state['loads']:
        st.markdown("---")
        for i, l in enumerate(st.session_state['loads']):
            type_str = "Uniform" if l['type']=='U' else "Point"
            val_str = f"{l['w']} kg/m" if l['type']=='U' else f"{l['P']} kg @ {l['x']}m"
            
            c_txt, c_del = st.columns([4, 1])
            c_txt.text(f"{i+1}. Span {l['span_idx']+1}: {type_str} ({val_str})")
            if c_del.button("❌", key=f"del_{i}"):
                st.session_state['loads'].pop(i)
                st.rerun()

    return st.session_state['loads']
