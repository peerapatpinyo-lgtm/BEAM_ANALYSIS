import streamlit as st
import pandas as pd

def render_sidebar():
    st.markdown("### 🧱 Materials")
    fc = st.number_input("Concrete fc' (ksc)", 240.0, step=10.0)
    fy = st.number_input("Rebar fy (ksc)", 4000.0, step=100.0)
    return {"fc": fc, "fy": fy, "u_len": "m", "u_force": "kg"}

def render_model_inputs(params):
    st.markdown("##### 1. Geometry Setup")
    n_spans = st.number_input("Total Spans", 1, 10, 2)
    
    # Spans
    st.caption("Define Span Lengths (m)")
    cols = st.columns(n_spans)
    spans = []
    for i in range(n_spans):
        val = cols[i].number_input(f"L{i+1}", 0.5, value=4.0, step=0.5, key=f"s_{i}")
        spans.append(val)
        
    # Supports
    st.caption("Define Supports")
    n_nodes = n_spans + 1
    sup_cols = st.columns(n_nodes)
    sup_data = []
    
    # Smart defaults
    defaults = ["Pin"] + ["Roller"]*(n_nodes-2) + ["Roller"]
    
    for i in range(n_nodes):
        stype = sup_cols[i].selectbox(f"Sup #{i+1}", ["Pin", "Fixed", "Roller", "None"], 
                                      index=["Pin", "Fixed", "Roller", "None"].index(defaults[i]), 
                                      key=f"sup_{i}", label_visibility="collapsed")
        if stype != "None": sup_data.append({"id": i, "type": stype})
    
    st.caption(f"Supports at: " + ", ".join([f"#{d['id']+1}" for d in sup_data]))
            
    return n_spans, spans, pd.DataFrame(sup_data), (len(sup_data) >= 2)

def render_loads(n_spans, spans, params):
    if 'loads' not in st.session_state: st.session_state['loads'] = []
    
    st.markdown("##### 2. Loading Conditions")
    
    # Container for "Add Load" Tool
    with st.container(border=True):
        st.markdown("**🛠️ Add New Load**")
        
        # 1. Select Span
        c_span, c_type = st.columns([1, 1.5])
        with c_span:
            span_idx = st.selectbox("Apply to Span", range(1, n_spans+1), format_func=lambda x: f"Span {x} (L={spans[x-1]}m)") - 1
        with c_type:
            l_type = st.radio("Load Type", ["Uniform (kg/m)", "Point (kg)"], horizontal=True)
            
        # 2. Magnitude & Location
        c_mag, c_loc, c_btn = st.columns([1.5, 1.5, 1])
        
        with c_mag:
            mag = st.number_input("Magnitude", value=1000.0, step=100.0, format="%.2f")
        
        with c_loc:
            loc = 0.0
            if "Point" in l_type:
                loc = st.number_input(f"Distance x (0-{spans[span_idx]}m)", 
                                      min_value=0.0, max_value=spans[span_idx], value=spans[span_idx]/2.0)
            else:
                st.text_input("Location", "Full Span", disabled=True)
        
        with c_btn:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True) # Spacer
            if st.button("➕ Add Load", type="secondary", use_container_width=True):
                st.session_state['loads'].append({
                    "span_idx": span_idx,
                    "type": "U" if "Uniform" in l_type else "P",
                    "w": mag if "Uniform" in l_type else 0,
                    "P": mag if "Point" in l_type else 0,
                    "x": loc
                })
                st.rerun()

    # List of Active Loads
    if st.session_state['loads']:
        st.markdown("**📋 Active Loads List**")
        for i, l in enumerate(st.session_state['loads']):
            with st.container():
                c1, c2, c3 = st.columns([1, 4, 1])
                c1.markdown(f"**#{i+1}**")
                
                desc = f"**Span {l['span_idx']+1}**: "
                if l['type'] == 'U':
                    desc += f"Uniform Load <span style='color:#DC2626'>w = {l['w']} kg/m</span>"
                else:
                    desc += f"Point Load <span style='color:#DC2626'>P = {l['P']} kg</span> @ {l['x']} m"
                
                c2.markdown(desc, unsafe_allow_html=True)
                
                if c3.button("🗑️", key=f"del_{i}"):
                    st.session_state['loads'].pop(i)
                    st.rerun()
                st.divider()

    return st.session_state['loads']
