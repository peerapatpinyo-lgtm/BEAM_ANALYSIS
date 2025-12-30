import streamlit as st
import pandas as pd
import numpy as np

def render_sidebar():
    st.sidebar.header("⚙️ Design Parameters")
    units = st.sidebar.selectbox("Units System", ["Metric (kg, m)", "SI (kN, m)"])
    if "kg" in units:
        u_force, u_len = "kg", "m"
    else:
        u_force, u_len = "kN", "m"
        
    st.sidebar.subheader("Materials (Concrete)")
    fc = st.sidebar.number_input("fc' (ksc/MPa)", value=240)
    fy = st.sidebar.number_input("fy (ksc/MPa)", value=4000)
    
    return {'u_force': u_force, 'u_len': u_len, 'fc': fc, 'fy': fy}

def render_model_inputs(params):
    c1, c2 = st.columns([1, 2])
    with c1:
        n_spans = st.number_input("Number of Spans", min_value=1, max_value=5, value=2)
    
    spans = []
    sup_types = []
    
    # Dynamic Columns for Spans
    cols = st.columns(n_spans)
    for i, col in enumerate(cols):
        with col:
            s = st.number_input(f"L{i+1} ({params['u_len']})", min_value=1.0, value=5.0, key=f"s_{i}")
            spans.append(s)
            
    # Supports
    st.markdown("**Supports**")
    sup_cols = st.columns(n_spans + 1)
    sup_config = []
    
    options = ['Pin', 'Roller', 'Fixed', 'None']
    defaults = ['Pin'] + ['Roller']*(n_spans-1) + ['Roller']
    
    for i, col in enumerate(sup_cols):
        with col:
            t = st.selectbox(f"S{i}", options, index=options.index(defaults[min(i, len(defaults)-1)]), key=f"sup_{i}")
            if t != 'None':
                sup_config.append({'id': i, 'type': t})
    
    sup_df = pd.DataFrame(sup_config)
    
    # Stability Check (Basic)
    stable = True
    if len(sup_df) < 2 and not any(s['type'] == 'Fixed' for s in sup_config):
        stable = False
        
    return n_spans, spans, sup_df, stable

def render_loads(n_spans, spans, params):
    if 'loads' not in st.session_state:
        st.session_state['loads'] = []

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        l_type = st.selectbox("Load Type", ["Point Load (P)", "Uniform Load (w)"])
    with c2:
        span_idx = st.selectbox("Span No.", range(1, n_spans+1)) - 1
    
    if l_type == "Point Load (P)":
        with c3:
            mag = st.number_input(f"P ({params['u_force']})", value=1000.0)
        loc = st.slider(f"Position from Left ({params['u_len']})", 0.0, spans[span_idx], spans[span_idx]/2)
        if st.button("Add Point Load"):
            st.session_state['loads'].append({'type': 'P', 'span_idx': span_idx, 'P': mag, 'x': loc})
            
    else:
        with c3:
            mag = st.number_input(f"w ({params['u_force']}/{params['u_len']})", value=500.0)
        st.caption(f"Applied to full length of Span {span_idx+1}")
        if st.button("Add Uniform Load"):
            st.session_state['loads'].append({'type': 'U', 'span_idx': span_idx, 'w': mag})
            
    # Display Current Loads
    if st.session_state['loads']:
        st.markdown("---")
        st.markdown("**Applied Loads:**")
        for i, l in enumerate(st.session_state['loads']):
            cc1, cc2 = st.columns([4, 1])
            with cc1:
                if l['type'] == 'P':
                    st.text(f"Span {l['span_idx']+1}: Point Load {l['P']} at {l['x']}")
                else:
                    st.text(f"Span {l['span_idx']+1}: UDL {l['w']}")
            with cc2:
                if st.button("❌", key=f"del_{i}"):
                    st.session_state['loads'].pop(i)
                    st.rerun()
                    
    return st.session_state['loads']
