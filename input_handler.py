
import streamlit as st
import pandas as pd
import numpy as np

def render_all_sidebar_inputs():
    """
    Renders ALL inputs (Materials, Geometry, Supports, Loads) inside the Sidebar.
    """
    params = {}
    
    # --- 1. Material & Section ---
    st.subheader("🧱 1. Material & Section")
    c1, c2 = st.columns(2)
    with c1:
        E_gpa = st.number_input("E (GPa)", 10.0, 200.0, 30.0, 1.0)
    with c2:
        fc = st.number_input("fc' (MPa)", 10, 50, 24)
    
    c3, c4 = st.columns(2)
    with c3:
        b = st.number_input("b (m)", 0.1, 1.0, 0.25, 0.05)
    with c4:
        h = st.number_input("h (m)", 0.1, 2.0, 0.50, 0.05)
    
    fy = st.number_input("fy (Main) MPa", 240, 500, 400, step=100)
    
    params['E'] = E_gpa * 1e9
    params['b'] = b
    params['h'] = h
    params['fc'] = fc
    params['fy'] = fy
    params['I'] = (b * h**3) / 12

    st.divider()

    # --- 2. Geometry (Spans) ---
    st.subheader("📏 2. Geometry")
    n_spans = st.number_input("Num Spans", 1, 10, 2)
    spans = []
    
    # Use Expander to save space
    with st.expander("Edit Span Lengths", expanded=True):
        for i in range(n_spans):
            val = st.number_input(f"Span {i+1} (m)", 0.5, 50.0, 5.0, 0.5, key=f"span_{i}")
            spans.append(val)

    # --- 3. Supports ---
    st.subheader("pk 3. Supports")
    cum_dist = [0] + list(np.cumsum(spans))
    n_nodes = len(cum_dist)
    
    # Default supports
    if 'sup_types' not in st.session_state:
        st.session_state.sup_types = ["Pin"] + ["Roller"] * (n_nodes - 1)
        
    # Re-sync if node count changes
    if len(st.session_state.sup_types) != n_nodes:
         st.session_state.sup_types = ["Pin"] + ["Roller"] * (n_nodes - 1)

    sup_data = []
    with st.expander("Edit Supports", expanded=False):
        for i in range(n_nodes):
            st.caption(f"Node {i} @ {cum_dist[i]:.2f}m")
            sType = st.selectbox(
                f"Type", ["Pin", "Roller", "Fixed", "Free"], 
                index=["Pin", "Roller", "Fixed", "Free"].index(st.session_state.sup_types[i]) if i < len(st.session_state.sup_types) else 1,
                key=f"sup_sel_{i}", label_visibility="collapsed"
            )
            sup_data.append({"id": i, "x": cum_dist[i], "type": sType})

    sup_df = pd.DataFrame(sup_data)
    
    # Check Stability
    types = [s['type'] for s in sup_data]
    stable = True
    if types.count('Fixed') == 0 and types.count('Pin') == 0 and types.count('Roller') < 2: stable = False
    if types.count('Free') == len(types): stable = False

    st.divider()

    # --- 4. Loads ---
    st.subheader("⬇️ 4. Loads")
    
    if "load_list" not in st.session_state:
        st.session_state.load_list = []

    with st.expander("➕ Add / Manage Loads", expanded=True):
        # Input Form
        l_type = st.selectbox("Type", ["Point (P)", "Uniform (w)"], key="l_type_in")
        span_idx = st.selectbox("Span", range(n_spans), format_func=lambda x: f"Span {x+1}", key="l_span_in")
        
        if "Point" in l_type:
            pos = st.number_input("x (m)", 0.0, spans[span_idx], spans[span_idx]/2, key="l_pos_in")
            mag = st.number_input("P (kN)", value=10.0, key="l_mag_in")
            dist = 0.0
        else:
            pos = st.number_input("Start x (m)", 0.0, spans[span_idx], 0.0, key="l_pos_in")
            rem = spans[span_idx] - pos
            mag = st.number_input("w (kN/m)", value=10.0, key="l_mag_in")
            dist = st.number_input("Len (m)", 0.0, rem, rem, key="l_dist_in")
            
        case = st.selectbox("Case", ["DL", "LL"], key="l_case_in")
        
        if st.button("Add Load"):
            st.session_state.load_list.append({
                "id": len(st.session_state.load_list),
                "type": "P" if "Point" in l_type else "U",
                "span_index": span_idx,
                "x": pos,
                "mag": mag * 1000, # N
                "dist": dist,
                "case": case
            })
            st.rerun()
            
        # Mini Table of Loads
        if st.session_state.load_list:
            st.markdown("---")
            for i, l in enumerate(st.session_state.load_list):
                txt = f"**{l['case']}** "
                txt += f"P={l['mag']/1000}kN @ {l['x']}m" if l['type']=='P' else f"w={l['mag']/1000}kN/m ({l['x']}-{l['x']+l['dist']}m)"
                col_del1, col_del2 = st.columns([4, 1])
                with col_del1: st.caption(f"{i+1}. Span {l['span_index']+1}: {txt}")
                with col_del2: 
                    if st.button("x", key=f"del_{i}"):
                        st.session_state.load_list.pop(i)
                        st.rerun()

    return params, n_spans, spans, sup_df, pd.DataFrame(st.session_state.load_list), stable
