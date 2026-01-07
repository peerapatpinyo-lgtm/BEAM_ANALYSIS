import streamlit as st
import pandas as pd
import numpy as np

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Design Parameters")
        st.subheader("1. Material Properties")
        E = st.number_input("Elastic Modulus (E) [MPa]", value=200000.0, step=1000.0)
        
        st.subheader("2. Beam Section")
        b = st.number_input("Width (b) [m]", value=0.25, step=0.05)
        h = st.number_input("Height (h) [m]", value=0.50, step=0.05)
        I = (b * h**3) / 12
        
        st.subheader("3. Load Factors")
        c1, c2 = st.columns(2)
        gamma_dead = c1.number_input("Dead Load", value=1.4)
        gamma_live = c2.number_input("Live Load", value=1.7)
        
        return {
            "E": E * 1e6, # Pa
            "b": b,
            "h": h,
            "I": I,
            "gamma_dead": gamma_dead,
            "gamma_live": gamma_live
        }

def render_model_inputs(params):
    st.header("1. Model Geometry")
    col_span, col_viz = st.columns([1, 2])
    
    with col_span:
        n_spans = st.number_input("Number of Spans", min_value=1, max_value=10, value=2)
        spans = []
        for i in range(n_spans):
            s = st.number_input(f"Span {i+1} Length (m)", min_value=0.1, value=5.0, key=f"span_{i}")
            spans.append(s)
            
    with col_viz:
        st.markdown("##### Support Conditions")
        cum_dist = [0] + list(np.cumsum(spans))
        nodes = len(cum_dist)
        sup_data = []
        cols = st.columns(nodes)
        
        fixed_count = 0
        has_fixed = False
        
        for i, c in enumerate(cols):
            with c:
                st.caption(f"Node {i} (x={cum_dist[i]:.2f})")
                def_idx = 0 if i == 0 else 1 
                sType = st.selectbox("", ["Pin", "Roller", "Fixed", "None"], index=def_idx, key=f"sup_{i}", label_visibility="collapsed")
                sup_data.append({"id": i, "type": sType, "x": cum_dist[i]})
                if sType == "Fixed": has_fixed = True
                if sType != "None": fixed_count += 1

    stable = True
    if fixed_count < 2 and not has_fixed:
        stable = False
        
    return n_spans, spans, pd.DataFrame(sup_data), stable
    
def get_self_weight_load(b, h):
    # คอนกรีตเสริมเหล็กหนักประมาณ 2400 kg/m3 
    # แรงโน้มถ่วง g ≈ 9.81 m/s2 -> 2400 * 9.81 = 23,544 N/m3 (หรือประมาณ 24 kN/m3)
    gamma_concrete = 24.0 # kN/m3
    w_sw = b * h * gamma_concrete # kN/m
    return w_sw
    
def render_loads(n_spans, spans, params, sup_df):
    st.header("2. Loads Definition")
    
    if "load_list" not in st.session_state:
        st.session_state.load_list = []
        
    with st.expander("➕ Add New Load", expanded=True):
        c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
        with c1:
            l_type = st.selectbox("Type", ["Point Load (P)", "Uniform Load (U)", "Moment (M)"])
        with c2:
            span_idx = st.selectbox("Span Index", range(n_spans), format_func=lambda x: f"Span {x+1}")
        with c3:
            lx = st.number_input("Start Pos (x) [m]", min_value=0.0, max_value=spans[span_idx], value=0.0)
        with c4:
            mag = st.number_input("Mag (kN, kN/m)", value=10.0)
        with c5:
            dist = 0.0
            if l_type == "Uniform Load (U)":
                rem_len = spans[span_idx] - lx
                dist = st.number_input("Length (m)", min_value=0.0, max_value=rem_len, value=rem_len)
            case = st.selectbox("Case", ["DL", "LL"])

        if st.button("Add Load"):
            new_load = {
                "id": len(st.session_state.load_list),
                "type": l_type[0], 
                "span_index": span_idx,
                "x": lx,
                "mag": mag * 1000, # Store as N
                "dist": dist,
                "case": case
            }
            st.session_state.load_list.append(new_load)
            
    if st.session_state.load_list:
        df = pd.DataFrame(st.session_state.load_list)
        df_show = df.copy()
        df_show['mag'] = df_show['mag'] / 1000.0
        df_show['Span'] = df_show['span_index'] + 1
        df_show['Type'] = df_show['type'].map({'P': 'Point', 'U': 'Uniform', 'M': 'Moment'})
        df_show['End x'] = df_show.apply(lambda r: r['x'] + r['dist'] if r['type'] == 'U' else '-', axis=1)
        
        st.dataframe(df_show[['Type', 'Span', 'x', 'End x', 'mag', 'case']], use_container_width=True)
        
        if st.button("Clear All Loads"):
            st.session_state.load_list = []
            st.rerun()
        return df
    return None
