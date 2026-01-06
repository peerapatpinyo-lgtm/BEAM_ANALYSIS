import streamlit as st
import pandas as pd
import numpy as np

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Engineering Parameters")
        st.subheader("1. Materials")
        fc = st.number_input("f'c (Concrete Strength) [MPa]", value=28.0)
        fy = st.number_input("fy (Steel Yield) [MPa]", value=400.0)
        E = st.number_input("E (Elastic Modulus) [MPa]", value=25000.0) * 1e6 # Pa
        
        st.subheader("2. Beam Section")
        b = st.number_input("Width (b) [m]", value=0.25)
        h = st.number_input("Height (h) [m]", value=0.50)
        cover = st.number_input("Covering [mm]", value=40.0)
        db_main = st.selectbox("Main Bar Size", [12, 16, 20, 25, 28], index=1)
        I = (b * h**3) / 12
        
        st.subheader("3. Load Factors")
        c1, c2 = st.columns(2)
        g_d = c1.number_input("Gamma DL", value=1.4)
        g_l = c2.number_input("Gamma LL", value=1.7)
        
        return {"fc": fc, "fy": fy, "E": E, "b": b, "h": h, "I": I, 
                "cover": cover, "db_main": db_main, "gamma_dead": g_d, "gamma_live": g_l}

def render_model_inputs(params):
    st.header("1. Model Geometry")
    n_spans = st.number_input("Number of Spans", min_value=1, max_value=10, value=2)
    spans = [st.number_input(f"Span {i+1} Length (m)", value=5.0, key=f"s_{i}") for i in range(n_spans)]
    
    st.markdown("##### Support Conditions")
    cum_dist = [0] + list(np.cumsum(spans))
    sup_data = []
    cols = st.columns(len(cum_dist))
    fixed_count = 0
    for i, c in enumerate(cols):
        with c:
            st.caption(f"Node {i} (x={cum_dist[i]})")
            stype = st.selectbox("", ["Pin", "Roller", "Fixed", "None"], index=(0 if i==0 else 1), key=f"sup_{i}")
            sup_data.append({"id": i, "type": stype, "x": cum_dist[i]})
            if stype != "None": fixed_count += 1
            
    return n_spans, spans, pd.DataFrame(sup_data), (fixed_count >= 2)

def render_loads(n_spans, spans, params, sup_df):
    st.header("2. Loads Definition")
    if "load_list" not in st.session_state: st.session_state.load_list = []
    
    with st.expander("➕ Add New Load", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        l_type = c1.selectbox("Type", ["Point Load (P)", "Uniform Load (U)", "Moment (M)"])
        s_idx = c2.selectbox("Span", range(n_spans), format_func=lambda x: f"Span {x+1}")
        lx = c3.number_input("Start x (m)", min_value=0.0, max_value=spans[s_idx], value=0.0)
        mag = c4.number_input("Magnitude", value=10.0)
        case = c5.selectbox("Case", ["DL", "LL"])
        dist = 0.0
        if "Uniform" in l_type:
            dist = st.number_input("Length (m)", min_value=0.1, max_value=spans[s_idx]-lx, value=spans[s_idx]-lx)
            
        if st.button("Add Load"):
            st.session_state.load_list.append({"id": len(st.session_state.load_list), "type": l_type[0], 
                                              "span_index": s_idx, "x": lx, "mag": mag*1000, "dist": dist, "case": case})
    
    if st.session_state.load_list:
        df = pd.DataFrame(st.session_state.load_list)
        st.dataframe(df[["type", "span_index", "x", "mag", "case"]], use_container_width=True)
        if st.button("Clear All"): st.session_state.load_list = []; st.rerun()
        return df
    return None
