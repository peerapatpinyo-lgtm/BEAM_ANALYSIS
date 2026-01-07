import streamlit as st
import pandas as pd
import numpy as np

def render_sidebar_params():
    """
    Renders material and section properties in the sidebar.
    """
    st.subheader("1. Material Properties")
    E_gpa = st.number_input("Elastic Modulus (E) [GPa]", value=30.0, step=1.0)
    
    st.subheader("2. Beam Section")
    c1, c2 = st.columns(2)
    with c1:
        b = st.number_input("Width (b) [m]", value=0.25, step=0.05)
    with c2:
        h = st.number_input("Depth (h) [m]", value=0.50, step=0.05)
        
    I = (b * h**3) / 12
    
    return {
        "E": E_gpa * 1e9,
        "b": b,
        "h": h,
        "I": I
    }

def render_model_inputs_main(params):
    """
    Renders span configuration and supports setup in the MAIN area.
    """
    st.header("1. Model Geometry")
    
    # --- Part 1: Spans ---
    col_span, col_viz = st.columns([1, 2])
    
    with col_span:
        st.markdown("##### Span Configuration")
        n_spans = st.number_input("Number of Spans", min_value=1, max_value=10, value=2)
        
        spans = []
        with st.container():
            for i in range(n_spans):
                s = st.number_input(f"Span {i+1} Length (m)", min_value=0.1, value=5.0, key=f"span_{i}")
                spans.append(s)

    # --- Part 2: Supports ---
    with col_viz:
        st.markdown("##### Support Conditions")
        
        cum_dist = [0] + list(np.cumsum(spans))
        n_nodes = len(cum_dist)
        
        default_types = ["Pin"] + ["Roller"] * (n_nodes - 1)
        
        # [FIX] เปลี่ยนชื่อคอลัมน์จาก "node_id" เป็น "id" เพื่อให้ตรงกับ design_view.py
        sup_data = {
            "id": range(n_nodes), 
            "x": cum_dist,
            "type": default_types
        }
        df_sup_init = pd.DataFrame(sup_data)

        # แสดงตาราง
        edited_sup = st.data_editor(
            df_sup_init,
            column_config={
                "id": "Node", # แก้ config ให้ตรงกับ id
                "x": st.column_config.NumberColumn("Position (m)", disabled=True, format="%.2f"),
                "type": st.column_config.SelectboxColumn(
                    "Support Type",
                    options=["Pin", "Roller", "Fixed", "Free"],
                    required=True
                )
            },
            hide_index=True,
            use_container_width=True,
            key="sup_editor"
        )
        
    stable = True
    types = edited_sup['type'].tolist()
    fixed_count = types.count('Fixed')
    pin_count = types.count('Pin')
    roller_count = types.count('Roller')
    
    if fixed_count == 0 and pin_count == 0 and roller_count < 2:
        stable = False
    if types.count('Free') == len(types):
        stable = False

    return n_spans, spans, edited_sup, stable

def render_loads_main(n_spans, spans, params, sup_df):
    """
    Renders load input interface in the MAIN area.
    """
    st.header("2. Loads Definition")
    
    if "load_list" not in st.session_state:
        st.session_state.load_list = []
        
    with st.expander("➕ Add New Load", expanded=True):
        c1, c2, c3, c4, c5 = st.columns([1.5, 1, 1, 1, 1])
        
        with c1:
            l_type = st.selectbox("Load Type", ["Point Load (P)", "Uniform Load (U)"])
        
        with c2:
            span_idx = st.selectbox("On Span", range(n_spans), format_func=lambda x: f"Span {x+1}")
            
        with c3:
            if "Point" in l_type:
                lx = st.number_input("Pos x (m)", min_value=0.0, max_value=float(spans[span_idx]), value=float(spans[span_idx])/2)
                dist = 0.0
            else:
                lx = st.number_input("Start x (m)", min_value=0.0, max_value=float(spans[span_idx]), value=0.0)
                rem_len = spans[span_idx] - lx
                
        with c4:
            if "Point" in l_type:
                mag = st.number_input("Mag (kN)", value=10.0)
                dist = 0.0 # Assign default
            else:
                mag = st.number_input("Mag (kN/m)", value=10.0)
                dist = st.number_input("Length (m)", min_value=0.0, max_value=rem_len, value=rem_len)
                
        with c5:
            case = st.selectbox("Case", ["DL", "LL"])
            st.write("")
            add_btn = st.button("Add Load", type="secondary")

        if add_btn:
            new_load = {
                "id": len(st.session_state.load_list),
                "type": "P" if "Point" in l_type else "U",
                "span_index": span_idx,
                "x": lx,
                "mag": mag * 1000,
                "dist": dist,
                "case": case
            }
            st.session_state.load_list.append(new_load)
            st.rerun()

    if st.session_state.load_list:
        st.markdown("##### Current Loads List")
        
        df = pd.DataFrame(st.session_state.load_list)
        df_show = df.copy()
        
        df_show['Span'] = df_show['span_index'].apply(lambda x: f"Span {x+1}")
        df_show['Magnitude'] = df_show['mag'] / 1000.0
        df_show['Position'] = df_show.apply(lambda r: f"x={r['x']:.2f}" if r['type'] == 'P' else f"{r['x']:.2f} to {r['x']+r['dist']:.2f} m", axis=1)
        
        st.dataframe(
            df_show[['type', 'Span', 'Magnitude', 'Position', 'case']], 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "type": "Type",
                "Magnitude": "Mag (kN or kN/m)",
                "case": "Case"
            }
        )
        
        if st.button("🗑️ Clear All Loads"):
            st.session_state.load_list = []
            st.rerun()
            
        return pd.DataFrame(st.session_state.load_list)
    
    return None
