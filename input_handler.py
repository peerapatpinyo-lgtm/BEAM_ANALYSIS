import streamlit as st
import pandas as pd

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Design Parameters")
        
        st.subheader("1. Material & Section")
        E = st.number_input("Elastic Modulus (E)", value=2e11, format="%e", help="Pa (N/m^2)")
        I = st.number_input("Moment of Inertia (I)", value=5e-5, format="%e", help="m^4")
        
        st.subheader("2. Load Factors (Strength)")
        gamma_dead = st.number_input("Dead Load Factor (DL)", value=1.4, step=0.1)
        gamma_live = st.number_input("Live Load Factor (LL)", value=1.7, step=0.1)
        
        st.subheader("3. Units")
        u_force = st.selectbox("Force Unit", ["kN", "N", "kg"])
        u_len = st.text_input("Length Unit", "m")
        
        return {
            "E": E, "I": I, 
            "gamma_dead": gamma_dead, "gamma_live": gamma_live,
            "u_force": u_force, "u_len": u_len
        }

def render_model_inputs(params):
    st.subheader("1. Geometry & Supports")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        n_spans = st.number_input("Number of Spans", min_value=1, max_value=10, value=2)
    
    spans = []
    with col2:
        cols = st.columns(n_spans)
        for i in range(n_spans):
            spans.append(cols[i].number_input(f"L{i+1}", min_value=1.0, value=5.0, key=f"span_{i}"))

    # Support Config
    st.markdown("##### Support Configuration")
    sup_data = []
    num_nodes = n_spans + 1
    
    cols_sup = st.columns(num_nodes)
    possible_sups = ["Pin", "Roller", "Fixed", "None"]
    
    # Default supports: Pin at start, Roller at others
    defaults = ["Pin"] + ["Roller"] * (num_nodes - 1)
    
    for i in range(num_nodes):
        # Handle index out of range for defaults if user increases spans
        def_val = defaults[i] if i < len(defaults) else "Roller"
        
        s_type = cols_sup[i].selectbox(f"Node {i+1}", possible_sups, index=possible_sups.index(def_val), key=f"sup_{i}")
        if s_type != "None":
            sup_data.append({"id": i, "type": s_type})
    
    # Stability Check (Simple)
    sup_df = pd.DataFrame(sup_data)
    stable = True
    if len(sup_data) < 2:
        if len(sup_data) == 1 and sup_data[0]['type'] == 'Fixed':
            pass
        else:
            stable = False
            
    return n_spans, spans, sup_df, stable

def render_loads(n_spans, spans, params, sup_df):
    st.subheader("2. Applied Loads")
    
    if "load_list" not in st.session_state:
        st.session_state.load_list = []
        
    # Input Form
    with st.form("add_load_form"):
        c1, c2, c3, c4, c5 = st.columns([1, 1.2, 1, 1, 1])
        
        span_choice = c1.selectbox("Span No.", options=list(range(1, n_spans+1)))
        l_type = c2.selectbox("Type", ["Point (P)", "Uniform (w)", "Moment (M)"])
        l_case = c3.selectbox("Case", ["DL (Dead)", "LL (Live)"]) 
        mag = c4.number_input(f"Mag ({params['u_force']})", value=1000.0)
        
        # Logic for Position/Distance
        current_span_len = spans[span_choice-1]
        x_loc = c5.number_input("Dist x (m)", value=current_span_len/2, max_value=float(current_span_len))
        
        # New Field: Load Length (Only for UDL)
        # We add it generally, but only use it if UDL
        dist_load = 0.0
        if "Uniform" in l_type:
             st.caption(f"Load extends from x={x_loc} to end?")
             # For simplicity in this UI, let's assume UDL covers the whole span or starts at x
             # But solver needs 'dist'. Let's default 'dist' to rest of span if UDL
             dist_load = current_span_len - x_loc

        submitted = st.form_submit_button("➕ Add Load")
        
        if submitted:
            # Map type to code
            type_code = 'P'
            dist_val = 0.0
            
            if "Uniform" in l_type: 
                type_code = 'U'
                dist_val = dist_load # Use the calculated rest of span or input
            elif "Moment" in l_type: 
                type_code = 'M'
            
            st.session_state.load_list.append({
                "span_index": span_choice - 1, # <--- แก้ไขชื่อตัวแปรตรงนี้ (จาก span_idx เป็น span_index)
                "type": type_code,
                "case": "DL" if "DL" in l_case else "LL",
                "mag": mag,
                "x": x_loc,
                "dist": dist_val # <--- เพิ่มตัวแปรนี้ เพื่อให้ solver ไม่ error
            })
            
    # Display Loads
    if st.session_state.load_list:
        loads_df = pd.DataFrame(st.session_state.load_list)
        
        for i, l in enumerate(st.session_state.load_list):
            l_text = f"Span {l['span_index']+1}: {l['type']} = {l['mag']} ({l['case']}) @ x={l['x']}"
            if l['type'] == 'U':
                l_text += f" (Len={l['dist']:.2f})"
                
            c_del_1, c_del_2 = st.columns([8, 1])
            c_del_1.text(l_text)
            if c_del_2.button("❌", key=f"del_{i}"):
                st.session_state.load_list.pop(i)
                st.rerun()
                
        return loads_df

    return None
