import streamlit as st
import pandas as pd
import numpy as np

def check_stability(supports_df):
    if supports_df.empty: return False, "No supports defined."
    types = supports_df[supports_df['type'] != 'None']['type'].tolist()
    if not types: return False, "No active supports."
    if len(types) < 2 and "Fixed" not in types: return False, "Unstable Structure (Mechanism)."
    return True, "Stable"

def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Project Settings")
        unit_sys = st.radio("Unit System", ["Metric (kg, m)", "SI (kN, m)"])
        u_force = "kg" if "Metric" in unit_sys else "kN"
        u_stress = "ksc" if "Metric" in unit_sys else "MPa"
        
        st.markdown("### 🧱 Material Properties")
        with st.expander("Concrete & Steel", expanded=True):
            fc = st.number_input(f"fc' ({u_stress})", value=240.0)
            fy = st.number_input(f"fy Main ({u_stress})", value=4000.0)
            fys = st.number_input(f"fy Stirrup ({u_stress})", value=2400.0)
        
        with st.expander("Section & Rebar Selection", expanded=True):
            b = st.number_input("Width b (cm)", value=25.0)
            h = st.number_input("Depth h (cm)", value=50.0)
            cover = st.number_input("Covering (cm)", value=3.0)
            # --- เอากลับมาตามคำขอ ---
            st.markdown("**Preferred Bar Sizes:**")
            db_main = st.selectbox("Main Bar (mm)", [12, 16, 20, 25, 28, 32], index=1)
            db_stirrup = st.selectbox("Stirrup (mm)", [6, 9, 12], index=0)
            
        st.markdown("### ⚖️ Design Factors")
        method = st.radio("Method", ["SDM (Strength)", "WSD (Working)"], index=0)
        if "SDM" in method:
            c1, c2 = st.columns(2)
            fdl = c1.number_input("DL Factor", value=1.4)
            fll = c2.number_input("LL Factor", value=1.7)
        else:
            fdl, fll = 1.0, 1.0
            
        return {'fc':fc, 'fy':fy, 'fys':fys, 'b':b, 'h':h, 'cv':cover, 
                'db_main': db_main, 'db_stirrup': db_stirrup,
                'fdl':fdl, 'fll':fll, 'method':method, 'unit':unit_sys, 
                'u_force': u_force, 'u_len': 'm'}

def render_geometry():
    st.markdown('<div class="section-header">1️⃣ Geometry & Supports</div>', unsafe_allow_html=True)
    n = st.number_input("Number of Spans", 1, 10, 2)
    
    spans = []
    st.markdown("**Span Lengths (m)**")
    cols = st.columns(4)
    for i in range(n):
        with cols[i%4]:
            spans.append(st.number_input(f"L{i+1}", 1.0, 50.0, 5.0, key=f"s{i}"))
            
    st.markdown("**Support Types**")
    sups = []
    sup_opts = ["Pin", "Roller", "Fixed", "None"]
    cols = st.columns(5)
    for i in range(n+1):
        with cols[i%5]:
            def_idx = 0 if i==0 else (1 if i<n else 1)
            sups.append(st.selectbox(f"Node {i+1}", sup_opts, index=def_idx, key=f"sup{i}"))
            
    df_sup = pd.DataFrame({'x': [0]+list(np.cumsum(spans)), 'type': sups})
    ok, msg = check_stability(df_sup)
    if not ok: st.error(msg)
    return n, spans, df_sup, ok

def render_loads(n, spans, p):
    st.markdown('<div class="section-header">2️⃣ Loading Conditions</div>', unsafe_allow_html=True)
    loads = []
    tabs = st.tabs([f"Span {i+1}" for i in range(n)])
    
    u_load = "kg/m" if "Metric" in p['unit'] else "kN/m"
    u_point = "kg" if "Metric" in p['unit'] else "kN"
    
    for i, tab in enumerate(tabs):
        with tab:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"#### Uniform Load ({u_load})")
                wdl = st.number_input("DL", 0.0, key=f"wdl{i}")
                wll = st.number_input("LL", 0.0, key=f"wll{i}")
                wu = wdl*p['fdl'] + wll*p['fll']
                
                if wu != 0:
                    st.info(f"**Calc:** $W_u = {wu:.2f}$")
                    loads.append({'span_idx':i, 'type':'U', 'w':wu})

            with c2:
                st.markdown(f"#### Point Load ({u_point})")
                qty = st.number_input("Qty", 0, 5, 0, key=f"q{i}")
                for j in range(qty):
                    cc1, cc2, cc3 = st.columns(3)
                    pd_val = cc1.number_input(f"P_DL {j+1}", key=f"pd{i}{j}")
                    pl_val = cc2.number_input(f"P_LL {j+1}", key=f"pl{i}{j}")
                    px = cc3.number_input(f"x (m)", 0.0, spans[i], spans[i]/2.0, key=f"px{i}{j}")
                    
                    pu = pd_val*p['fdl'] + pl_val*p['fll']
                    if pu != 0:
                        st.info(f"**Calc:** $P_u = {pu:.2f}$")
                        loads.append({'span_idx':i, 'type':'P', 'P':pu, 'x':px})
    return loads
