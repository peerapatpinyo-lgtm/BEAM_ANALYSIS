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
        st.markdown("### ⚙️ Project Standards")
        unit_sys = st.radio("Unit System", ["Metric (kg, m)", "SI (kN, m)"])
        u_force = "kg" if "Metric" in unit_sys else "kN"
        u_stress = "ksc" if "Metric" in unit_sys else "MPa"
        
        st.markdown("### 🧱 Material & Section")
        with st.expander("Properties", expanded=True):
            fc = st.number_input(f"fc' ({u_stress})", value=240.0, format="%.1f")
            fy = st.number_input(f"fy Main ({u_stress})", value=4000.0, format="%.1f")
            fys = st.number_input(f"fy Stirrup ({u_stress})", value=2400.0, format="%.1f")
            b = st.number_input("Width b (cm)", value=25.0, format="%.1f")
            h = st.number_input("Depth h (cm)", value=50.0, format="%.1f")
            cover = st.number_input("Covering (cm)", value=3.0, format="%.1f")

        st.markdown("### 🏗️ Reinforcement Control")
        with st.expander("Rebar Selection", expanded=True):
            db_main = st.selectbox("Main Bar (mm)", [12, 16, 20, 25, 28, 32], index=1)
            db_stirrup = st.selectbox("Stirrup (mm)", [6, 9, 12], index=0)
            # เพิ่มช่องใส่ Spacing Step ตามขอ
            stirrup_step = st.selectbox("Stirrup Spacing Step (cm)", [1.0, 2.5, 5.0], index=1, help="Round stirrup spacing down to nearest X cm")
            
        st.markdown("### ⚖️ Load Factors")
        method = st.radio("Design Method", ["SDM (Strength Design)"], index=0) # Lock to SDM for now
        c1, c2 = st.columns(2)
        fdl = c1.number_input("Factor DL", value=1.4, format="%.2f")
        fll = c2.number_input("Factor LL", value=1.7, format="%.2f")
            
        return {'fc':fc, 'fy':fy, 'fys':fys, 'b':b, 'h':h, 'cv':cover, 
                'db_main': db_main, 'db_stirrup': db_stirrup, 's_step': stirrup_step,
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
            spans.append(st.number_input(f"Span {i+1}", 1.0, 50.0, 5.0, key=f"s{i}"))
            
    st.markdown("**Support Conditions**")
    sups = []
    sup_opts = ["Pin", "Roller", "Fixed", "None"]
    cols = st.columns(5)
    for i in range(n+1):
        with cols[i%5]:
            def_idx = 0 if i==0 else (1 if i<n else 1)
            sups.append(st.selectbox(f"Node {i+1}", sup_opts, index=def_idx, key=f"sup{i}"))
            
    df_sup = pd.DataFrame({'x': [0]+list(np.cumsum(spans)), 'type': sups})
    ok, msg = check_stability(df_sup)
    if not ok: st.error(f"⛔ {msg}")
    return n, spans, df_sup, ok

def render_loads(n, spans, p):
    st.markdown('<div class="section-header">2️⃣ Loading Definitions</div>', unsafe_allow_html=True)
    loads = []
    tabs = st.tabs([f"Span {i+1}" for i in range(n)])
    
    u_load = "kg/m" if "Metric" in p['unit'] else "kN/m"
    u_point = "kg" if "Metric" in p['unit'] else "kN"
    
    for i, tab in enumerate(tabs):
        with tab:
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown(f"**Uniform Load ($W$)**")
                wdl = st.number_input("Dead Load (DL)", 0.0, key=f"wdl{i}")
                wll = st.number_input("Live Load (LL)", 0.0, key=f"wll{i}")
                
                # คำนวณละเอียด 1.4DL + 1.7LL
                wu = wdl*p['fdl'] + wll*p['fll']
                if wu != 0:
                    st.latex(f"W_u = ({p['fdl']}\\times{wdl}) + ({p['fll']}\\times{wll}) = \\mathbf{{{wu:,.2f}}}\; {u_load}")
                    loads.append({'span_idx':i, 'type':'U', 'w':wu})
                else:
                    st.caption("No Uniform Load")

            with c2:
                st.markdown(f"**Point Load ($P$)**")
                qty = st.number_input("Quantity", 0, 5, 0, key=f"q{i}")
                for j in range(qty):
                    cc1, cc2, cc3 = st.columns(3)
                    pd_val = cc1.number_input(f"P(DL)", key=f"pd{i}{j}")
                    pl_val = cc2.number_input(f"P(LL)", key=f"pl{i}{j}")
                    px = cc3.number_input(f"x (m)", 0.0, spans[i], spans[i]/2.0, key=f"px{i}{j}")
                    
                    # คำนวณละเอียด
                    pu = pd_val*p['fdl'] + pl_val*p['fll']
                    if pu != 0:
                        st.latex(f"P_u = ({p['fdl']}\\times{pd_val}) + ({p['fll']}\\times{pl_val}) = \\mathbf{{{pu:,.2f}}}\; {u_point}")
                        loads.append({'span_idx':i, 'type':'P', 'P':pu, 'x':px})
    return loads
