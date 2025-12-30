import streamlit as st
import pandas as pd
import numpy as np

def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        unit = st.radio("Unit System", ["Metric (kg, m)", "SI (kN, m)"])
        
        st.markdown("### Material Properties")
        with st.expander("Materials", expanded=True):
            fc = st.number_input("fc' (ksc/MPa)", value=240.0)
            fy = st.number_input("fy Main (ksc/MPa)", value=4000.0)
            fys = st.number_input("fy Stirrup (ksc/MPa)", value=2400.0)
        
        st.markdown("### Section Dimensions")
        with st.expander("Dimensions", expanded=True):
            b = st.number_input("Width b (cm)", value=25.0)
            h = st.number_input("Depth h (cm)", value=50.0)
            cv = st.number_input("Covering (cm)", value=3.0)
        
        st.markdown("### Rebar & Spacing")
        with st.expander("Rebar Selection", expanded=True):
            db_main = st.selectbox("Main Bar (mm)", [12, 16, 20, 25, 28, 32], index=1)
            db_stir = st.selectbox("Stirrup Bar (mm)", [6, 9, 12], index=0)
            s_step = st.selectbox("Spacing Step (cm)", [1.0, 2.5, 5.0], index=1)
        
        st.markdown("### Safety Factors")
        c1, c2 = st.columns(2)
        fdl = c1.number_input("Factor DL", value=1.4)
        fll = c2.number_input("Factor LL", value=1.7)
        
        u_force = "kg" if "Metric" in unit else "kN"
        
        return {
            'fc': fc, 'fy': fy, 'fys': fys, 'b': b, 'h': h, 'cv': cv,
            'db_main': db_main, 'db_stirrup': db_stir, 's_step': s_step,
            'fdl': fdl, 'fll': fll, 'unit': unit, 'u_force': u_force
        }

def render_geometry():
    st.markdown("### 1️⃣ Geometry & Supports")
    col1, col2 = st.columns([1, 2])
    with col1:
        n = st.number_input("Number of Spans", 1, 10, 2)
    
    spans = []
    st.write("**Span Lengths (m)**")
    cols = st.columns(min(n, 4))
    for i in range(n):
        with cols[i%4]:
            spans.append(st.number_input(f"Span {i+1}", 1.0, 50.0, 5.0, key=f"span_{i}"))
            
    st.write("**Support Conditions**")
    sup_types = []
    cols_sup = st.columns(min(n+1, 5))
    opts = ["Pin", "Roller", "Fixed", "None"]
    for i in range(n+1):
        with cols_sup[i%5]:
            def_idx = 0 if i==0 else (1 if i<n else 1)
            sup_types.append(st.selectbox(f"Node {i+1}", opts, index=def_idx, key=f"sup_{i}"))
            
    df_sup = pd.DataFrame({'x': [0]+list(np.cumsum(spans)), 'type': sup_types})
    
    valid_sups = [t for t in sup_types if t != 'None']
    stable = True
    # Stability Check
    if len(valid_sups) == 0:
        stable = False
        st.error("❌ **Structure Unstable:** No supports defined.")
    elif len(valid_sups) < 2 and "Fixed" not in valid_sups:
        stable = False
        st.error("❌ **Structure Unstable:** Needs at least 2 supports (or 1 Fixed) to be stable.")
    else:
        st.success("✅ Structure appears stable (Geometric check)")
    
    return n, spans, df_sup, stable

def render_loads(n, spans, params):
    st.markdown("### 2️⃣ Load Configuration")
    loads = []
    tabs = st.tabs([f"Span {i+1}" for i in range(n)])
    
    u_load = "kg/m" if "Metric" in params['unit'] else "kN/m"
    u_point = "kg" if "Metric" in params['unit'] else "kN"
    
    for i, tab in enumerate(tabs):
        with tab:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Uniform Load ($W$)**")
                dl = st.number_input(f"DL ({u_load})", 0.0, key=f"wdl_{i}")
                ll = st.number_input(f"LL ({u_load})", 0.0, key=f"wll_{i}")
                
                # UDL Calc
                wu = dl*params['fdl'] + ll*params['fll']
                if wu > 0:
                    st.latex(f"W_u = {params['fdl']}({dl}) + {params['fll']}({ll}) = \\mathbf{{{wu:,.2f}}}\; {u_load}")
                    loads.append({'span_idx': i, 'type': 'U', 'w': wu})
                elif wu == 0:
                    st.caption("No Uniform Load")
                    
            with c2:
                st.markdown(f"**Point Load ($P$)**")
                cnt = st.number_input("Count", 0, 5, 0, key=f"p_cnt_{i}")
                
                # Collect inputs
                raw_points = []
                for j in range(cnt):
                    st.markdown(f"**Load #{j+1}**")
                    cc1, cc2, cc3 = st.columns([1,1,1.2])
                    p_dl = cc1.number_input(f"PDL", key=f"pd_{i}_{j}")
                    p_ll = cc2.number_input(f"PLL", key=f"pl_{i}_{j}")
                    px = cc3.number_input(f"x (m)", 0.0, spans[i], spans[i]/2, key=f"px_{i}_{j}")
                    
                    pu = p_dl*params['fdl'] + p_ll*params['fll']
                    
                    # --- REQ 1: Show Pu Calculation for each point load ---
                    if pu > 0:
                        st.caption(f"Calc: {params['fdl']}*{p_dl} + {params['fll']}*{p_ll}")
                        st.latex(f"P_{{u,{j+1}}} = \\mathbf{{{pu:,.2f}}}\; {u_point} \quad @ x={px:.2f}m")
                        raw_points.append({'P': pu, 'x': px})
                        st.divider()

                # Aggregate logic
                merged_points = {}
                for p in raw_points:
                    pos = p['x']
                    if pos in merged_points:
                        merged_points[pos] += p['P']
                    else:
                        merged_points[pos] = p['P']
                
                for pos, total_p in merged_points.items():
                    loads.append({'span_idx': i, 'type': 'P', 'P': total_p, 'x': pos})

    return loads
