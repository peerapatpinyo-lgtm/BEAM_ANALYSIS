import streamlit as st
import pandas as pd
import numpy as np

def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Global Settings")
        unit = st.radio("Unit System", ["Metric (kg, m)", "SI (kN, m)"])
        
        st.markdown("### Material Properties")
        with st.expander("Materials", expanded=True):
            fc = st.number_input("fc' (ksc/MPa)", value=240.0)
            fy = st.number_input("fy Main (ksc/MPa)", value=4000.0)
            fys = st.number_input("fy Stirrup (ksc/MPa)", value=2400.0)
        
        # Dimensions removed from sidebar, moved to Span Definitions
        
        st.markdown("### Rebar Options")
        with st.expander("Rebar Selection", expanded=True):
            db_main = st.selectbox("Main Bar (mm)", [12, 16, 20, 25, 28, 32], index=1)
            db_stir = st.selectbox("Stirrup Bar (mm)", [6, 9, 12], index=0)
            s_step = st.selectbox("Stirrup Spacing Step (cm)", [1.0, 2.5, 5.0], index=1)
        
        st.markdown("### Safety Factors")
        c1, c2 = st.columns(2)
        fdl = c1.number_input("Factor DL", value=1.4)
        fll = c2.number_input("Factor LL", value=1.7)
        
        u_force = "kg" if "Metric" in unit else "kN"
        
        return {
            'fc': fc, 'fy': fy, 'fys': fys,
            'db_main': db_main, 'db_stirrup': db_stir, 's_step': s_step,
            'fdl': fdl, 'fll': fll, 'unit': unit, 'u_force': u_force
        }

def render_geometry(n_default=2):
    st.markdown("### 1️⃣ Geometry & Section Properties")
    
    col_n, col_dummy = st.columns([1, 3])
    with col_n:
        n = st.number_input("Number of Spans", 1, 10, n_default)
    
    # List to store properties for each span
    span_props = []
    
    st.markdown("**Span Definitions & Sections**")
    tabs = st.tabs([f"Span {i+1}" for i in range(n)])
    
    spans_len = []
    for i, tab in enumerate(tabs):
        with tab:
            c1, c2, c3, c4 = st.columns(4)
            l = c1.number_input(f"Length (m)", 1.0, 50.0, 5.0, key=f"L{i}")
            # Section Inputs per Span
            b = c2.number_input(f"Width b (cm)", 10.0, 100.0, 25.0, key=f"b{i}")
            h = c3.number_input(f"Depth h (cm)", 10.0, 200.0, 50.0, key=f"h{i}")
            cv = c4.number_input(f"Cover (cm)", 1.0, 10.0, 3.0, key=f"cv{i}")
            
            spans_len.append(l)
            span_props.append({'b': b, 'h': h, 'cv': cv})

    st.markdown("**Support Conditions**")
    cols_sup = st.columns(min(n+1, 6))
    opts = ["Pin", "Roller", "Fixed", "None"]
    sup_types = []
    for i in range(n+1):
        with cols_sup[i%6]:
            def_idx = 2 if i==0 else (1 if i<n else 1) # Default: Fixed - Roller...
            s = st.selectbox(f"Node {i+1}", opts, index=def_idx, key=f"sup_{i}")
            sup_types.append(s)

    df_sup = pd.DataFrame({'x': [0]+list(np.cumsum(spans_len)), 'type': sup_types})
    
    valid = [t for t in sup_types if t != 'None']
    stable = True
    if len(valid) < 2 and "Fixed" not in valid: 
        stable = False
        st.error("❌ Structure Unstable: Need at least 2 supports or 1 Fixed support.")
    else:
        st.success("✅ Geometry & Sections Defined.")
        
    # RETURNS 5 VALUES
    return n, spans_len, df_sup, stable, span_props

def render_loads(n, spans, params):
    st.markdown("### 2️⃣ Load Configuration")
    loads = []
    tabs = st.tabs([f"Span {i+1}" for i in range(n)])
    u_load, u_point = ("kg/m", "kg") if "Metric" in params['unit'] else ("kN/m", "kN")
    
    for i, tab in enumerate(tabs):
        with tab:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Uniform Load ($W$)**")
                dl = st.number_input(f"DL ({u_load})", 0.0, key=f"wdl{i}")
                ll = st.number_input(f"LL ({u_load})", 0.0, key=f"wll{i}")
                wu = dl*params['fdl'] + ll*params['fll']
                if wu > 0:
                    st.latex(f"W_u = {params['fdl']}({dl}) + {params['fll']}({ll}) = \\mathbf{{{wu:,.0f}}}\; {u_load}")
                    loads.append({'span_idx': i, 'type': 'U', 'w': wu})
            with c2:
                st.markdown(f"**Point Loads ($P$)**")
                cnt = st.number_input("Count", 0, 5, 0, key=f"pcnt{i}")
                raw_p = []
                for j in range(cnt):
                    cc1, cc2, cc3 = st.columns([1,1,1.2])
                    pdl = cc1.number_input(f"P{j+1} DL", key=f"pdl{i}{j}")
                    pll = cc2.number_input(f"P{j+1} LL", key=f"pll{i}{j}")
                    px = cc3.number_input(f"x (m)", 0.0, spans[i], spans[i]/2, key=f"px{i}{j}")
                    pu = pdl*params['fdl'] + pll*params['fll']
                    if pu > 0:
                        st.latex(f"P_{{u{j+1}}} = {params['fdl']}({pdl}) + {params['fll']}({pll}) = \\mathbf{{{pu:,.0f}}}\; {u_point} @ {px}m")
                        raw_p.append({'P': pu, 'x': px})
                
                merged = {}
                for p in raw_p: merged[p['x']] = merged.get(p['x'], 0) + p['P']
                for x, p in merged.items(): loads.append({'span_idx': i, 'type': 'P', 'P': p, 'x': x})
    return loads
