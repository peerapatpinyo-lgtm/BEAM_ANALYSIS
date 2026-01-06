import streamlit as st
import pandas as pd
import numpy as np
from solver import BeamSolver
import design_view

# --- Page Config ---
st.set_page_config(page_title="Beam Analysis & Design Pro", layout="wide", page_icon="🏗️")

# --- Session State ---
if 'spans' not in st.session_state: st.session_state['spans'] = [5.0, 5.0]
if 'supports' not in st.session_state: 
    st.session_state['supports'] = [{'id': 0, 'type': 'Pin'}, {'id': 1, 'type': 'Roller'}, {'id': 2, 'type': 'Roller'}]
if 'loads' not in st.session_state: st.session_state['loads'] = []

# --- Sidebar ---
st.sidebar.title("🏗️ Beam & Design Settings")
st.sidebar.markdown("---")

st.sidebar.markdown("### 1. Section & Materials")
fc_prime_input = st.sidebar.number_input("Concrete Strength (f'c) [MPa]", 20.0, 50.0, 25.0)
fy_input = st.sidebar.number_input("Steel Strength (fy) [MPa]", 240.0, 500.0, 400.0)

E = st.sidebar.number_input("Elastic Modulus (E) [Pa]", value=2e11, format="%.2e")

input_method = st.sidebar.radio("Input Method", ["Rectangular Size (b x h)", "Custom Properties"])
if input_method == "Rectangular Size (b x h)":
    c1, c2 = st.sidebar.columns(2)
    b_val = c1.number_input("Width (b) [m]", 0.1, 1.0, 0.30)
    h_val = c2.number_input("Depth (h) [m]", 0.1, 2.0, 0.50)
    I = (b_val * h_val**3) / 12
    A = b_val * h_val
    st.sidebar.info(f"I = {I:.2e} m⁴ | A = {A:.2f} m²")
else:
    I = st.sidebar.number_input("Inertia (I) [m^4]", 5e-5, format="%.2e")
    A = st.sidebar.number_input("Area (A) [m^2]", 0.01, format="%.4f")
    b_val, h_val = 0.3, 0.5 # Default for design logic

st.sidebar.markdown("---")
st.sidebar.markdown("### 2. Load Factors (ULS)")
dl_factor = st.sidebar.number_input("Dead Load Factor", 1.4)
ll_factor = st.sidebar.number_input("Live Load Factor", 1.7)

# --- Main Interface ---
st.title("🏗️ Structural Beam Analysis & RC Design")

tab1, tab2, tab3 = st.tabs(["1️⃣ Spans", "2️⃣ Supports", "3️⃣ Loads"])

with tab1:
    col_s1, _ = st.columns([2, 1])
    with col_s1:
        n = st.number_input("Number of Spans", 1, 10, len(st.session_state['spans']))
        current = st.session_state['spans']
        if len(current) < n: current.extend([5.0]*(n-len(current)))
        else: current = current[:n]
        new_spans = []
        cols = st.columns(min(n, 4))
        for i in range(n):
            new_spans.append(cols[i%4].number_input(f"Span {i+1} (m)", value=float(current[i]), min_value=0.1, key=f"s_{i}"))
        st.session_state['spans'] = new_spans

with tab2:
    sup_data = []
    nodes_count = len(st.session_state['spans']) + 1
    current_sups = {int(s.get('id', -1)): s.get('type') for s in st.session_state['supports'] if 'id' in s}
    for i in range(nodes_count):
        stype = current_sups.get(i, "None")
        sup_data.append({"Node ID": i+1, "Support Type": stype}) 
    edited = st.data_editor(pd.DataFrame(sup_data), column_config={
        "Node ID": st.column_config.NumberColumn(format="%d", disabled=True), 
        "Support Type": st.column_config.SelectboxColumn(options=["None","Pin","Roller","Fixed"], required=True)
    }, hide_index=True, use_container_width=True)
    st.session_state['supports'] = [{'id': r['Node ID']-1, 'type': r['Support Type']} for _, r in edited.iterrows() if r['Support Type'] != "None"]

with tab3:
    c1, c2, c3 = st.columns([1,1,2])
    span_idx = c1.selectbox("Select Span", range(len(st.session_state['spans'])), format_func=lambda x: f"Span {x+1}")
    l_type = c2.selectbox("Load Type", ["Point Load (P)", "Uniform Load (U)", "Moment (M)"])
    l_case = c2.selectbox("Case", ["DL", "LL"])
    mag = c3.number_input("Magnitude (kg, kg/m)", value=1000.0)
    sl = st.session_state['spans'][span_idx]
    if "Uniform" in l_type:
        cp = st.columns(2); x1 = cp[0].number_input("Start (m)", 0.0, float(sl), 0.0)
        x2 = cp[1].number_input("End (m)", 0.0, float(sl), float(sl)); x_loc, dist = x1, x2-x1
    else:
        x_loc = st.number_input("Position x (m)", 0.0, float(sl), float(sl)/2); dist = 0
    if st.button("➕ Add Load", type="primary"):
        code = 'P' if 'Point' in l_type else ('U' if 'Uniform' in l_type else 'M')
        st.session_state['loads'].append({'span_index': span_idx, 'type': code, 'mag': mag, 'x': x_loc, 'dist': dist, 'case': l_case})
        st.rerun()
    if st.session_state['loads']:
        st.dataframe(st.session_state['loads'], use_container_width=True)
        if st.button("Clear Last Load"): st.session_state['loads'].pop(); st.rerun()

# --- RUN ANALYSIS & DESIGN ---
st.markdown("---")
if st.button("🚀 Run Analysis & Design", type="primary", use_container_width=True):
    if len(st.session_state['supports']) < 2:
        st.error("Error: Unstable Structure.")
    else:
        g = 9.81
        valid_loads = []
        for l in st.session_state['loads']:
            if int(l['span_index']) < len(st.session_state['spans']):
                new_l = l.copy()
                new_l['mag'] *= (dl_factor if l['case']=='DL' else ll_factor) * g
                valid_loads.append(new_l)
        
        solver = BeamSolver(st.session_state['spans'], st.session_state['supports'], valid_loads, E, I, A, b=b_val, h=h_val)
        df, r, summ = solver.solve()
        rc_design = solver.design_rc_section(fc_prime_input, fy_input)

        if not df.empty:
            design_view.draw_interactive_diagrams(df, r, st.session_state['spans'], st.session_state['supports'], valid_loads, dl_factor, ll_factor)
            
            with st.expander("📊 Analysis & RC Design Results", expanded=True):
                # Analysis Summary
                st.markdown("#### 1. Analysis Summary (ULS)")
                c1, c2, c3 = st.columns(3)
                c1.metric("Max Shear", f"{summ['V_max']['value']/1000:.2f} kN")
                c2.metric("Max Moment (+)", f"{summ['M_pos']['value']/1000:.2f} kNm")
                c3.metric("Max Moment (-)", f"{summ['M_neg']['value']/1000:.2f} kNm")
                
                # RC Design Summary
                st.markdown("---")
                st.markdown("#### 2. Reinforcement Design (SDM)")
                
                d1, d2, d3 = st.columns(3)
                d1.metric("Top Steel (As_neg)", f"{rc_design['as_neg']:.2f} cm²")
                d2.metric("Bottom Steel (As_pos)", f"{rc_design['as_pos']:.2f} cm²")
                d3.info(f"Section: {rc_design['b_mm']:.0f}x{rc_design['h_mm']:.0f} mm\nf'c: {fc_prime_input} MPa")
                
                if rc_design['as_pos'] == -1 or rc_design['as_neg'] == -1:
                    st.error("❌ Section size is too small for the applied moment. Please increase b or h.")
