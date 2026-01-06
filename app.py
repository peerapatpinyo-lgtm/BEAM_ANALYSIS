import streamlit as st
import pandas as pd
import numpy as np
from solver import BeamSolver
import design_view

st.set_page_config(page_title="Beam Pro Analysis & Design", layout="wide", page_icon="🏗️")

if 'spans' not in st.session_state: st.session_state['spans'] = [5.0, 5.0]
if 'supports' not in st.session_state: 
    st.session_state['supports'] = [{'id': 0, 'type': 'Pin'}, {'id': 1, 'type': 'Roller'}, {'id': 2, 'type': 'Roller'}]
if 'loads' not in st.session_state: st.session_state['loads'] = []

st.sidebar.title("🏗️ Settings")
st.sidebar.markdown("### 1. Section & Materials")
fc_prime = st.sidebar.number_input("Concrete Strength (f'c) [MPa]", 20.0, 50.0, 25.0)
fy = st.sidebar.number_input("Steel Strength (fy) [MPa]", 240.0, 500.0, 400.0)
E = st.sidebar.number_input("E [Pa]", value=2e11, format="%.2e")

c1, c2 = st.sidebar.columns(2)
b_in = c1.number_input("Width b [m]", 0.1, 1.0, 0.30)
h_in = c2.number_input("Depth h [m]", 0.1, 2.0, 0.50)
I = (b_in * h_in**3) / 12
A_area = b_in * h_in

st.sidebar.markdown("### 2. Load Factors")
dl_f = st.sidebar.number_input("DL Factor", 1.4)
ll_f = st.sidebar.number_input("LL Factor", 1.7)

st.title("🏗️ Beam Analysis & Design")
tab1, tab2, tab3 = st.tabs(["1️⃣ Spans", "2️⃣ Supports", "3️⃣ Loads"])

with tab1:
    n = st.number_input("Number of Spans", 1, 10, len(st.session_state['spans']))
    current = st.session_state['spans']
    if len(current) < n: current.extend([5.0]*(n-len(current)))
    else: current = current[:n]
    new_spans = []
    cols = st.columns(min(n, 4))
    for i in range(n):
        new_spans.append(cols[i%4].number_input(f"Span {i+1}", value=float(current[i]), key=f"s_{i}"))
    st.session_state['spans'] = new_spans

with tab2:
    sup_data = [{"Node ID": i+1, "Support Type": "None"} for i in range(len(st.session_state['spans'])+1)]
    current_sups = {int(s['id']): s['type'] for s in st.session_state['supports']}
    for d in sup_data: d["Support Type"] = current_sups.get(d["Node ID"]-1, "None")
    edited = st.data_editor(pd.DataFrame(sup_data), hide_index=True, use_container_width=True)
    st.session_state['supports'] = [{'id': r['Node ID']-1, 'type': r['Support Type']} for _, r in edited.iterrows() if r['Support Type'] != "None"]

with tab3:
    c1, c2, c3 = st.columns([1,1,2])
    span_idx = c1.selectbox("Span", range(len(st.session_state['spans'])))
    l_type = c2.selectbox("Type", ["Point Load (P)", "Uniform Load (U)"])
    mag = c3.number_input("Mag (kg)", 1000.0)
    if l_type == "Point Load (P)":
        x_loc = st.number_input("x (m)", 0.0, float(st.session_state['spans'][span_idx]), 2.5)
        dist = 0
    else:
        x_loc = 0; dist = st.session_state['spans'][span_idx]
    if st.button("➕ Add"):
        st.session_state['loads'].append({'span_index': span_idx, 'type': l_type[0], 'mag': mag, 'x': x_loc, 'dist': dist, 'case': 'DL'})
        st.rerun()

if st.button("🚀 Run Analysis & Design", type="primary", use_container_width=True):
    g = 9.81
    valid_loads = []
    check_force_y = 0
    for l in st.session_state['loads']:
        factored = l['mag'] * dl_f * g
        valid_loads.append({**l, 'mag': factored})
        check_force_y += factored if l['type'] == 'P' else factored * l['dist']
    
    solver = BeamSolver(st.session_state['spans'], st.session_state['supports'], valid_loads, E, I, A_area, b=b_in, h=h_in)
    df, r, summ = solver.solve()
    rc = solver.design_rc_section(fc_prime, fy)
    sh = solver.design_shear(fc_prime, fy)

    if not df.empty:
        design_view.draw_interactive_diagrams(df, r, st.session_state['spans'], st.session_state['supports'], valid_loads, dl_f, ll_f)
        
        with st.expander("📊 1. Analysis Summary (ULS)", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Max Shear", f"{summ['V_max']['value']/1000:.2f} kN", f"@ {summ['V_max']['x']:.2f} m")
            c2.metric("Max Moment (+)", f"{summ['M_pos']['value']/1000:.2f} kNm", f"@ {summ['M_pos']['x']:.2f} m")
            c3.metric("Max Moment (-)", f"{summ['M_neg']['value']/1000:.2f} kNm", f"@ {summ['M_neg']['x']:.2f} m")
            c4.metric("Max Deflection", f"{summ['D_max']['value']*1000:.2f} mm")
            
            st.markdown("---")
            st.markdown("#### Support Reactions")
            total_r = 0
            cols = st.columns(len(st.session_state['spans'])+1)
            for i in range(len(cols)):
                total_r += r[2*i]
                cols[i].write(f"**Node {i+1}**\n\nFy: {r[2*i]/1000:.2f} kN")
            
            diff = total_r - check_force_y
            if abs(diff) < 1.0: st.success(f"✅ Equilibrium Passed (Diff: {diff:.2f} N)")
            else: st.error(f"❌ Equilibrium Error: {diff:.2f} N")

        with st.expander("🏗️ 2. Reinforcement Design (SDM)", expanded=True):
            
            st.markdown(f"**Design for:** DB Size selection")
            db_size = st.selectbox("Select Bar Size", [12, 16, 20, 25], index=1, format_func=lambda x: f"DB{x}")
            as_bar = (np.pi * (db_size/10)**2) / 4
            
            d1, d2, d3 = st.columns(3)
            d1.metric("Top Steel", f"{rc['as_neg']:.2f} cm²", f"{np.ceil(rc['as_neg']/as_bar):.0f}-DB{db_size}")
            d2.metric("Bottom Steel", f"{rc['as_pos']:.2f} cm²", f"{np.ceil(rc['as_pos']/as_bar):.0f}-DB{db_size}")
            d3.metric("Stirrups (RB9)", f"@{sh['spacing_mm']:.0f} mm", f"Vu={sh['vu_kn']:.1f} kN")
