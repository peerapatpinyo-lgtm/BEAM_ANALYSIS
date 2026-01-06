import streamlit as st
import pandas as pd
import numpy as np
from solver import BeamSolver
import design_view

st.set_page_config(page_title="Professional Beam Solver", layout="wide")

# Initialize Session States
for key, val in [('loads', []), ('spans', [5.0]), ('supports', [{'id': 0, 'type': 'Pin'}, {'id': 1, 'type': 'Roller'}])]:
    if key not in st.session_state: st.session_state[key] = val

st.sidebar.title("🏗️ Project Materials")
fc = st.sidebar.number_input("Concrete f'c (MPa)", 20, 50, 25)
fy = st.sidebar.number_input("Steel fy (MPa)", 240, 500, 400)
b = st.sidebar.number_input("Beam Width (m)", 0.1, 0.6, 0.20)
h = st.sidebar.number_input("Beam Depth (m)", 0.1, 1.2, 0.40)
I = (b * h**3) / 12

# Interface Tabs
t1, t2, t3 = st.tabs(["📏 Spans", "⚖️ Supports", "📥 Add Loads"])

with t1:
    n_spans = st.number_input("Number of Spans", 1, 10, len(st.session_state['spans']))
    st.session_state['spans'] = [st.columns(4)[i%4].number_input(f"Span {i+1}", 0.1, 20.0, float(st.session_state['spans'][i] if i < len(st.session_state['spans']) else 5.0), key=f"s{i}") for i in range(n_spans)]

with t2:
    sup_df = pd.DataFrame([{"Node": i+1, "Type": "None"} for i in range(len(st.session_state['spans'])+1)])
    curr = {int(s['id']): s['type'] for s in st.session_state['supports']}
    sup_df['Type'] = sup_df['Node'].apply(lambda x: curr.get(x-1, "None"))
    ed_sup = st.data_editor(sup_df, hide_index=True, use_container_width=True)
    st.session_state['supports'] = [{'id': r['Node']-1, 'type': r['Type']} for _, r in ed_sup.iterrows() if r['Type'] != "None"]

with t3:
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    s_idx = c1.selectbox("Span", range(len(st.session_state['spans'])))
    l_type = c2.selectbox("Load Type", ["Point (P)", "UDL (U)", "Moment (M)"])
    l_mag = c3.number_input("Magnitude (kg or kg/m)", 0.0, 100000.0, 1000.0)
    l_x = c4.number_input("Start x (m)", 0.0, float(st.session_state['spans'][s_idx]), 0.0)
    l_dist = st.number_input("UDL Length (m)", 0.0, float(st.session_state['spans'][s_idx]-l_x), float(st.session_state['spans'][s_idx]-l_x)) if "UDL" in l_type else 0.0
    
    if st.button("➕ Add Load"):
        st.session_state['loads'].append({'span_index': s_idx, 'type': l_type[0], 'mag': l_mag, 'x': l_x, 'dist': l_dist})
        st.rerun()
    if st.session_state['loads']:
        st.table(st.session_state['loads'])
        if st.button("🗑️ Clear Loads"): st.session_state['loads'] = []; st.rerun()

# Execution
if st.button("🚀 RUN FULL ANALYSIS & DESIGN", type="primary", use_container_width=True):
    g = 9.81
    factored_loads = [{**l, 'mag': l['mag'] * 1.4 * g} for l in st.session_state['loads']]
    total_input_force = sum(l['mag'] if l['type']=='P' else l['mag']*l['dist'] for l in factored_loads)

    solver = BeamSolver(st.session_state['spans'], st.session_state['supports'], factored_loads, 2e11, I, b=b, h=h)
    df, r, summ = solver.solve()
    
    design_view.draw_interactive_diagrams(df, r, st.session_state['spans'], st.session_state['supports'], factored_loads, 1.4, 1.7)
    
    with st.expander("📊 1. Analysis Summary (ULS)", expanded=True):
        c = st.columns(4)
        c[0].metric("V_max", f"{summ['V_max']['value']/1000:.2f} kN")
        c[1].metric("M_pos", f"{summ['M_pos']['value']/1000:.2f} kNm")
        c[2].metric("M_neg", f"{abs(summ['M_neg']['value'])/1000:.2f} kNm")
        c[3].metric("Max Defl.", f"{summ['D_max']['value']*1000:.2f} mm")
        
        st.write("**Support Reactions:**")
        cols = st.columns(len(st.session_state['spans'])+1)
        for i in range(len(cols)): cols[i].info(f"Node {i+1}: {r[2*i]/1000:.2f} kN")
        
        # Equilibrium Check
        diff = sum(r[::2]) - total_input_force
        if abs(diff) < 1.0: st.success(f"✅ Equilibrium Passed (Total: {total_input_force/1000:.2f} kN)")
        else: st.error(f"❌ Equilibrium Error: {diff:.2f} N")

    with st.expander("🏗️ 2. RC Design (Construction Note)", expanded=True):
        bar_d = st.selectbox("Select Bar", [12, 16, 20, 25], index=1, format_func=lambda x: f"DB{x}")
        d_res = solver.pro_design(fc, fy, bar_d)
        
        st.columns(3)[0].metric("Top Bars", f"{d_res['n_neg']:.0f}-DB{bar_d}", f"As: {d_res['as_neg']:.2f} cm²")
        st.columns(3)[1].metric("Bottom Bars", f"{d_res['n_pos']:.0f}-DB{bar_d}", f"As: {d_res['as_pos']:.2f} cm²")
        st.columns(3)[2].info(f"**Site Note:**\n- Spacing: {'OK' if d_res['spacing_ok'] else '❌ Tight'}\n- Ld: {d_res['ld_mm']:.0f} mm")
