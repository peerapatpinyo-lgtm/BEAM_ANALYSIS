import streamlit as st
import pandas as pd
import numpy as np
from solver import BeamSolver
import design_view

st.set_page_config(page_title="Beam Analysis & Design", layout="wide")

# --- Initialize Session State (เพื่อไม่ให้ข้อมูลหายเวลา Refresh) ---
if 'spans' not in st.session_state: st.session_state['spans'] = [5.0]
if 'supports' not in st.session_state: 
    st.session_state['supports'] = [{'id': 0, 'type': 'Pin'}, {'id': 1, 'type': 'Roller'}]
if 'loads' not in st.session_state: st.session_state['loads'] = []

st.title("🏗️ Beam Analysis & RC Design")

# --- SIDEBAR: Materials & Section ---
st.sidebar.header("1. Materials & Section")
fc_input = st.sidebar.number_input("Concrete Strength (f'c) [MPa]", 20.0, 50.0, 25.0)
fy_input = st.sidebar.number_input("Steel Strength (fy) [MPa]", 240.0, 500.0, 400.0)
b_val = st.sidebar.number_input("Beam Width (b) [m]", 0.1, 1.0, 0.3)
h_val = st.sidebar.number_input("Beam Depth (h) [m]", 0.1, 2.0, 0.5)

E = 2e11  # Elastic Modulus (Pa)
I = (b_val * h_val**3) / 12  # Moment of Inertia

st.sidebar.markdown("---")
st.sidebar.header("2. Load Factors (ULS)")
dl_f = st.sidebar.number_input("Dead Load Factor", 1.0, 2.0, 1.4)
ll_f = st.sidebar.number_input("Live Load Factor", 1.0, 2.0, 1.7)

# --- MAIN UI: Input Tabs ---
tab1, tab2, tab3 = st.tabs(["📏 Spans", "⚓ Supports", "⚖️ Loads"])

with tab1:
    n_spans = st.number_input("Number of Spans", 1, 10, len(st.session_state['spans']))
    # Adjust span list size
    if n_spans != len(st.session_state['spans']):
        if n_spans > len(st.session_state['spans']):
            st.session_state['spans'] += [5.0] * (n_spans - len(st.session_state['spans']))
        else:
            st.session_state['spans'] = st.session_state['spans'][:n_spans]
    
    new_spans = []
    cols = st.columns(min(n_spans, 4))
    for i in range(n_spans):
        val = cols[i%4].number_input(f"Span {i+1} Length (m)", 0.1, 20.0, float(st.session_state['spans'][i]), key=f"span_{i}")
        new_spans.append(val)
    st.session_state['spans'] = new_spans

with tab2:
    n_nodes = len(st.session_state['spans']) + 1
    st.write(f"Total Nodes: {n_nodes}")
    
    # ดึงค่าเดิมมาแสดงใน Data Editor
    current_sups = {s['id']: s['type'] for s in st.session_state['supports']}
    sup_data = [{"Node ID": i, "Support Type": current_sups.get(i, "None")} for i in range(n_nodes)]
    
    edited_sups = st.data_editor(
        pd.DataFrame(sup_data),
        column_config={
            "Node ID": st.column_config.NumberColumn(disabled=True),
            "Support Type": st.column_config.SelectboxColumn(options=["None", "Pin", "Roller", "Fixed"], required=True)
        },
        hide_index=True, use_container_width=True
    )
    # Save back to session state
    st.session_state['supports'] = [
        {'id': r['Node ID'], 'type': r['Support Type']} 
        for _, r in edited_sups.iterrows() if r['Support Type'] != "None"
    ]

with tab3:
    c1, c2, c3 = st.columns([1, 1, 2])
    s_idx = c1.selectbox("Span Index", range(len(st.session_state['spans'])), format_func=lambda x: f"Span {x+1}")
    l_type = c2.selectbox("Type", ["Point Load (P)", "Uniform (U)"])
    l_case = c2.selectbox("Case", ["DL", "LL"])
    
    mag = c3.number_input("Magnitude (kN or kN/m)", 0.0, 1000.0, 10.0)
    span_L = st.session_state['spans'][s_idx]
    
    if l_type == "Point Load (P)":
        x_loc = st.number_input("Position x (m from left)", 0.0, float(span_L), float(span_L)/2)
        dist = 0
    else:
        x_start = st.number_input("Start x (m)", 0.0, float(span_L), 0.0)
        x_end = st.number_input("End x (m)", 0.0, float(span_L), float(span_L))
        x_loc, dist = x_start, (x_end - x_start)

    if st.button("➕ Add Load"):
        l_code = 'P' if "Point" in l_type else 'U'
        st.session_state['loads'].append({
            'span_index': s_idx, 'type': l_code, 'mag': mag * 1000, # Convert to N
            'x': x_loc, 'dist': dist, 'case': l_case
        })
        st.rerun()

    if st.session_state['loads']:
        load_df = pd.DataFrame(st.session_state['loads'])
        st.dataframe(load_df, use_container_width=True)
        if st.button("🗑️ Clear All Loads"):
            st.session_state['loads'] = []
            st.rerun()

# --- ANALYSIS EXECUTION ---
st.markdown("---")
if st.button("🚀 RUN ANALYSIS", type="primary", use_container_width=True):
    if len(st.session_state['supports']) < 2:
        st.error("Error: Need at least 2 supports for stability.")
    else:
        # Apply Factors to Loads
        factored_loads = []
        for l in st.session_state['loads']:
            f_l = l.copy()
            f_l['mag'] *= (dl_f if l['case'] == 'DL' else ll_f)
            factored_loads.append(f_l)

        # Call Solver
        solver = BeamSolver(st.session_state['spans'], st.session_state['supports'], factored_loads, E, I, b=b_val, h=h_val)
        df, reactions, summary = solver.solve()

        if not df.empty:
            # 1. แสดงกราฟจาก design_view
            design_view.draw_interactive_diagrams(df, reactions, st.session_state['spans'], st.session_state['supports'], factored_loads)

            # 2. แสดงผลสรุป
            st.subheader("📊 Analysis Summary (Factored)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Max Shear", f"{summary['V_max']['value']/1000:.2f} kN")
            c2.metric("Max Moment (+)", f"{summary['M_pos']['value']/1000:.2f} kNm")
            c3.metric("Max Moment (-)", f"{summary['M_neg']['value']/1000:.2f} kNm")
            
            # Deflection
            d_elastic = summary['D_max']['value'] * 1000  # mm
            L_total = sum(st.session_state['spans'])
            d_allow = (L_total * 1000) / 240
            c4.metric("Max Deflection", f"{d_elastic:.2f} mm", delta=f"Limit: {d_allow:.1f} mm", delta_color="inverse")

            # 3. RC Design Result
            st.markdown("---")
            st.subheader("🏗️ RC Design (SDM)")
            rc_res = solver.design_rc_section(fc_input, fy_input)
            
            d1, d2 = st.columns(2)
            d1.info(f"**Required Top Steel (As_neg):** {rc_res.get('as_neg', 0):.2f} cm²")
            d2.success(f"**Required Bottom Steel (As_pos):** {rc_res.get('as_pos', 0):.2f} cm²")
        else:
            st.error("Analysis failed. Please check your inputs.")
