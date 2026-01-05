import streamlit as st
import pandas as pd
import numpy as np
from solver import BeamSolver
import design_view

# --- Page Config ---
st.set_page_config(page_title="Beam Analysis Pro", layout="wide", page_icon="🏗️")

# --- Session State ---
if 'spans' not in st.session_state: st.session_state['spans'] = [5.0, 5.0]
if 'supports' not in st.session_state: 
    st.session_state['supports'] = [{'id': 0, 'type': 'Pin'}, {'id': 1, 'type': 'Roller'}, {'id': 2, 'type': 'Roller'}]
if 'loads' not in st.session_state: st.session_state['loads'] = []

# --- Sidebar ---
st.sidebar.title("🏗️ Beam Settings")
st.sidebar.markdown("---")
if st.sidebar.button("Reset Project", type="primary"):
    st.session_state['spans'] = [5.0]
    st.session_state['supports'] = [{'id': 0, 'type': 'Pin'}, {'id': 1, 'type': 'Roller'}]
    st.session_state['loads'] = []
    st.rerun()

st.sidebar.markdown("### 1. Section Properties")
E = st.sidebar.number_input("Elastic Modulus (E) [Pa]", value=2e11, format="%.2e")

input_method = st.sidebar.radio("Input Method", ["Rectangular Size (b x h)", "Custom Properties (I, A)"])
if input_method == "Rectangular Size (b x h)":
    c1, c2 = st.sidebar.columns(2)
    b = c1.number_input("Width (b) [m]", 0.30)
    h = c2.number_input("Depth (h) [m]", 0.50)
    I = (b * h**3) / 12
    A = b * h
    st.sidebar.info(f"I = {I:.2e} m⁴ | A = {A:.2f} m²")
else:
    I = st.sidebar.number_input("Inertia (I) [m^4]", 5e-5, format="%.2e")
    A = st.sidebar.number_input("Area (A) [m^2]", 0.01, format="%.4f")

use_timoshenko = st.sidebar.checkbox("Advanced: Timoshenko", False)
G = st.sidebar.number_input("Shear Modulus (G)", 7.7e10, format="%.2e") if use_timoshenko else None

st.sidebar.markdown("---")
st.sidebar.markdown("### 2. Load Factors")
dl_factor = st.sidebar.number_input("Dead Load Factor", 1.4)
ll_factor = st.sidebar.number_input("Live Load Factor", 1.7)

# --- Main Interface ---
st.title("🏗️ Structural Beam Analysis")
# 1. แสดง Note วิธีการคำนวณ
st.info("ℹ️ **Analysis Method:** Direct Stiffness Method (FEM) for displacements, combined with **Statics Integration (Method of Sections)** for exact Shear & Moment diagrams.")

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
            new_spans.append(cols[i%4].number_input(f"Span {i+1}", value=float(current[i]), min_value=0.1, key=f"s_{i}"))
        st.session_state['spans'] = new_spans
    st.caption(f"Total Length: {sum(new_spans):.2f} m")

with tab2:
    sup_data = []
    # แปลงข้อมูล Support เดิมให้เป็น Format ตาราง
    nodes_count = len(st.session_state['spans']) + 1
    # สร้าง Map เดิม
    current_sups = {int(s.get('id', -1)): s.get('type') for s in st.session_state['supports'] if 'id' in s}
    
    for i in range(nodes_count):
        stype = current_sups.get(i, "None")
        sup_data.append({"Node ID": i+1, "Support Type": stype}) # Display 1-based

    edited = st.data_editor(pd.DataFrame(sup_data), column_config={
        "Node ID": st.column_config.NumberColumn(format="%d", disabled=True), 
        "Support Type": st.column_config.SelectboxColumn(options=["None","Pin","Roller","Fixed"], required=True)
    }, hide_index=True, use_container_width=True)
    
    # Save กลับเป็น 0-based ID
    st.session_state['supports'] = [{'id': r['Node ID']-1, 'type': r['Support Type']} for _, r in edited.iterrows() if r['Support Type'] != "None"]

with tab3:
    c1, c2, c3 = st.columns([1,1,2])
    span_idx = c1.selectbox("Select Span", range(len(st.session_state['spans'])), format_func=lambda x: f"Span {x+1}")
    l_type = c2.selectbox("Load Type", ["Point Load (P)", "Uniform Load (U)", "Moment (M)"])
    l_case = c2.selectbox("Case", ["DL", "LL"])
    mag = c3.number_input("Magnitude (kg, kg/m, kg-m)", value=1000.0)
    
    sl = st.session_state['spans'][span_idx]
    if "Uniform" in l_type:
        cp = st.columns(2)
        x1 = cp[0].number_input("Start (m)", 0.0, float(sl), 0.0)
        x2 = cp[1].number_input("End (m)", 0.0, float(sl), float(sl))
        x_loc, dist = x1, x2-x1
    else:
        x_loc = st.number_input("Position x (m)", 0.0, float(sl), float(sl)/2)
        dist = 0
        
    if st.button("➕ Add Load", type="primary"):
        code = 'P' if 'Point' in l_type else ('U' if 'Uniform' in l_type else 'M')
        # เก็บ Span Index ไปด้วยเพื่อความแม่นยำ
        st.session_state['loads'].append({'span_index': span_idx, 'type': code, 'mag': mag, 'x': x_loc, 'dist': dist, 'case': l_case})
        st.rerun()
        
    if st.session_state['loads']:
        # แสดงผล
        disp_data = []
        for l in st.session_state['loads']:
            s_idx = l.get('span_index', 0)
            disp_data.append({
                "Span": s_idx + 1,
                "Type": l['type'],
                "Mag": l['mag'],
                "Pos": f"x={l['x']}" + (f" to {l['x']+l['dist']}" if l['type']=='U' else ""),
                "Case": l['case']
            })
        st.dataframe(pd.DataFrame(disp_data), use_container_width=True)
        if st.button("Clear Last Load"): 
            st.session_state['loads'].pop()
            st.rerun()

# --- RUN ---
st.markdown("---")
if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
    if len(st.session_state['supports']) < 2:
        st.error("Need at least 2 supports to analyze.")
    else:
        # Load Factors & Gravity
        g = 9.81
        calc_loads = []
        for l in st.session_state['loads']:
            fac = dl_factor if l['case']=='DL' else ll_factor
            new_l = l.copy()
            new_l['mag'] = l['mag'] * fac * g # Convert kg -> N
            calc_loads.append(new_l)
        
        solver = BeamSolver(st.session_state['spans'], st.session_state['supports'], calc_loads, E, I, A, G)
        df, r, summ = solver.solve()
        
        if not df.empty:
            design_view.draw_interactive_diagrams(df, r, st.session_state['spans'], st.session_state['supports'], st.session_state['loads'], dl_factor, ll_factor)
            
            with st.expander("📊 View Critical Values & Reactions Details", expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Max Shear (V)", f"{summ['V_max']['value']/1000:.2f} kN")
                c2.metric("Max Moment (+)", f"{summ['M_pos']['value']/1000:.2f} kNm")
                c3.metric("Max Moment (-)", f"{summ['M_neg']['value']/1000:.2f} kNm", delta_color="inverse")
                
                # Deflection Check Logic
                max_span = max(st.session_state['spans'])
                limit = (max_span / 360) * 1000 # mm
                d_mm = summ['D_max']['value'] * 1000
                status = "✅ PASS" if abs(d_mm) < limit else "⚠️ CHECK"
                
                # 2. แก้ไข Label Deflection ให้ระบุเงื่อนไข
                c4.metric("Max Deflection", f"{d_mm:.4f} mm", f"{status} (Limit: L/360)")
                
                st.markdown("---")
                st.write("**Reactions (kN, kNm):**")
                
                # Reaction Logic Display
                # Map nodes to support types
                sup_map = {int(s['id']): s['type'] for s in st.session_state['supports'] if 'id' in s}
                n_nodes = len(st.session_state['spans']) + 1
                cols = st.columns(n_nodes)
                for i in range(n_nodes):
                    with cols[i]:
                        if i in sup_map:
                            st.markdown(f"**Node {i+1} ({sup_map[i]})**")
                            fy = r[2*i]
                            mz = r[2*i+1]
                            if abs(fy) > 1: st.write(f"Fy: {fy/1000:.2f}")
                            if abs(mz) > 1: st.write(f"Mz: {mz/1000:.2f}")
                        else:
                            st.caption(f"Node {i+1}")
                
                st.markdown("---")
                design_view.render_result_tables(df, r, st.session_state['spans'])
