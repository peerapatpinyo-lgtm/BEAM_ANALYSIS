import streamlit as st
import pandas as pd
import numpy as np
from solver import BeamSolver
import design_view

st.set_page_config(page_title="Professional Continuous Beam Design", layout="wide")

# --- 1. ฟื้นฟูระบบจัดการหน่วยและวัสดุ (Materials & Units) ---
st.sidebar.title("🏗️ Project Setting")
fc = st.sidebar.number_input("Concrete f'c (MPa)", 20, 50, 25)
fy = st.sidebar.number_input("Steel fy (MPa)", 240, 500, 400)
E_val = 4700 * np.sqrt(fc) * 1e6  # Pa (N/m2) - ฟื้นฟูสูตร ACI

st.sidebar.divider()
sec_type = st.sidebar.selectbox("Beam Section", ["Rectangular", "T-Beam"])
b = st.sidebar.number_input("Width (m)", 0.1, 0.6, 0.20)
h = st.sidebar.number_input("Depth (m)", 0.1, 1.2, 0.40)

if sec_type == "T-Beam":
    bf = st.sidebar.number_input("Flange bf (m)", 0.2, 2.0, 0.6)
    hf = st.sidebar.number_input("Flange hf (m)", 0.05, 0.3, 0.1)
    # ฟื้นฟูการคำนวณ I สำหรับ T-Beam
    area = (bf * hf) + (b * (h - hf))
    y_bar = ((bf * hf * (h - hf/2)) + (b * (h - hf) * (h - hf)/2)) / area
    I_val = (bf * hf**3)/12 + (bf * hf * (h - hf/2 - y_bar)**2) + \
            (b * (h - hf)**3)/12 + (b * (h - hf) * (y_bar - (h-hf)/2)**2)
else:
    I_val = (b * h**3) / 12
    area = b * h

# --- 2. ฟื้นฟู Session States (Data Persistence) ---
for key, val in [('loads', []), ('spans', [5.0, 5.0]), ('supports', [{'id': 1, 'type': 'Pin'}, {'id': 2, 'type': 'Roller'}, {'id': 3, 'type': 'Roller'}])]:
    if key not in st.session_state: st.session_state[key] = val

# --- 3. ส่วนจัดการ Geometry & Loading (Tabs) ---
tab1, tab2, tab3 = st.tabs(["📏 Geometry", "⚖️ Supports", "📥 Load Inputs"])

with tab1:
    n_spans = st.number_input("Spans", 1, 10, len(st.session_state['spans']))
    st.session_state['spans'] = [st.columns(4)[i%4].number_input(f"L{i+1}", 0.1, 20.0, float(st.session_state['spans'][i] if i < len(st.session_state['spans']) else 5.0), key=f"span_{i}") for i in range(n_spans)]

with tab2:
    # ฟื้นฟู Logic Node ID (1-based)
    nodes_count = len(st.session_state['spans']) + 1
    sup_df = pd.DataFrame([{"Node ID": i+1, "Type": "None"} for i in range(nodes_count)])
    curr_sup = {int(s['id']): s['type'] for s in st.session_state['supports']}
    sup_df['Type'] = sup_df['Node ID'].apply(lambda x: curr_sup.get(x, "None"))
    
    ed_sup = st.data_editor(sup_df, hide_index=True, use_container_width=True)
    st.session_state['supports'] = [{'id': r['Node ID'], 'type': r['Type']} for _, r in ed_sup.iterrows() if r['Type'] != "None"]

with tab3:
    c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
    s_idx = c1.selectbox("Span", range(len(st.session_state['spans'])))
    l_type = c2.selectbox("Type", ["Point (P)", "UDL (U)"])
    l_case = c3.selectbox("Case", ["DL", "LL"])
    l_mag = c4.number_input("Mag (kg)", 0.0, 50000.0, 1000.0)
    l_x = c5.number_input("Start x", 0.0, float(st.session_state['spans'][s_idx]), 0.0)
    
    l_dist = 0.0
    if l_type == "UDL (U)":
        l_dist = st.number_input("Length (m)", 0.0, float(st.session_state['spans'][s_idx]-l_x), float(st.session_state['spans'][s_idx]-l_x))
    
    if st.button("➕ Add Load"):
        st.session_state['loads'].append({'span_index': s_idx, 'type': l_type[0], 'mag': l_mag, 'x': l_x, 'dist': l_dist, 'case': l_case})
        st.rerun()

    if st.session_state['loads']:
        st.table(st.session_state['loads'])
        if st.button("🗑️ Clear Loads"): st.session_state['loads'] = []; st.rerun()

# --- 4. การคำนวณและแสดงผล (Execution & Results) ---
if st.button("🚀 EXECUTE PROFESSIONAL ANALYSIS", type="primary", use_container_width=True):
    # ฟื้นฟูระบบ Load Combination (1.4DL + 1.7LL) และแปลงหน่วย kg -> N
    g = 9.81
    factored_loads = []
    for l in st.session_state['loads']:
        factor = 1.4 if l['case'] == "DL" else 1.7
        factored_loads.append({**l, 'mag': l['mag'] * factor * g})

    solver = BeamSolver(st.session_state['spans'], st.session_state['supports'], factored_loads, E_val, I_val, A=area)
    df, r, summ = solver.solve()
    
    # วาดกราฟ (SFD, BMD)
    design_view.draw_interactive_diagrams(df, r, st.session_state['spans'], st.session_state['supports'], factored_loads, 1.4, 1.7)
    
    # ฟื้นฟูตาราง Reactions ที่ตกหล่นไป
    st.subheader("📊 Support Reactions (Total Factor)")
    reac_data = []
    for i in range(len(st.session_state['spans']) + 1):
        reac_data.append({
            "Support Node": i + 1,
            "Reaction (kN)": round(r[2*i] / 1000, 2),
            "Moment (kNm)": round(r[2*i+1] / 1000, 2)
        })
    st.table(reac_data)

    # สรุปผลการออกแบบ RC Design
    st.subheader("🏗️ RC Design Construction Note")
    
    
    phi, d = 0.9, h - 0.05
    as_pos = (summ['M_pos']['value']) / (phi * fy * 1e6 * 0.9 * d) * 10000 # cm2
    as_neg = (abs(summ['M_neg']['value'])) / (phi * fy * 1e6 * 0.9 * d) * 10000 # cm2
    
    col1, col2 = st.columns(2)
    col1.metric("Bottom Steel (Positive Moment)", f"{as_pos:.2f} cm²")
    col2.metric("Top Steel (Negative Moment)", f"{as_neg:.2f} cm²")
