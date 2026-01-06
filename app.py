import streamlit as st
import pandas as pd
import numpy as np
from solver import BeamSolver
import design_view

st.set_page_config(page_title="Pro Beam Builder", layout="wide")

if 'loads' not in st.session_state: st.session_state['loads'] = []
if 'spans' not in st.session_state: st.session_state['spans'] = [5.0]
if 'supports' not in st.session_state: st.session_state['supports'] = [{'id': 0, 'type': 'Pin'}, {'id': 1, 'type': 'Roller'}]

# Sidebar
st.sidebar.title("🏗️ Project Materials")
fc = st.sidebar.number_input("Concrete f'c (MPa)", 20, 40, 25)
fy = st.sidebar.number_input("Steel fy (MPa)", 240, 500, 400)
b = st.sidebar.number_input("Beam Width (m)", 0.1, 0.5, 0.20)
h = st.sidebar.number_input("Beam Depth (m)", 0.1, 1.0, 0.40)

# Tabs
t1, t2, t3 = st.tabs(["📏 Geometry", "⚖️ Supports", "📥 Loading"])

with t1:
    n_spans = st.number_input("Spans", 1, 10, len(st.session_state['spans']))
    cols = st.columns(4)
    new_spans = []
    for i in range(n_spans):
        val = st.session_state['spans'][i] if i < len(st.session_state['spans']) else 5.0
        new_spans.append(cols[i%4].number_input(f"Span {i+1} (m)", 0.5, 20.0, float(val), key=f"sp_{i}"))
    st.session_state['spans'] = new_spans

with t2:
    sup_df = pd.DataFrame([{"Node": i+1, "Type": "None"} for i in range(len(new_spans)+1)])
    curr = {int(s['id']): s['type'] for s in st.session_state['supports']}
    sup_df['Type'] = sup_df['Node'].apply(lambda x: curr.get(x-1, "None"))
    ed_sup = st.data_editor(sup_df, hide_index=True)
    st.session_state['supports'] = [{'id': r['Node']-1, 'type': r['Type']} for _, r in ed_sup.iterrows() if r['Type'] != "None"]

with t3:
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    s_idx = c1.selectbox("Span", range(len(new_spans)))
    l_type = c2.selectbox("Type", ["Point (P)", "UDL (U)", "Moment (M)"])
    l_mag = c3.number_input("Load (kg or kg/m)", 0.0, 100000.0, 1000.0)
    l_x = c4.number_input("Pos x (m)", 0.0, float(new_spans[s_idx]), 0.0)
    l_dist = 0.0
    if "UDL" in l_type:
        l_dist = st.number_input("Length (m)", 0.0, float(new_spans[s_idx] - l_x), float(new_spans[s_idx] - l_x))
    
    if st.button("➕ Add Load"):
        st.session_state['loads'].append({'span_index': s_idx, 'type': l_type[0], 'mag': l_mag, 'x': l_x, 'dist': l_dist, 'case': 'DL'})
        st.rerun()
    st.dataframe(st.session_state['loads'])
    if st.button("🗑️ Clear All"): st.session_state['loads'] = []; st.rerun()

# Run
if st.button("🚀 EXECUTE ANALYSIS & DESIGN", type="primary", use_container_width=True):
    solver = BeamSolver(st.session_state['spans'], st.session_state['supports'], 
                        [{**l, 'mag': l['mag']*1.4*9.81} for l in st.session_state['loads']], 
                        2e11, (b*h**3)/12, b*h, b=b, h=h)
    df, r, summ = solver.solve()
    
    # UI Results
    design_view.draw_interactive_diagrams(df, r, st.session_state['spans'], st.session_state['supports'], st.session_state['loads'], 1.4, 1.7)
    
    st.subheader("📊 Engineering Summary")
    c = st.columns(4)
    c[0].metric("V_max", f"{summ['V_max']['value']/1000:.1f} kN")
    c[1].metric("M_pos", f"{summ['M_pos']['value']/1000:.1f} kNm")
    c[2].metric("M_neg", f"{abs(summ['M_neg']['value'])/1000:.1f} kNm")
    c[3].metric("Deflection", f"{summ['D_max']['value']*1000:.2f} mm")

    st.divider()
    st.subheader("🏗️ Construction Detail (RC Design)")
    bar_d = st.selectbox("Select Main Steel", [12, 16, 20, 25, 28], index=1, format_func=lambda x: f"DB{x}")
    d_res = solver.pro_design(fc, fy, bar_d)
    
    

    col1, col2, col3 = st.columns(3)
    col1.warning(f"**Top Bar (Support)**\n\n{d_res['n_neg']:.0f} - DB{bar_d}\n\nArea: {d_res['as_neg']:.2f} cm²")
    col2.success(f"**Bottom Bar (Span)**\n\n{d_res['n_pos']:.0f} - DB{bar_d}\n\nArea: {d_res['as_pos']:.2f} cm²")
    col3.info(f"**Site Notes**\n\n- Dev. Length (Ld): {d_res['ld_mm']:.0f} mm\n- Spacing OK: {'✅' if d_res['spacing_ok'] else '❌ Too Tight'}")
