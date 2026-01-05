import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PIL 

# Fix Image Size Limit
PIL.Image.MAX_IMAGE_PIXELS = None 

from solver import BeamSolver

st.set_page_config(page_title="Beam Analysis", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 5px; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #f0f2f6; border-radius: 5px 5px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-bottom: 2px solid #4e8cff; }
</style>
""", unsafe_allow_html=True)

st.title("🏗️ Beam Analysis")

# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.header("Properties")
    E = st.number_input("Elastic Modulus (E) [Pa]", value=2.0e11, format="%.2e")
    I = st.number_input("Moment of Inertia (I) [m^4]", value=1.0e-4, format="%.2e")
    
    st.markdown("---")
    beam_type = st.selectbox("Beam Theory", ["Euler", "Timoshenko"])
    if beam_type == "Timoshenko":
        A = st.number_input("Cross-sectional Area (A) [m^2]", value=0.01, format="%.4f")
    else:
        A = 0.01

# --- INPUTS (3 TABS) ---
tab1, tab2, tab3 = st.tabs(["1. Geometry", "2. Supports", "3. Loads"])

# --- TAB 1: GEOMETRY ---
with tab1:
    st.subheader("Span Configuration")
    col1, col2 = st.columns([2, 1])
    with col1:
        spans_input = st.text_input("Span Lengths (m, comma separated)", "5, 5")
        try:
            spans = [float(x.strip()) for x in spans_input.split(',')]
            node_locs = np.concatenate(([0], np.cumsum(spans)))
            st.success(f"Total Length: {sum(spans)} m | Nodes: {len(node_locs)}")
        except:
            st.error("Invalid format")
            spans = []
            node_locs = []

# --- TAB 2: SUPPORTS ---
with tab2:
    st.subheader("Support Conditions")
    
    if 'supports_list' not in st.session_state:
        st.session_state.supports_list = [
            {'id': 0, 'type': 'Pin', 'settlement': 0.0, 'k_spring': 0.0},
            {'id': len(spans) if spans else 1, 'type': 'Roller', 'settlement': 0.0, 'k_spring': 0.0}
        ]
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Add Support**")
        if len(node_locs) > 0:
            s_node = st.selectbox("Node Index", range(len(node_locs)))
            s_type = st.selectbox("Type", ["Pin", "Roller", "Fixed"])
            s_settlement = st.number_input("Settlement (m)", value=0.0, step=0.001, format="%.4f")
            
            if st.button("Add Support"):
                st.session_state.supports_list = [s for s in st.session_state.supports_list if s['id'] != s_node]
                st.session_state.supports_list.append({
                    'id': s_node, 'type': s_type, 
                    'settlement': s_settlement, 'k_spring': 0.0
                })
                st.rerun()
                
    with c2:
        st.markdown("**Current Supports**")
        if st.session_state.supports_list:
            df_sup = pd.DataFrame(st.session_state.supports_list)
            if 'settlement' not in df_sup.columns: df_sup['settlement'] = 0.0
            df_sup = df_sup.sort_values(by='id')
            st.dataframe(df_sup[['id', 'type', 'settlement']], hide_index=True, use_container_width=True)
            if st.button("Clear Supports"):
                st.session_state.supports_list = []
                st.rerun()

# --- TAB 3: LOADS ---
with tab3:
    st.subheader("Applied Loads")
    
    if 'loads_list' not in st.session_state:
        st.session_state.loads_list = [{'span_idx': 0, 'type': 'U', 'mag': 10000, 'x': 0, 'dist': 5}]

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Add Load**")
        if spans:
            l_span = st.selectbox("Span Index", range(len(spans)))
            l_type = st.selectbox("Load Type", ["Point (P)", "Uniform (U)", "Moment (M)"])
            l_mag = st.number_input("Magnitude", value=1000.0)
            l_x = st.number_input("Location x (from left)", value=2.5)
            
            l_dist = 0.0
            if "Uniform" in l_type:
                l_dist = st.number_input("Dist (Length)", value=1.0)
            
            if st.button("Add Load"):
                l_code = 'P' if "Point" in l_type else ('U' if "Uniform" in l_type else 'M')
                new_load = {'span_idx': l_span, 'type': l_code, 'mag': l_mag, 'x': l_x}
                if l_code == 'U': new_load['dist'] = l_dist
                st.session_state.loads_list.append(new_load)
                st.rerun()
                
    with c2:
        st.markdown("**Current Loads**")
        if st.session_state.loads_list:
            df_load = pd.DataFrame(st.session_state.loads_list)
            st.dataframe(df_load, hide_index=True, use_container_width=True)
            if st.button("Clear Loads"):
                st.session_state.loads_list = []
                st.rerun()

# --- PLOTTING LOGIC (ORIGINAL CLEAN STYLE) ---
def plot_beam_diagram(ax, spans, supports, loads):
    total_len = sum(spans)
    node_x = np.concatenate(([0], np.cumsum(spans)))
    
    # Draw Beam
    ax.plot([0, total_len], [0, 0], 'k-', linewidth=3)
    ax.set_ylim(-2, 2)
    ax.set_xlim(-0.5, total_len + 0.5)
    ax.axis('off')
    
    # Draw Nodes
    ax.plot(node_x, np.zeros_like(node_x), 'ko', markersize=5)
    
    # Draw Supports
    for s in supports:
        if int(s['id']) < len(node_x):
            x = node_x[int(s['id'])]
            if s['type'] == 'Pin':
                ax.plot(x, -0.2, marker='^', color='green', markersize=12)
            elif s['type'] == 'Roller':
                ax.plot(x, -0.2, marker='o', color='green', markersize=10)
            elif s['type'] == 'Fixed':
                rect = patches.Rectangle((x-0.1, -0.4), 0.2, 0.8, color='green', alpha=0.5)
                ax.add_patch(rect)
    
    # Draw Loads
    max_load = 1
    if loads: max_load = max([abs(l['mag']) for l in loads])
    if max_load == 0: max_load = 1
    
    for l in loads:
        if int(l['span_idx']) < len(spans):
            x = node_x[int(l['span_idx'])] + l['x']
            mag = l['mag']
            
            if l['type'] == 'P':
                # Arrow points down for positive load
                dy = -0.8 if mag > 0 else 0.8
                ax.arrow(x, -dy, 0, dy*0.8, head_width=0.2, head_length=0.2, fc='red', ec='red')
                ax.text(x, -dy*1.2, f"P={mag}", ha='center', color='red')
            elif l['type'] == 'U':
                dist = l.get('dist', 1.0)
                rect = patches.Rectangle((x, 0), dist, 0.4, facecolor='orange', alpha=0.3)
                ax.add_patch(rect)
                ax.text(x + dist/2, 0.5, f"w={mag}", ha='center', color='orange')
            elif l['type'] == 'M':
                ax.text(x, 0.5, f"M={mag}", ha='center', color='purple')
    
    ax.set_title("System Diagram")

# --- MAIN CALCULATION ---
st.markdown("###")
if st.button("Calculate Analysis", type="primary"):
    if not spans:
        st.error("Please define spans first.")
    else:
        try:
            beam_props = {'E': E, 'I': I, 'A': A, 'type': beam_type}
            
            # SOLVER
            solver = BeamSolver(spans, st.session_state.supports_list, st.session_state.loads_list, beam_props)
            results, R = solver.solve()
            
            st.success("Calculation Complete")
            
            # PLOT (Standard 4 rows)
            fig, ax = plt.subplots(4, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [1, 2, 2, 2], 'hspace': 0.4})
            
            # 1. System
            plot_beam_diagram(ax[0], spans, st.session_state.supports_list, st.session_state.loads_list)
            
            # 2. Shear
            ax[1].plot(results['x'], results['shear'], 'b-', linewidth=1.5)
            ax[1].fill_between(results['x'], results['shear'], color='blue', alpha=0.1)
            ax[1].set_ylabel("Shear Force (N)")
            ax[1].set_title("Shear Force Diagram")
            ax[1].grid(True, linestyle=':', alpha=0.6)
            ax[1].axhline(0, color='black', linewidth=0.8)
            
            # 3. Moment (Standard View)
            ax[2].plot(results['x'], results['moment'], 'r-', linewidth=1.5)
            ax[2].fill_between(results['x'], results['moment'], color='red', alpha=0.1)
            ax[2].set_ylabel("Bending Moment (N-m)")
            ax[2].set_title("Bending Moment Diagram")
            ax[2].grid(True, linestyle=':', alpha=0.6)
            ax[2].axhline(0, color='black', linewidth=0.8)

            # 4. Deflection
            ax[3].plot(results['x'], results['deflection'], 'g-', linewidth=1.5)
            ax[3].set_ylabel("Deflection (m)")
            ax[3].set_xlabel("Position (m)")
            ax[3].set_title("Deflection")
            ax[3].grid(True, linestyle=':', alpha=0.6)
            ax[3].axhline(0, color='black', linewidth=0.8)
            
            # Max Deflection Label
            if len(results['deflection']) > 0:
                min_val = results['deflection'].min() # Typically negative
                min_idx = results['deflection'].idxmin()
                ax[3].plot(results.iloc[min_idx]['x'], min_val, 'ko', markersize=4)
                ax[3].text(results.iloc[min_idx]['x'], min_val, f" Max: {min_val:.4e} m", va='top')

            st.pyplot(fig, dpi=100)
            
        except Exception as e:
            st.error(f"Error: {e}")
