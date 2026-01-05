import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from solver import BeamSolver

st.set_page_config(page_title="Beam Analysis", layout="wide")

# --- CUSTOM CSS FOR BETTER LOOK ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 5px; }
    .reportview-container .main .block-container { max-width: 1000px; }
</style>
""", unsafe_allow_html=True)

st.title("🏗️ Beam Analysis (High Precision)")

# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.header("⚙️ Parameters")
    E = st.number_input("Elastic Modulus (E) [Pa]", value=2.0e11, format="%.2e")
    I = st.number_input("Moment of Inertia (I) [m^4]", value=1.0e-4, format="%.2e")
    
    st.markdown("---")
    st.markdown("**Advanced Settings**")
    beam_type = st.selectbox("Theory", ["Euler", "Timoshenko"])
    if beam_type == "Timoshenko":
        A = st.number_input("Cross-sectional Area (A) [m^2]", value=0.01, format="%.4f")
    else:
        A = 0.01

# --- INPUT SECTION (CLASSIC STYLE) ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Spans")
    spans_input = st.text_input("Enter span lengths (m, comma separated)", "5, 5")
    try:
        spans = [float(x.strip()) for x in spans_input.split(',')]
        n_nodes = len(spans) + 1
        # Node locations for reference
        node_locs = np.concatenate(([0], np.cumsum(spans)))
    except:
        st.error("Invalid span format")
        spans = []
        node_locs = []

with col2:
    if spans:
        st.info(f"Beam has {len(spans)} spans. Total Length: {sum(spans)} m.")
        st.write(f"Nodes are at: {list(node_locs)}")

st.markdown("---")

# --- SUPPORTS & LOADS (SIMPLE UI) ---
c1, c2 = st.columns(2)

with c1:
    st.subheader("2. Supports")
    # Simple list storage
    if 'supports_list' not in st.session_state:
        st.session_state.supports_list = [{'id': 0, 'type': 'Pin'}, {'id': len(spans), 'type': 'Roller'}]
    
    # Input Form
    with st.expander("Add/Edit Supports", expanded=True):
        s_node = st.selectbox("Node Index", range(len(node_locs)))
        s_type = st.selectbox("Type", ["Pin", "Roller", "Fixed"])
        s_settlement = st.number_input("Settlement (m)", value=0.0, step=0.001, format="%.4f")
        
        if st.button("Add Support"):
            # Remove existing at this node if any
            st.session_state.supports_list = [s for s in st.session_state.supports_list if s['id'] != s_node]
            st.session_state.supports_list.append({
                'id': s_node, 'type': s_type, 
                'settlement': s_settlement, 'k_spring': 0
            })
            st.rerun() # Refresh to show in list

    # Display Current Supports
    st.write("**Current Supports:**")
    if st.session_state.supports_list:
        df_sup = pd.DataFrame(st.session_state.supports_list)
        st.dataframe(df_sup[['id', 'type', 'settlement']], hide_index=True, use_container_width=True)
        if st.button("Clear All Supports"):
            st.session_state.supports_list = []
            st.rerun()

with c2:
    st.subheader("3. Loads")
    if 'loads_list' not in st.session_state:
        st.session_state.loads_list = [{'span_idx': 0, 'type': 'U', 'mag': 10000, 'x': 0, 'dist': 5}]

    with st.expander("Add Loads", expanded=True):
        l_span = st.selectbox("Span Index", range(len(spans)))
        l_type = st.selectbox("Load Type", ["Point (P)", "Uniform (U)", "Moment (M)"])
        l_mag = st.number_input("Magnitude (N or N/m)", value=1000.0)
        l_x = st.number_input("Distance from Left of Span (x)", value=2.5)
        
        l_dist = 0.0
        if "Uniform" in l_type:
            l_dist = st.number_input("Length of Load (dist)", value=1.0)
        
        if st.button("Add Load"):
            l_code = 'P' if "Point" in l_type else ('U' if "Uniform" in l_type else 'M')
            new_load = {'span_idx': l_span, 'type': l_code, 'mag': l_mag, 'x': l_x}
            if l_code == 'U': new_load['dist'] = l_dist
            st.session_state.loads_list.append(new_load)
            st.rerun()

    st.write("**Current Loads:**")
    if st.session_state.loads_list:
        df_load = pd.DataFrame(st.session_state.loads_list)
        st.dataframe(df_load, hide_index=True, use_container_width=True)
        if st.button("Clear All Loads"):
            st.session_state.loads_list = []
            st.rerun()

# --- VISUALIZATION FUNCTION (THE CLASSIC BEAM DIAGRAM) ---
def plot_beam_diagram(ax, spans, supports, loads):
    total_len = sum(spans)
    node_x = np.concatenate(([0], np.cumsum(spans)))
    
    # Draw Beam
    ax.plot([0, total_len], [0, 0], 'k-', linewidth=3)
    ax.set_ylim(-2, 2)
    ax.set_xlim(-0.5, total_len + 0.5)
    ax.axis('off')
    
    # Draw Nodes
    ax.plot(node_x, np.zeros_like(node_x), 'ko', markersize=4)
    
    # Draw Supports
    for s in supports:
        x_pos = node_x[int(s['id'])]
        if s['type'] == 'Pin':
            ax.plot(x_pos, -0.2, marker='^', color='green', markersize=12)
        elif s['type'] == 'Roller':
            ax.plot(x_pos, -0.2, marker='o', color='green', markersize=10)
        elif s['type'] == 'Fixed':
            rect = patches.Rectangle((x_pos-0.1, -0.4), 0.2, 0.8, color='green', alpha=0.5)
            ax.add_patch(rect)
            
    # Draw Loads
    max_load = 1
    if loads: max_load = max([l['mag'] for l in loads])
    
    for l in loads:
        x_start = node_x[int(l['span_idx'])] + l['x']
        mag_scale = (l['mag'] / max_load) * 1.0 # Scale arrow size
        
        if l['type'] == 'P':
            ax.arrow(x_start, 1.0, 0, -0.8, head_width=0.2, head_length=0.2, fc='red', ec='red')
            ax.text(x_start, 1.1, f"P={l['mag']}", ha='center', color='red')
        elif l['type'] == 'U':
            dist = l.get('dist', spans[int(l['span_idx'])] - l['x'])
            rect = patches.Rectangle((x_start, 0), dist, 0.5, facecolor='orange', alpha=0.3, edgecolor='orange')
            ax.add_patch(rect)
            ax.text(x_start + dist/2, 0.6, f"w={l['mag']}", ha='center', color='orange')
        elif l['type'] == 'M':
            # Draw a curved arrow
            style = "Simple,tail_width=0.5,head_width=4,head_length=8"
            kw = dict(arrowstyle=style, color="purple")
            arc = patches.FancyArrowPatch((x_start-0.2, 0.5), (x_start+0.2, 0.5), connectionstyle="arc3,rad=.5", **kw)
            ax.add_patch(arc)
            ax.text(x_start, 0.8, f"M={l['mag']}", ha='center', color='purple')

# --- MAIN CALCULATION ---
if st.button("🚀 Calculate Analysis", type="primary"):
    if not spans:
        st.error("Please enter span lengths.")
    else:
        try:
            # Prepare Data for Solver
            beam_props = {'E': E, 'I': I, 'A': A, 'type': beam_type}
            
            # Call Solver
            solver = BeamSolver(spans, st.session_state.supports_list, st.session_state.loads_list, beam_props)
            results, R = solver.solve()
            
            # --- PLOTTING (RESTORED STYLE) ---
            st.success("Analysis Complete!")
            
            # Create 4 Subplots (Diagram, V, M, D)
            fig, ax = plt.subplots(4, 1, figsize=(10, 14), gridspec_kw={'height_ratios': [1, 2, 2, 2]})
            
            # 1. Physical Diagram
            plot_beam_diagram(ax[0], spans, st.session_state.supports_list, st.session_state.loads_list)
            ax[0].set_title("Beam Structure Diagram")
            
            # 2. Shear Force
            ax[1].plot(results['x'], results['shear'], 'b-', linewidth=1.5)
            ax[1].fill_between(results['x'], results['shear'], color='blue', alpha=0.1)
            ax[1].set_ylabel("Shear (N)")
            ax[1].grid(True, linestyle=':', alpha=0.6)
            ax[1].axhline(0, color='black', linewidth=0.8)
            
            # 3. Bending Moment (Inverted for Civil Engineering convention usually, but standard here)
            ax[2].plot(results['x'], results['moment'], 'r-', linewidth=1.5)
            ax[2].fill_between(results['x'], results['moment'], color='red', alpha=0.1)
            ax[2].set_ylabel("Moment (N-m)")
            ax[2].grid(True, linestyle=':', alpha=0.6)
            ax[2].axhline(0, color='black', linewidth=0.8)

            # 4. Deflection
            # *CRITICAL FIX FOR SMOOTH PLOT*: The solver returns sorted X, so plot should be smooth.
            ax[3].plot(results['x'], results['deflection'], 'g-', linewidth=2)
            ax[3].set_ylabel("Deflection (m)")
            ax[3].set_xlabel("Position (m)")
            ax[3].grid(True, linestyle=':', alpha=0.6)
            ax[3].axhline(0, color='black', linewidth=0.8)
            
            # Mark max deflection
            min_y = results['deflection'].min()
            min_idx = results['deflection'].idxmin()
            if not np.isnan(min_y):
                ax[3].plot(results.iloc[min_idx]['x'], min_y, 'ko')
                ax[3].text(results.iloc[min_idx]['x'], min_y, f" Max: {min_y:.2e} m", va='top')

            plt.tight_layout()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error: {e}")
