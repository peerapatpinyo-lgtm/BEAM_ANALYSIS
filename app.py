import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from solver import BeamSolver

st.set_page_config(page_title="Beam Analysis", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 5px; }
    .reportview-container .main .block-container { max-width: 1000px; }
</style>
""", unsafe_allow_html=True)

st.title("🏗️ Beam Analysis (Classic View)")

# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.header("⚙️ Properties")
    E = st.number_input("Elastic Modulus (E) [Pa]", value=2.0e11, format="%.2e")
    I = st.number_input("Moment of Inertia (I) [m^4]", value=1.0e-4, format="%.2e")
    
    st.markdown("---")
    beam_type = st.selectbox("Theory", ["Euler", "Timoshenko"])
    if beam_type == "Timoshenko":
        A = st.number_input("Cross-sectional Area (A) [m^2]", value=0.01, format="%.4f")
    else:
        A = 0.01

# --- INPUTS ---
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("1. Spans")
    spans_input = st.text_input("Enter span lengths (m, comma separated)", "5, 5")
    try:
        spans = [float(x.strip()) for x in spans_input.split(',')]
        node_locs = np.concatenate(([0], np.cumsum(spans)))
    except:
        spans = []
        node_locs = []

with col2:
    if spans:
        st.info(f"Total Length: {sum(spans)} m | Nodes: {len(node_locs)}")

st.markdown("---")

# --- SUPPORTS & LOADS ---
c1, c2 = st.columns(2)

with c1:
    st.subheader("2. Supports")
    if 'supports_list' not in st.session_state:
        st.session_state.supports_list = [
            {'id': 0, 'type': 'Pin', 'settlement': 0.0, 'k_spring': 0.0},
            {'id': len(spans), 'type': 'Roller', 'settlement': 0.0, 'k_spring': 0.0}
        ]
    
    with st.expander("Add/Edit Supports", expanded=True):
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

    if st.session_state.supports_list:
        df_sup = pd.DataFrame(st.session_state.supports_list)
        if 'settlement' not in df_sup.columns: df_sup['settlement'] = 0.0
        st.dataframe(df_sup[['id', 'type', 'settlement']], hide_index=True, use_container_width=True)
        if st.button("Clear Supports"):
            st.session_state.supports_list = []
            st.rerun()

with c2:
    st.subheader("3. Loads")
    if 'loads_list' not in st.session_state:
        st.session_state.loads_list = [{'span_idx': 0, 'type': 'U', 'mag': 10000, 'x': 0, 'dist': 5}]

    with st.expander("Add Loads", expanded=True):
        if spans:
            l_span = st.selectbox("Span Index", range(len(spans)))
            l_type = st.selectbox("Load Type", ["Point (P)", "Uniform (U)", "Moment (M)"])
            l_mag = st.number_input("Magnitude", value=1000.0)
            l_x = st.number_input("Location (x)", value=2.5)
            l_dist = st.number_input("Dist (UDL)", value=1.0) if "Uniform" in l_type else 0.0
            
            if st.button("Add Load"):
                l_code = 'P' if "Point" in l_type else ('U' if "Uniform" in l_type else 'M')
                new_load = {'span_idx': l_span, 'type': l_code, 'mag': l_mag, 'x': l_x}
                if l_code == 'U': new_load['dist'] = l_dist
                st.session_state.loads_list.append(new_load)
                st.rerun()

    if st.session_state.loads_list:
        df_load = pd.DataFrame(st.session_state.loads_list)
        st.dataframe(df_load, hide_index=True, use_container_width=True)
        if st.button("Clear Loads"):
            st.session_state.loads_list = []
            st.rerun()

# --- HELPER: PLOT BEAM SCHEMATIC ---
def plot_beam_diagram(ax, spans, supports, loads):
    total_len = sum(spans)
    node_x = np.concatenate(([0], np.cumsum(spans)))
    
    # Beam Line
    ax.plot([0, total_len], [0, 0], 'k-', linewidth=4, solid_capstyle='round')
    
    # Nodes
    ax.scatter(node_x, np.zeros_like(node_x), color='white', edgecolor='black', zorder=10, s=30)
    
    # Supports
    for s in supports:
        if int(s['id']) < len(node_x):
            x = node_x[int(s['id'])]
            if s['type'] == 'Pin':
                ax.plot(x, -0.3, marker='^', color='#444', markersize=14)
            elif s['type'] == 'Roller':
                ax.plot(x, -0.3, marker='o', color='#444', markersize=12)
            elif s['type'] == 'Fixed':
                ax.add_patch(patches.Rectangle((x-0.1, -0.6), 0.2, 1.2, color='#444'))
    
    # Loads
    max_load = max([abs(l['mag']) for l in loads]) if loads else 1
    for l in loads:
        if int(l['span_idx']) < len(spans):
            x = node_x[int(l['span_idx'])] + l['x']
            mag = l['mag']
            # Scale arrows visually
            if l['type'] == 'P':
                dy = -0.8 if mag > 0 else 0.8
                ax.arrow(x, -dy, 0, dy*0.8, head_width=0.15, head_length=0.2, fc='red', ec='red', linewidth=2)
                ax.text(x, -dy*1.2, f"P={mag}", ha='center', color='red', fontweight='bold')
            elif l['type'] == 'U':
                dist = l.get('dist', 1.0)
                ax.add_patch(patches.Rectangle((x, 0.1), dist, 0.4, facecolor='orange', alpha=0.5))
                ax.text(x + dist/2, 0.8, f"w={mag}", ha='center', color='orange', fontweight='bold')
            elif l['type'] == 'M':
                ax.text(x, 0.5, f"M={mag}", ha='center', color='purple', fontweight='bold')
                
    ax.set_ylim(-2, 2)
    ax.set_xlim(-0.5, total_len + 0.5)
    ax.axis('off')
    ax.set_title("System Diagram", loc='left', fontsize=12, fontweight='bold')

# --- PLOT HELPER: ANNOTATE MAX/MIN ---
def annotate_peaks(ax, x, y, color):
    # Find local peaks/valleys
    ymax = np.max(y)
    ymin = np.min(y)
    xmax = x[np.argmax(y)]
    xmin = x[np.argmin(y)]
    
    # Annotate Max
    ax.annotate(f"{ymax:.2f}", xy=(xmax, ymax), xytext=(xmax, ymax),
                textcoords="offset points", xycoords='data', ha='center', va='bottom',
                color=color, fontweight='bold', fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec=color))
    
    # Annotate Min
    ax.annotate(f"{ymin:.2f}", xy=(xmin, ymin), xytext=(xmin, ymin),
                textcoords="offset points", xycoords='data', ha='center', va='top',
                color=color, fontweight='bold', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec=color))

# --- MAIN LOGIC ---
if st.button("🚀 Calculate", type="primary"):
    if not spans:
        st.error("Please enter spans.")
    else:
        try:
            beam_props = {'E': E, 'I': I, 'A': A, 'type': beam_type}
            solver = BeamSolver(spans, st.session_state.supports_list, st.session_state.loads_list, beam_props)
            results, R = solver.solve()
            
            # --- PLOTTING ---
            st.success("Calculation Successful")
            
            fig, ax = plt.subplots(4, 1, figsize=(10, 16), sharex=False, gridspec_kw={'height_ratios': [1, 2, 2, 2]})
            
            # 1. System Diagram
            plot_beam_diagram(ax[0], spans, st.session_state.supports_list, st.session_state.loads_list)
            
            # 2. Shear Force (SFD)
            x_vals = results['x']
            v_vals = results['shear']
            ax[1].plot(x_vals, v_vals, color='#1f77b4', linewidth=2)
            ax[1].fill_between(x_vals, v_vals, 0, color='#1f77b4', alpha=0.2)
            ax[1].set_ylabel("Shear (N)", fontsize=10, fontweight='bold')
            ax[1].set_title("Shear Force Diagram (SFD)", loc='left', fontsize=11)
            ax[1].grid(True, linestyle='--', alpha=0.5)
            ax[1].axhline(0, color='black', linewidth=1)
            annotate_peaks(ax[1], x_vals, v_vals, '#1f77b4')
            
            # 3. Bending Moment (BMD) - INVERTED Y
            m_vals = results['moment']
            ax[2].plot(x_vals, m_vals, color='#d62728', linewidth=2)
            ax[2].fill_between(x_vals, m_vals, 0, color='#d62728', alpha=0.2)
            ax[2].set_ylabel("Moment (N·m)", fontsize=10, fontweight='bold')
            ax[2].set_title("Bending Moment Diagram (BMD)", loc='left', fontsize=11)
            ax[2].grid(True, linestyle='--', alpha=0.5)
            ax[2].axhline(0, color='black', linewidth=1)
            ax[2].invert_yaxis() # <--- INVERTED FOR CIVIL ENG STYLE
            annotate_peaks(ax[2], x_vals, m_vals, '#d62728')
            
            # 4. Deflection
            d_vals = results['deflection']
            ax[3].plot(x_vals, d_vals, color='#2ca02c', linewidth=2)
            ax[3].fill_between(x_vals, d_vals, 0, color='#2ca02c', alpha=0.1)
            ax[3].set_ylabel("Deflection (m)", fontsize=10, fontweight='bold')
            ax[3].set_xlabel("Length (m)", fontsize=10)
            ax[3].set_title(f"Deflection Curve ({beam_type})", loc='left', fontsize=11)
            ax[3].grid(True, linestyle='--', alpha=0.5)
            ax[3].axhline(0, color='black', linewidth=1)
            # Find max absolute deflection
            max_d_idx = np.argmax(np.abs(d_vals))
            max_d_val = d_vals[max_d_idx]
            max_d_x = x_vals[max_d_idx]
            ax[3].annotate(f"Max: {max_d_val:.4e} m", xy=(max_d_x, max_d_val), 
                           xytext=(0, 15 if max_d_val<0 else -15), textcoords="offset points",
                           ha='center', color='#2ca02c', fontweight='bold',
                           arrowprops=dict(arrowstyle="->", color='#2ca02c'))

            plt.tight_layout()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Computation Error: {e}")
