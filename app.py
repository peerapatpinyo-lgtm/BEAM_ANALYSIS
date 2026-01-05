import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from solver import BeamSolver  # เรียกใช้ Solver ตัวใหม่

st.set_page_config(page_title="Professional Beam Analysis", layout="wide")

# --- HEADER ---
st.title("🏗️ Professional Beam Analysis (FEM)")
st.markdown("""
**Features:** * Auto-Meshing (High Accuracy) 
* Timoshenko / Euler-Bernoulli Theory
* Support Settlement & Spring Supports
""")

# --- SIDEBAR: GLOBAL SETTINGS ---
with st.sidebar:
    st.header("⚙️ Material & Properties")
    E = st.number_input("Elastic Modulus (E) [Pa]", value=2.0e11, format="%.2e")
    I = st.number_input("Moment of Inertia (I) [m^4]", value=1.0e-4, format="%.2e")
    
    # Advanced Options
    st.markdown("---")
    beam_type = st.selectbox("Beam Theory", ["Euler", "Timoshenko"])
    
    # Show Area input only if Timoshenko is selected (or just show it generally)
    if beam_type == "Timoshenko":
        st.info("Timoshenko beam requires Cross-sectional Area.")
        A = st.number_input("Cross-sectional Area (A) [m^2]", value=0.01, format="%.4f")
    else:
        A = 0.01 # Default value

# --- MAIN: INPUTS ---

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Geometry")
    spans_input = st.text_input("Span Lengths (m) - comma separated", "5, 5")
    try:
        spans = [float(x.strip()) for x in spans_input.split(',')]
        n_nodes = len(spans) + 1
        total_length = sum(spans)
        st.success(f"Total Length: {total_length} m | Nodes: {n_nodes}")
    except:
        st.error("Invalid span input")
        spans = []

with col2:
    st.info("💡 **Tip:** Define Supports and Loads using the tables below.")

# --- DATA EDITORS FOR SUPPORTS & LOADS ---
st.subheader("2. Supports & Boundary Conditions")
# Default Supports Data
default_supports = pd.DataFrame([
    {"Node ID": 0, "Type": "Pin", "Settlement (m)": 0.0, "Spring K (N/m)": 0.0},
    {"Node ID": len(spans), "Type": "Roller", "Settlement (m)": 0.0, "Spring K (N/m)": 0.0}
])

support_config = {
    "Type": st.column_config.SelectboxColumn(options=["Pin", "Roller", "Fixed"]),
    "Node ID": st.column_config.NumberColumn(min_value=0, max_value=len(spans), step=1),
    "Settlement (m)": st.column_config.NumberColumn(help="Vertical settlement (down is positive)", format="%.4f"),
    "Spring K (N/m)": st.column_config.NumberColumn(help="Spring stiffness (0 for rigid)", format="%.2e")
}

edited_supports = st.data_editor(
    default_supports, 
    column_config=support_config, 
    num_rows="dynamic", 
    use_container_width=True,
    key="supports_editor"
)

st.subheader("3. Loads")
# Default Loads Data
default_loads = pd.DataFrame([
    {"Span Index": 0, "Type": "U", "Magnitude": 10000.0, "Location (x)": 0.0, "Dist (UDL only)": 5.0},
])

load_config = {
    "Type": st.column_config.SelectboxColumn(
        options=["P", "U", "M"], 
        help="P: Point Load, U: Uniform Load, M: Moment"
    ),
    "Span Index": st.column_config.NumberColumn(min_value=0, max_value=len(spans)-1, step=1),
    "Magnitude": st.column_config.NumberColumn(help="Positive = Downward Force / Counter-Clockwise Moment"),
    "Location (x)": st.column_config.NumberColumn(help="Distance from left of span"),
    "Dist (UDL only)": st.column_config.NumberColumn(help="Length of UDL coverage")
}

edited_loads = st.data_editor(
    default_loads, 
    column_config=load_config, 
    num_rows="dynamic", 
    use_container_width=True,
    key="loads_editor"
)

# --- CALCULATION LOGIC ---
if st.button("🚀 Analyze Structure", type="primary"):
    if not spans:
        st.error("Please define span lengths.")
    else:
        try:
            # 1. Prepare Properties
            beam_props = {
                'E': E,
                'I': I,
                'A': A,
                'type': beam_type
            }

            # 2. Parse Supports from DataFrame
            supports_list = []
            for _, row in edited_supports.iterrows():
                supports_list.append({
                    'id': int(row['Node ID']),
                    'type': row['Type'],
                    'settlement': float(row['Settlement (m)']),
                    'k_spring': float(row['Spring K (N/m)'])
                })

            # 3. Parse Loads from DataFrame
            loads_list = []
            for _, row in edited_loads.iterrows():
                load_data = {
                    'span_idx': int(row['Span Index']),
                    'type': row['Type'],
                    'mag': float(row['Magnitude']),
                    'x': float(row['Location (x)'])
                }
                if row['Type'] == 'U':
                    load_data['dist'] = float(row['Dist (UDL only)'])
                loads_list.append(load_data)

            # 4. INSTANTIATE SOLVER
            solver = BeamSolver(spans, supports_list, loads_list, beam_props)
            
            # 5. SOLVE
            results, R = solver.solve()

            # --- DISPLAY RESULTS ---
            st.success("Analysis Complete!")
            
            # Show Reactions
            st.subheader("Results: Reactions")
            # Map reactions back to readable format
            # Note: R vector index matches the mesh nodes. We need to filter only support nodes.
            # But for simplicity, we can just show the raw R vector mapped to Global X or improve mapping later.
            # Here let's just show max values or simply the diagrams.
            
            # Create Visualization
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
            
            # Shear Diagram
            ax1.plot(results['x'], results['shear'], color='blue', label='Shear (V)')
            ax1.fill_between(results['x'], results['shear'], color='blue', alpha=0.1)
            ax1.set_ylabel("Shear Force (N)")
            ax1.set_title("Shear Force Diagram")
            ax1.grid(True, linestyle='--', alpha=0.6)
            ax1.legend()

            # Moment Diagram (Invert Y for structural convention usually, but keeping math standard here)
            ax2.plot(results['x'], results['moment'], color='red', label='Moment (M)')
            ax2.fill_between(results['x'], results['moment'], color='red', alpha=0.1)
            ax2.set_ylabel("Bending Moment (N-m)")
            ax2.set_title("Bending Moment Diagram")
            ax2.grid(True, linestyle='--', alpha=0.6)
            ax2.legend()

            # Deflection Diagram
            ax3.plot(results['x'], results['deflection'], color='green', linewidth=2, label='Deflection (y)')
            ax3.set_ylabel("Deflection (m)")
            ax3.set_xlabel("Position (m)")
            ax3.set_title(f"Deflection Diagram ({beam_type})")
            ax3.grid(True, linestyle='--', alpha=0.6)
            
            # Mark Supports on Deflection Graph
            mesh_nodes = solver.nodes # Access nodes from solver
            ax3.scatter(mesh_nodes, np.zeros_like(mesh_nodes), color='black', marker='^', s=50, label='Nodes/Supports')
            ax3.legend()

            st.pyplot(fig)
            
            # Data Table Download
            with st.expander("See Detailed Data Points"):
                st.dataframe(results)

        except Exception as e:
            st.error(f"Analysis Failed: {str(e)}")
            st.warning("Please check if the structure is stable (e.g., not enough supports).")
