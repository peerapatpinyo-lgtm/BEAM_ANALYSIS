import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from solver import BeamSolver # ตรวจสอบให้แน่ใจว่าไฟล์ solver.py อยู่โฟลเดอร์เดียวกัน

# --- Page Configuration ---
st.set_page_config(page_title="Beam Analysis Pro", layout="wide")

st.title("🏗️ Beam Analysis Pro (Timoshenko & FEM)")
st.markdown("---")

# --- Sidebar: Material & Section Properties ---
st.sidebar.header("1. Material & Section")

E = st.sidebar.number_input("Elastic Modulus (E) [Pa]", value=2e11, format="%.2e")
I = st.sidebar.number_input("Moment of Inertia (I) [m^4]", value=5e-5, format="%.2e")

# Timoshenko Inputs (Optional)
use_timoshenko = st.sidebar.checkbox("Advanced: Timoshenko Inputs", value=False)
if use_timoshenko:
    A = st.sidebar.number_input("Cross-sectional Area (A) [m^2]", value=0.01, format="%.4f")
    G = st.sidebar.number_input("Shear Modulus (G) [Pa]", value=7.7e10, format="%.2e")
else:
    A, G = None, None # ให้ Solver คำนวณ Default เอง

# --- Main Interface: Geometry ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("2. Geometry Setup")
    num_spans = st.number_input("Number of Spans", min_value=1, max_value=10, value=2)
    
    spans = []
    for i in range(num_spans):
        l = st.number_input(f"Span {i+1} Length (m)", min_value=0.1, value=5.0, key=f"span_{i}")
        spans.append(l)

with col2:
    st.subheader("3. Supports & Loads")
    
    # Supports
    with st.expander("Support Configuration", expanded=True):
        num_nodes = num_spans + 1
        support_data = []
        
        # Default supports: Pin at start, Roller at others
        cols = st.columns(num_nodes)
        for i in range(num_nodes):
            default_type = "Pin" if i == 0 else "Roller"
            sType = cols[i].selectbox(f"Node {i}", ["None", "Pin", "Roller", "Fixed"], index=["None", "Pin", "Roller", "Fixed"].index(default_type))
            if sType != "None":
                support_data.append({"id": i, "type": sType})
    
    # Loads
    with st.expander("Load Configuration", expanded=True):
        if 'load_list' not in st.session_state:
            st.session_state.load_list = []
        
        # Input Form
        c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
        l_type = c1.selectbox("Type", ["Point (P)", "Uniform (U)"])
        l_span = c2.number_input("Span Index", 0, num_spans-1, 0)
        l_mag = c3.number_input("Mag (N, N/m)", value=1000.0)
        l_x = c4.number_input("Dist x (m)", 0.0, 100.0, 2.5)
        
        l_dist = None
        if l_type == "Uniform (U)":
            l_dist = c5.number_input("Length (m)", 0.1, 100.0, 2.5)
        else:
            c5.write("") # Spacer
            
        if st.button("Add Load"):
            st.session_state.load_list.append({
                "type": "P" if "Point" in l_type else "U",
                "span_idx": l_span,
                "mag": l_mag,
                "x": l_x,
                "dist": l_dist
            })

        # Display Loads
        if st.session_state.load_list:
            st.write("Current Loads:")
            st.table(pd.DataFrame(st.session_state.load_list))
            if st.button("Clear Loads"):
                st.session_state.load_list = []

# --- Analysis ---
st.markdown("---")
if st.button("🚀 Analyze Beam", type="primary"):
    
    # Prepare Data
    supports_df = pd.DataFrame(support_data)
    loads_df = pd.DataFrame(st.session_state.load_list)
    
    # Instantiate Solver
    solver = BeamSolver(spans, supports_df, loads_df, E, I, A, G)
    
    # --- HERE IS THE FIX: Unpack 3 values ---
    results, reactions, summary = solver.solve()
    
    if results.empty:
        st.error("Structure is Unstable! Please check supports.")
    else:
        # 1. Critical Values Summary (New Feature)
        st.subheader("📊 Critical Design Values")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Max Shear", f"{summary['V_max']['value']:.2f} N", f"@ {summary['V_max']['x']:.2f} m")
        m2.metric("Max Moment (+)", f"{summary['M_pos']['value']:.2f} Nm", f"@ {summary['M_pos']['x']:.2f} m")
        m3.metric("Max Moment (-)", f"{summary['M_neg']['value']:.2f} Nm", f"@ {summary['M_neg']['x']:.2f} m")
        m4.metric("Max Deflection", f"{summary['D_max']['value']*1000:.4f} mm", f"@ {summary['D_max']['x']:.2f} m")
        
        st.markdown("---")

        # 2. Plotting
        st.subheader("📈 Diagrams")
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        x = results['x']
        
        # Shear
        ax1.plot(x, results['shear'], color='tab:blue', linewidth=2)
        ax1.fill_between(x, results['shear'], color='tab:blue', alpha=0.1)
        ax1.set_ylabel("Shear Force (N)")
        ax1.set_title("Shear Force Diagram (SFD)")
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Moment (Invert Y for Engineering Convention)
        ax2.plot(x, results['moment'], color='tab:red', linewidth=2)
        ax2.fill_between(x, results['moment'], color='tab:red', alpha=0.1)
        ax2.set_ylabel("Bending Moment (Nm)")
        ax2.set_title("Bending Moment Diagram (BMD)")
        ax2.invert_yaxis() 
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        # Deflection
        ax3.plot(x, results['deflection']*1000, color='tab:green', linewidth=2) # Convert to mm
        ax3.set_ylabel("Deflection (mm)")
        ax3.set_xlabel("Position (m)")
        ax3.set_title("Deflection Curve")
        ax3.invert_yaxis()
        ax3.grid(True, linestyle='--', alpha=0.6)
        
        # Draw Supports on plots (Visual Aid)
        total_len = sum(spans)
        current_x = 0
        nodes_x = [0]
        for l in spans:
            current_x += l
            nodes_x.append(current_x)
            
        for nx in nodes_x:
            for ax in [ax1, ax2, ax3]:
                ax.axvline(nx, color='black', linestyle=':', linewidth=0.8)

        st.pyplot(fig)
        
        # 3. Data Tables
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Reactions")
            # Format reaction vector for display
            r_display = []
            for i in range(num_nodes):
                r_display.append({
                    "Node": i, 
                    "Fy (N)": reactions[2*i], 
                    "Mz (Nm)": reactions[2*i+1]
                })
            st.dataframe(pd.DataFrame(r_display))
            
        with c2:
            st.subheader("Detailed Results (Head)")
            st.dataframe(results.head(10))
