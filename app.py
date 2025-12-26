import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ==========================================
# 🧠 PART 1: THE CUSTOM BEAM ENGINE (MANUAL CALC)
# ==========================================
class SimpleBeamSolver:
    def __init__(self, spans, supports, loads, E=200e9, I=500e-6):
        self.spans = spans
        self.supports = supports
        self.loads = loads
        self.E = E
        self.I = I
        self.nodes = len(spans) + 1
        self.dof = 2 * self.nodes  # 2 DOF per node (Deflection Y, Rotation Theta)
        
        # Global Matrices
        self.K = np.zeros((self.dof, self.dof))
        self.F = np.zeros(self.dof)
        
    def solve(self):
        # 1. Assemble Stiffness Matrix (K)
        # --------------------------------
        current_x = 0
        for i, L in enumerate(self.spans):
            # Element connects node i and i+1
            # Indices in Global Matrix
            # Node i:   2*i (Y), 2*i+1 (Theta)
            # Node i+1: 2*(i+1) (Y), 2*(i+1)+1 (Theta)
            
            k = (self.E * self.I / L**3) * np.array([
                [12, 6*L, -12, 6*L],
                [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L],
                [6*L, 2*L**2, -6*L, 4*L**2]
            ])
            
            idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
            for r in range(4):
                for c in range(4):
                    self.K[idx[r], idx[c]] += k[r, c]

        # 2. Apply Loads (Fixed End Forces to Global F)
        # ---------------------------------------------
        # Load Vector F = F_external - F_equivalent
        # We assume no external nodal loads, only member loads
        
        for load in self.loads:
            span_idx = load['span_idx']
            L = self.spans[span_idx]
            w = load['total_w'] # Uniform Load (kN/m) -> Convert to N/m outside
            
            # Node Indices
            n1 = span_idx
            n2 = span_idx + 1
            idx = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
            
            fem = np.zeros(4)
            
            if load['type'] == 'Uniform Load':
                # FEM for Uniform Load w
                # Reaction Y: wL/2
                # Moment: wL^2/12
                # Vector: [Fy1, M1, Fy2, M2]
                # Force is DOWN (-), so Reaction is UP (+)
                # Equivalent Nodal Force is -Reaction (Logic: K u = F_ext + F_eq)
                # Let's stick to: F_total = F_nodal - F_fixed_end
                
                # Fixed End Actions (Forces exerted BY beam ON nodes)
                # Left Node: Up wL/2, Moment CCW (+) wL^2/12
                # Right Node: Up wL/2, Moment CW (-) wL^2/12
                
                # We need Equivalent Forces to ADD to F vector
                # This is SAME direction as Fixed End Reactions? No, Opposite.
                # Actually, standard FEM: Load Vector = - (Fixed End Actions)
                
                # Fixed End Reactions (Upwards +, CCW +)
                Ry = (w * L) / 2.0
                M = (w * L**2) / 12.0
                
                # Equivalent Nodal Forces (External forces that would cause same deformation)
                # Downward force w -> Nodal force is Down
                fem[0] = -Ry      # Force Y Node 1
                fem[1] = -M       # Moment Node 1
                fem[2] = -Ry      # Force Y Node 2
                fem[3] = +M       # Moment Node 2
                
            elif load['type'] == 'Point Load':
                P = load['total_w'] # Point Load (kN) -> N
                a = load['pos']     # Distance from left node
                b = L - a
                
                # Fixed End Moments
                M1 = (P * a * b**2) / (L**2)
                M2 = (P * a**2 * b) / (L**2)
                
                # Fixed End Shears
                R1 = (P * b**2 * (3*a + b)) / (L**3)
                R2 = (P * a**2 * (a + 3*b)) / (L**3)
                
                # Equivalent Nodal Forces (Opposite to reactions)
                fem[0] = -R1
                fem[1] = -M1
                fem[2] = -R2
                fem[3] = +M2

            # Add to Global Force Vector
            self.F[idx] += fem

        # 3. Apply Boundary Conditions
        # ----------------------------
        fixed_dof = []
        for i, supp in enumerate(self.supports):
            if supp in ['Pin', 'Roller']:
                fixed_dof.append(2*i)       # Fix Y translation
            elif supp == 'Fix':
                fixed_dof.append(2*i)       # Fix Y translation
                fixed_dof.append(2*i+1)     # Fix Rotation
        
        free_dof = [x for x in range(self.dof) if x not in fixed_dof]
        
        # Partition Matrices
        K_ff = self.K[np.ix_(free_dof, free_dof)]
        F_f = self.F[free_dof]
        
        # 4. Solve for Displacements
        # --------------------------
        try:
            u_f = np.linalg.solve(K_ff, F_f)
        except np.linalg.LinAlgError:
            return None, "Structure Unstable"
            
        # Reconstruct Full Displacement Vector
        self.u = np.zeros(self.dof)
        self.u[free_dof] = u_f
        
        # Calculate Reactions: R = K*u - F_equiv (Original Load Vector)
        self.R = self.K @ self.u - self.F
        
        return self.u, None

    def get_internal_forces(self, num_points=100):
        # Calculate V and M by "Section Method" using Reactions + Loads
        # This is more robust than using Shape Functions for internal forces
        
        x_total_coords = []
        shear_vals = []
        moment_vals = []
        
        # Map global reactions back to nodes
        node_reactions_y = {}
        node_reactions_m = {}
        for i in range(self.nodes):
            node_reactions_y[i] = self.R[2*i]
            node_reactions_m[i] = self.R[2*i+1]
            
        current_x_start = 0
        
        # Iterate span by span to avoid discontinuity issues
        for i, L in enumerate(self.spans):
            x_local = np.linspace(0, L, num_points)
            x_global = current_x_start + x_local
            
            v_span = []
            m_span = []
            
            for x_curr in x_global:
                # CUT SECTION AT x_curr
                # Sum Forces and Moments to the LEFT
                
                V = 0.0
                M = 0.0
                
                # 1. Reactions to the left
                for node_i in range(i + 1): # Nodes up to current span start
                    # Reaction Force (Up is positive)
                    rx = 0 # X pos of node
                    for k in range(node_i): rx += self.spans[k]
                    
                    if rx <= x_curr + 1e-6:
                        ry = node_reactions_y[node_i]
                        rm = node_reactions_m[node_i]
                        
                        V += ry
                        M += -rm + ry * (x_curr - rx) # Moment from Force + Concentrated Moment (Reaction)
                
                # 2. Loads to the left
                for load in self.loads:
                    # Determine Load Position
                    start_load_x = 0
                    for k in range(load['span_idx']): start_load_x += self.spans[k]
                    
                    if load['type'] == 'Point Load':
                        px = start_load_x + load['pos']
                        if px <= x_curr: # Load is to the left
                            P = load['total_w'] # Downward force
                            V -= P
                            M -= P * (x_curr - px)
                            
                    elif load['type'] == 'Uniform Load':
                        # Load covers from start_load_x to start_load_x + L_span
                        lx_start = start_load_x
                        lx_end = start_load_x + self.spans[load['span_idx']]
                        
                        # Overlap of load and current section
                        # We only care about the portion of load to the LEFT of x_curr
                        eff_start = lx_start
                        eff_end = min(x_curr, lx_end)
                        
                        if eff_end > eff_start:
                            w = load['total_w']
                            length = eff_end - eff_start
                            load_force = w * length
                            center_dist = x_curr - (eff_start + length/2.0)
                            
                            V -= load_force
                            M -= load_force * center_dist
                
                v_span.append(V)
                m_span.append(M)
                
            x_total_coords.extend(x_global)
            shear_vals.extend(v_span)
            moment_vals.extend(m_span)
            
            current_x_start += L
            
        return pd.DataFrame({
            'x': x_total_coords,
            'shear': np.array(shear_vals) / 1000.0, # Convert N -> kN
            'moment': np.array(moment_vals) / 1000.0 # Convert Nm -> kNm
        })

# ==========================================
# 🎨 PART 2: UI & LOGIC
# ==========================================
st.set_page_config(page_title="Manual Beam Calc", layout="wide")

st.title("🏗️ Continuous Beam (Custom Engine)")
st.caption("No external libraries. Pure Math Calculation. 100% Reliable.")

# --- INPUTS ---
col1, col2 = st.columns([1, 2])

with col1:
    n_span = st.number_input("Number of Spans", 1, 5, 2)
    spans = []
    for i in range(n_span):
        spans.append(st.number_input(f"Span {i+1} Length (m)", 1.0, 20.0, 4.0, key=f"s_{i}"))

with col2:
    st.write("Supports")
    supports = []
    cols = st.columns(n_span + 1)
    opts = ['Pin', 'Roller', 'Fix']
    for i in range(n_span + 1):
        def_idx = 0 if i==0 else 1
        supports.append(cols[i].selectbox(f"S{i+1}", opts, index=def_idx, key=f"sup_{i}"))
        
    st.write("Loads (Factored: 1.4(DL+SDL) + 1.7LL)")
    loads_input = []
    for i in range(n_span):
        with st.expander(f"Span {i+1} Loads", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            ltype = c1.selectbox("Type", ["Uniform Load", "Point Load"], key=f"lt_{i}")
            dl = c2.number_input("DL (kN)", 10.0, key=f"dl_{i}")
            sdl = c3.number_input("SDL (kN)", 2.0, key=f"sdl_{i}")
            ll = c4.number_input("LL (kN)", 5.0, key=f"ll_{i}")
            
            pos = 0.0
            if ltype == "Point Load":
                pos = st.slider("Position (m)", 0.0, spans[i], spans[i]/2, key=f"pos_{i}")
            
            # Combine Load Here (Passed to solver as N or N/m)
            w_total = (1.4 * (dl + sdl) + 1.7 * ll) * 1000.0 # Convert kN -> N
            
            loads_input.append({
                'span_idx': i,
                'type': ltype,
                'total_w': w_total,
                'pos': pos
            })

if st.button("🚀 Run Manual Calculation", type="primary"):
    # Initialize Solver
    solver = SimpleBeamSolver(spans, supports, loads_input)
    
    # Solve
    u, err = solver.solve()
    
    if err:
        st.error(f"Calculation Error: {err} (Structure is unstable)")
    else:
        # Get Data for Plotting
        df = solver.get_internal_forces(num_points=50) # 50 points per span
        
        # Display Max Values
        max_m_pos = df['moment'].max()
        max_m_neg = df['moment'].min()
        max_v = df['shear'].abs().max()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Max Moment (+)", f"{max_m_pos:.2f} kN-m")
        c2.metric("Max Moment (-)", f"{max_m_neg:.2f} kN-m")
        c3.metric("Max Shear", f"{max_v:.2f} kN")
        
        # Plot SFD
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(
            x=df['x'], y=df['shear'], 
            fill='tozeroy', mode='lines', 
            line=dict(color='#FF4B4B', width=2),
            name='Shear (kN)'
        ))
        fig_v.update_layout(
            title="Shear Force Diagram (SFD)", 
            xaxis_title="Distance (m)", 
            yaxis_title="Shear (kN)",
            hovermode="x"
        )
        st.plotly_chart(fig_v, use_container_width=True)
        
        # Plot BMD
        fig_m = go.Figure()
        # Invert Moment for Civil Engineering convention (Optional, but here we plot normal)
        # Usually engineers like (-) moment on top. Let's stick to standard math plot first.
        fig_m.add_trace(go.Scatter(
            x=df['x'], y=df['moment'], 
            fill='tozeroy', mode='lines', 
            line=dict(color='#1E88E5', width=2),
            name='Moment (kN-m)'
        ))
        fig_m.update_layout(
            title="Bending Moment Diagram (BMD)", 
            xaxis_title="Distance (m)", 
            yaxis_title="Moment (kN-m)",
            hovermode="x"
        )
        st.plotly_chart(fig_m, use_container_width=True)

        # Design Report Logic (Brief)
        st.divider()
        st.header("📝 Design Check")
        des_mu = max(abs(max_m_pos), abs(max_m_neg))
        des_vu = max_v
        
        # Hardcoded Design Params for Quick Check
        fc, fy = 24, 400
        b, h, cover = 25, 50, 4
        d = (h-cover)/100
        
        # Design Calc
        phi_b, phi_v = 0.9, 0.85
        Mn_req = (des_mu * 1000) / phi_b
        Rn = Mn_req / ((b/100) * d**2)
        m = fy/(0.85*fc)
        rho = (1/m)*(1 - np.sqrt(1 - (2*m*Rn)/(fy*1e6))) if (1 - (2*m*Rn)/(fy*1e6)) >= 0 else 0
        As_req = rho * (b/100) * d * 10000
        
        st.write(f"**Design Moment (Mu):** {des_mu:.2f} kN-m")
        st.write(f"**Required Reinforcement (As):** {As_req:.2f} cm² (Approx)")
