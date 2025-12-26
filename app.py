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
        current_x = 0
        for i, L in enumerate(self.spans):
            # Stiffness Matrix for Beam Element
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

        # 2. Apply Loads (Fixed End Forces -> Nodal Forces)
        # Allows multiple loads per span (Superposition)
        for load in self.loads:
            span_idx = load['span_idx']
            L = self.spans[span_idx]
            
            # Node Indices
            n1 = span_idx
            n2 = span_idx + 1
            idx = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
            
            fem = np.zeros(4)
            
            if load['type'] == 'Uniform':
                w = load['total_w'] # N/m
                # Fixed End Reactions (Upwards +, CCW +)
                Ry = (w * L) / 2.0
                M = (w * L**2) / 12.0
                
                # Equivalent Nodal Forces = - (Fixed End Actions)
                fem[0] = -Ry      # Fy Node 1
                fem[1] = -M       # M Node 1
                fem[2] = -Ry      # Fy Node 2
                fem[3] = +M       # M Node 2
                
            elif load['type'] == 'Point':
                P = load['total_w'] # N
                a = load['pos']     # Distance from left node
                b = L - a
                
                # Fixed End Moments
                M1 = (P * a * b**2) / (L**2)
                M2 = (P * a**2 * b) / (L**2)
                
                # Fixed End Shears
                R1 = (P * b**2 * (3*a + b)) / (L**3)
                R2 = (P * a**2 * (a + 3*b)) / (L**3)
                
                fem[0] = -R1
                fem[1] = -M1
                fem[2] = -R2
                fem[3] = +M2

            # Add to Global Force Vector (Superposition)
            self.F[idx] += fem

        # 3. Apply Boundary Conditions
        fixed_dof = []
        for i, supp in enumerate(self.supports):
            if supp in ['Pin', 'Roller']:
                fixed_dof.append(2*i)       # Fix Y translation
            elif supp == 'Fix':
                fixed_dof.append(2*i)       # Fix Y translation
                fixed_dof.append(2*i+1)     # Fix Rotation
        
        free_dof = [x for x in range(self.dof) if x not in fixed_dof]
        
        # Partition and Solve
        K_ff = self.K[np.ix_(free_dof, free_dof)]
        F_f = self.F[free_dof]
        
        try:
            u_f = np.linalg.solve(K_ff, F_f)
        except np.linalg.LinAlgError:
            return None, "Structure Unstable"
            
        self.u = np.zeros(self.dof)
        self.u[free_dof] = u_f
        
        # Calculate Reactions
        self.R = self.K @ self.u - self.F
        return self.u, None

    def get_internal_forces(self, num_points=100):
        # Calculate using Section Method (Statics)
        x_total_coords = []
        shear_vals = []
        moment_vals = []
        
        # Map reactions
        node_reactions_y = {i: self.R[2*i] for i in range(self.nodes)}
        node_reactions_m = {i: self.R[2*i+1] for i in range(self.nodes)}
            
        current_x_start = 0
        
        for i, L in enumerate(self.spans):
            x_local = np.linspace(0, L, num_points)
            x_global = current_x_start + x_local
            
            v_span = []
            m_span = []
            
            for x_curr in x_global:
                V, M = 0.0, 0.0
                
                # 1. Reactions to the left
                for node_i in range(i + 1):
                    rx = sum(self.spans[:node_i])
                    if rx <= x_curr + 1e-6:
                        ry = node_reactions_y[node_i]
                        rm = node_reactions_m[node_i]
                        V += ry
                        M += -rm + ry * (x_curr - rx)
                
                # 2. Loads to the left
                for load in self.loads:
                    start_load_x = sum(self.spans[:load['span_idx']])
                    
                    if load['type'] == 'Point':
                        px = start_load_x + load['pos']
                        if px <= x_curr:
                            P = load['total_w']
                            V -= P
                            M -= P * (x_curr - px)
                            
                    elif load['type'] == 'Uniform':
                        lx_start = start_load_x
                        lx_end = start_load_x + self.spans[load['span_idx']]
                        
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
            'shear': np.array(shear_vals) / 1000.0,
            'moment': np.array(moment_vals) / 1000.0
        })

# ==========================================
# 🎨 PART 2: UI & LOGIC
# ==========================================
st.set_page_config(page_title="Beam Calc Pro", layout="wide")

st.title("🏗️ Continuous Beam Analysis (Pro)")
st.caption("Engine: Custom Direct Stiffness Method | Features: Superposition & Tension-Side BMD")

# --- INPUTS ---
col1, col2 = st.columns([1, 2])

with col1:
    n_span = st.number_input("Number of Spans", 1, 6, 2)
    spans = []
    st.subheader("Geometry (m)")
    for i in range(n_span):
        spans.append(st.number_input(f"Span {i+1}", 1.0, 20.0, 4.0, key=f"s_{i}"))

with col2:
    st.subheader("Supports")
    supports = []
    cols = st.columns(n_span + 1)
    opts = ['Pin', 'Roller', 'Fix']
    for i in range(n_span + 1):
        def_idx = 0 if i==0 else 1
        supports.append(cols[i].selectbox(f"S{i+1}", opts, index=def_idx, key=f"sup_{i}"))
        
    st.subheader("Loads (1.4DL + 1.7LL)")
    loads_input = []
    
    # NEW: Loop to allow BOTH Uniform and Point loads on same span
    for i in range(n_span):
        with st.expander(f"📍 Loads on Span {i+1}", expanded=True):
            c_u, c_p = st.columns(2)
            
            # 1. Uniform Load Inputs
            with c_u:
                st.markdown("**Uniform Load (w)**")
                u_dl = st.number_input(f"U-DL (kN/m)", 0.0, key=f"udl_{i}")
                u_sdl = st.number_input(f"U-SDL (kN/m)", 0.0, key=f"usdl_{i}")
                u_ll = st.number_input(f"U-LL (kN/m)", 0.0, key=f"ull_{i}")
                
                if (u_dl + u_sdl + u_ll) > 0:
                    w_total = (1.4 * (u_dl + u_sdl) + 1.7 * u_ll) * 1000.0 # N/m
                    loads_input.append({
                        'span_idx': i, 'type': 'Uniform', 'total_w': w_total
                    })

            # 2. Point Load Inputs
            with c_p:
                st.markdown("**Point Load (P)**")
                p_dl = st.number_input(f"P-DL (kN)", 0.0, key=f"pdl_{i}")
                p_ll = st.number_input(f"P-LL (kN)", 0.0, key=f"pll_{i}")
                p_pos = st.slider(f"Position (m)", 0.0, spans[i], spans[i]/2, key=f"ppos_{i}")
                
                if (p_dl + p_ll) > 0:
                    p_total = (1.4 * p_dl + 1.7 * p_ll) * 1000.0 # N
                    loads_input.append({
                        'span_idx': i, 'type': 'Point', 'total_w': p_total, 'pos': p_pos
                    })

if st.button("🚀 Calculate", type="primary"):
    solver = SimpleBeamSolver(spans, supports, loads_input)
    u, err = solver.solve()
    
    if err:
        st.error(f"Error: {err}")
    else:
        df = solver.get_internal_forces(num_points=50)
        
        m_max = df['moment'].max()
        m_min = df['moment'].min()
        v_max = df['shear'].abs().max()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Max Moment (+)", f"{m_max:.2f} kN-m")
        c2.metric("Max Moment (-)", f"{m_min:.2f} kN-m")
        c3.metric("Max Shear", f"{v_max:.2f} kN")
        
        # --- SFD Plot ---
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(
            x=df['x'], y=df['shear'], fill='tozeroy', 
            line=dict(color='#FF4B4B'), name='Shear'
        ))
        fig_v.update_layout(
            title="Shear Force Diagram (SFD)", 
            yaxis_title="Shear (kN)", hovermode="x"
        )
        st.plotly_chart(fig_v, use_container_width=True)
        
        # --- BMD Plot (Tension Side) ---
        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(
            x=df['x'], y=df['moment'], fill='tozeroy', 
            line=dict(color='#1E88E5'), name='Moment'
        ))
        
        # KEY CHANGE: Reverse Y-Axis for Tension Side Plotting
        # Positive Moment (Sagging) -> Down
        # Negative Moment (Hogging) -> Up
        fig_m.update_layout(
            title="Bending Moment Diagram (BMD) - Tension Side", 
            yaxis_title="Moment (kN-m)",
            yaxis=dict(autorange="reversed"), # <--- Tension Side Trick
            hovermode="x"
        )
        # Add annotation for convention
        fig_m.add_annotation(
            text="Values plotted on Tension Side (Sagging + is Down)",
            xref="paper", yref="paper", x=0.5, y=1.1, showarrow=False,
            font=dict(size=12, color="gray")
        )
        
        st.plotly_chart(fig_m, use_container_width=True)

        # --- Design Section ---
        st.divider()
        st.subheader("📝 Design Requirements")
        des_mu = max(abs(m_max), abs(m_min))
        
        # Quick Calc
        fc, fy = 24, 400
        b, h, cov = 25, 50, 4
        d = (h-cov)/100
        phi_b = 0.9
        Mn = (des_mu * 1000) / phi_b
        Rn = Mn / ((b/100) * d**2)
        m_ratio = fy/(0.85*fc)
        
        term = 1 - (2*m_ratio*Rn)/(fy*1e6)
        rho = 0
        if term >= 0:
            rho = (1/m_ratio)*(1 - np.sqrt(term))
        
        As = rho * (b/100) * d * 10000
        st.info(f"Design Mu: {des_mu:.2f} kN-m | Approx As: {As:.2f} cm² (using b={b}, h={h})")
