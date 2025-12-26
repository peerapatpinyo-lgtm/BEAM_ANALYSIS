import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ==========================================
# 🧠 PART 1: THE CUSTOM BEAM ENGINE
# ==========================================
class SimpleBeamSolver:
    def __init__(self, spans, supports, loads, E=200e9, I=500e-6):
        self.spans = spans
        self.supports = supports
        self.loads = loads
        self.E = E
        self.I = I
        self.nodes = len(spans) + 1
        self.dof = 2 * self.nodes
        
        # Global Matrices
        self.K = np.zeros((self.dof, self.dof))
        self.F = np.zeros(self.dof)
        
    def solve(self):
        # 1. Assemble Stiffness Matrix (K)
        for i, L in enumerate(self.spans):
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

        # 2. Apply Loads (Superposition)
        for load in self.loads:
            span_idx = load['span_idx']
            L = self.spans[span_idx]
            
            n1, n2 = span_idx, span_idx + 1
            idx = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
            fem = np.zeros(4)
            
            if load['type'] == 'Uniform':
                w = load['total_w']
                Ry = (w * L) / 2.0
                M = (w * L**2) / 12.0
                fem = np.array([-Ry, -M, -Ry, M])
                
            elif load['type'] == 'Point':
                P = load['total_w']
                a = load['pos']
                b = L - a
                M1 = (P * a * b**2) / (L**2)
                M2 = (P * a**2 * b) / (L**2)
                R1 = (P * b**2 * (3*a + b)) / (L**3)
                R2 = (P * a**2 * (a + 3*b)) / (L**3)
                fem = np.array([-R1, -M1, -R2, M2])

            self.F[idx] += fem

        # 3. Boundary Conditions
        fixed_dof = []
        for i, supp in enumerate(self.supports):
            if supp in ['Pin', 'Roller']: fixed_dof.append(2*i)
            elif supp == 'Fix': fixed_dof.extend([2*i, 2*i+1])
        
        free_dof = [x for x in range(self.dof) if x not in fixed_dof]
        
        # Solve
        K_ff = self.K[np.ix_(free_dof, free_dof)]
        F_f = self.F[free_dof]
        
        try:
            u_f = np.linalg.solve(K_ff, F_f)
        except np.linalg.LinAlgError:
            return None, "Structure Unstable"
            
        self.u = np.zeros(self.dof)
        self.u[free_dof] = u_f
        self.R = self.K @ self.u - self.F
        return self.u, None

    def get_internal_forces(self, num_points=100):
        x_total, v_vals, m_vals = [], [], []
        node_ry = {i: self.R[2*i] for i in range(self.nodes)}
        node_rm = {i: self.R[2*i+1] for i in range(self.nodes)}
            
        cur_x_start = 0
        for i, L in enumerate(self.spans):
            x_loc = np.linspace(0, L, num_points)
            x_glob = cur_x_start + x_loc
            
            for x_curr in x_glob:
                V, M = 0.0, 0.0
                # Reactions
                for ni in range(i + 1):
                    rx = sum(self.spans[:ni])
                    if rx <= x_curr + 1e-6:
                        V += node_ry[ni]
                        M += -node_rm[ni] + node_ry[ni]*(x_curr - rx)
                # Loads
                for load in self.loads:
                    lx_start = sum(self.spans[:load['span_idx']])
                    if load['type'] == 'Point':
                        px = lx_start + load['pos']
                        if px <= x_curr:
                            P = load['total_w']
                            V -= P
                            M -= P*(x_curr - px)
                    elif load['type'] == 'Uniform':
                        lx_s, lx_e = lx_start, lx_start + self.spans[load['span_idx']]
                        eff_s, eff_e = lx_s, min(x_curr, lx_e)
                        if eff_e > eff_s:
                            w = load['total_w']
                            force = w * (eff_e - eff_s)
                            dist = x_curr - (eff_s + (eff_e - eff_s)/2)
                            V -= force
                            M -= force * dist
                
                v_vals.append(V)
                m_vals.append(M)
            x_total.extend(x_glob)
            cur_x_start += L
            
        return pd.DataFrame({'x': x_total, 'shear': np.array(v_vals)/1000, 'moment': np.array(m_vals)/1000})

# ==========================================
# 🎨 PART 2: UI & PLOTTING
# ==========================================
st.set_page_config(page_title="Beam Pro V7", layout="wide")
st.title("🏗️ Continuous Beam Analysis (Pro V7)")
st.caption("Multiple Point Loads | Auto-Labeling | Custom Engine")

# --- HELPER: ADD ANNOTATIONS ---
def add_peak_labels(fig, x_data, y_data, inverted=False):
    # Find Max Positive
    max_idx = np.argmax(y_data)
    max_val = y_data[max_idx]
    max_x = x_data[max_idx]
    
    # Find Max Negative
    min_idx = np.argmin(y_data)
    min_val = y_data[min_idx]
    min_x = x_data[min_idx]

    # Add Label for Max
    fig.add_annotation(
        x=max_x, y=max_val,
        text=f"{max_val:.2f}",
        showarrow=True, arrowhead=1,
        yshift=10 if not inverted else -10,
        font=dict(color="black", size=11, family="Arial Black"),
        bgcolor="rgba(255,255,255,0.7)"
    )
    # Add Label for Min
    fig.add_annotation(
        x=min_x, y=min_val,
        text=f"{min_val:.2f}",
        showarrow=True, arrowhead=1,
        yshift=-10 if not inverted else 10,
        font=dict(color="black", size=11, family="Arial Black"),
        bgcolor="rgba(255,255,255,0.7)"
    )

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
    cols = st.columns(n_span + 1)
    opts = ['Pin', 'Roller', 'Fix']
    supports = [cols[i].selectbox(f"S{i+1}", opts, index=0 if i==0 else 1, key=f"sup_{i}") for i in range(n_span+1)]
        
    st.subheader("Loads (1.4DL + 1.7LL)")
    loads_input = []
    
    for i in range(n_span):
        with st.expander(f"📍 Loads on Span {i+1}", expanded=True):
            c_u, c_p = st.columns([1, 1.5])
            
            # 1. Uniform Load
            with c_u:
                st.markdown("**Uniform Load**")
                u_dl = st.number_input("DL (kN/m)", 0.0, key=f"udl_{i}")
                u_ll = st.number_input("LL (kN/m)", 0.0, key=f"ull_{i}")
                if (u_dl + u_ll) > 0:
                    loads_input.append({
                        'span_idx': i, 'type': 'Uniform', 
                        'total_w': (1.4*u_dl + 1.7*u_ll)*1000
                    })

            # 2. Multiple Point Loads
            with c_p:
                st.markdown("**Concentrated Loads**")
                # User selects HOW MANY point loads on this span
                num_pt = st.number_input(f"Qty Point Loads", 0, 10, 0, key=f"num_pt_{i}")
                
                if num_pt > 0:
                    for j in range(num_pt):
                        st.caption(f"--- Point Load #{j+1} ---")
                        cc1, cc2, cc3 = st.columns(3)
                        p_dl = cc1.number_input(f"DL (kN)", 0.0, key=f"pdl_{i}_{j}")
                        p_ll = cc2.number_input(f"LL (kN)", 0.0, key=f"pll_{i}_{j}")
                        p_pos = cc3.number_input(f"Pos (m)", 0.0, spans[i], spans[i]/2, key=f"ppos_{i}_{j}")
                        
                        if (p_dl + p_ll) > 0:
                            loads_input.append({
                                'span_idx': i, 'type': 'Point',
                                'total_w': (1.4*p_dl + 1.7*p_ll)*1000,
                                'pos': p_pos
                            })

if st.button("🚀 Calculate", type="primary"):
    solver = SimpleBeamSolver(spans, supports, loads_input)
    u, err = solver.solve()
    
    if err:
        st.error(f"Error: {err}")
    else:
        df = solver.get_internal_forces(num_points=100) # 100 points for smooth curve
        
        m_max = df['moment'].max()
        m_min = df['moment'].min()
        v_max = df['shear'].abs().max()
        
        # Plot Logic
        c1, c2, c3 = st.columns(3)
        c1.metric("Max Moment (+)", f"{m_max:.2f} kN-m")
        c2.metric("Max Moment (-)", f"{m_min:.2f} kN-m")
        c3.metric("Max Shear", f"{v_max:.2f} kN")
        
        # SFD
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color='#D32F2F'), name='Shear'))
        # Add Min/Max Labels
        add_peak_labels(fig_v, df['x'].values, df['shear'].values)
        fig_v.update_layout(title="Shear Force Diagram (SFD)", yaxis_title="Shear (kN)", hovermode="x")
        st.plotly_chart(fig_v, use_container_width=True)
        
        # BMD (Tension Side)
        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color='#1976D2'), name='Moment'))
        # Add Min/Max Labels (Flag inverted=True for correct label position)
        add_peak_labels(fig_m, df['x'].values, df['moment'].values, inverted=True)
        fig_m.update_layout(
            title="Bending Moment Diagram (BMD) - Tension Side", 
            yaxis_title="Moment (kN-m)",
            yaxis=dict(autorange="reversed"), # Reverse Y axis
            hovermode="x"
        )
        st.plotly_chart(fig_m, use_container_width=True)

        # Design Check
        st.divider()
        st.subheader("📝 Quick Design Info")
        des_mu = max(abs(m_max), abs(m_min))
        fc, fy = 24, 400
        b, h, cov = 25, 50, 4
        d = (h-cov)/100
        Rn = (des_mu * 1000 / 0.9) / ((b/100)*d**2)
        m_rat = fy/(0.85*fc)
        try:
            rho = (1/m_rat)*(1 - np.sqrt(1 - (2*m_rat*Rn)/(fy*1e6)))
            st.success(f"Design Mu: {des_mu:.2f} kN-m | Required As ≈ {rho*b*d*100:.2f} cm²")
        except:
            st.error("Section too small (Compression Failure)")
