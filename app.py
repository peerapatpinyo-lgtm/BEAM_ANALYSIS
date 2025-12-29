import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# ==============================================================================
# 1. PAGE CONFIGURATION & CSS STYLING
# ==============================================================================
st.set_page_config(
    page_title="RC Beam Pro: Platinum Edition",
    layout="wide",
    page_icon="🏗️",
    initial_sidebar_state="expanded"
)

# Custom CSS for Engineering Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Sarabun', sans-serif;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
        color: white; padding: 20px; border-radius: 10px;
        text-align: center; margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    
    /* Section Titles */
    .section-title {
        border-left: 6px solid #1565C0;
        padding-left: 15px; font-size: 1.4rem; font-weight: 700;
        color: #0D47A1; margin-top: 30px; margin-bottom: 15px;
        background-color: #E3F2FD; padding-top: 8px; padding-bottom: 8px;
        border-radius: 0 8px 8px 0;
    }

    /* Scrollable Plot Container */
    .plot-scroll-container {
        width: 100%; overflow-x: auto; white-space: nowrap;
        border: 1px solid #B0BEC5; border-radius: 8px;
        padding: 5px; background: white;
    }
    
    /* Tables */
    .dataframe { font-size: 0.95rem !important; }
    
    /* Inputs */
    .stNumberInput input { font-weight: 600; color: #1565C0; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. STRUCTURAL ENGINE (MATRIX METHOD + REACTIONS)
# ==============================================================================
class StructuralEngine:
    def __init__(self, spans, supports, loads, E=2e6, I=1e-3):
        self.spans = spans
        self.supports = supports
        self.loads = loads
        self.E = E
        self.I = I
        self.node_coords = [0] + list(np.cumsum(spans))
        self.n_nodes = len(self.node_coords)
        self.dof = 2 * self.n_nodes 

    def _get_element_stiffness(self, L):
        k_const = self.E * self.I / L**3
        K = np.array([
            [12,    6*L,    -12,    6*L],
            [6*L,   4*L**2, -6*L,   2*L**2],
            [-12,   -6*L,   12,     -6*L],
            [6*L,   2*L**2, -6*L,   4*L**2]
        ])
        return K * k_const

    def _compute_fem(self, span_idx):
        L = self.spans[span_idx]
        fem_vec = np.zeros(4)
        span_loads = [l for l in self.loads if l['span_idx'] == span_idx]
        
        for load in span_loads:
            if load['type'] == 'U':
                w = load['w']
                fem_vec += np.array([w*L/2, w*L**2/12, w*L/2, -w*L**2/12])
            elif load['type'] == 'P':
                P = load['P']; a = load['x']; b = L - a
                fem_vec += np.array([
                    (P*b**2*(3*a+b))/L**3, (P*a*b**2)/L**2,
                    (P*a**2*(a+3*b))/L**3, -(P*a**2*b)/L**2
                ])
        return fem_vec

    def solve(self):
        # 1. Global Stiffness Matrix
        K_global = np.zeros((self.dof, self.dof))
        for i, L in enumerate(self.spans):
            K_el = self._get_element_stiffness(L)
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            for r in range(4):
                for c in range(4):
                    K_global[idx[r], idx[c]] += K_el[r, c]

        # 2. Equivalent Nodal Forces (Subtract FEM)
        F_global = np.zeros(self.dof)
        # Store accumulated FEM for Reaction calculation later
        F_fem_equiv = np.zeros(self.dof) 
        
        for i in range(len(self.spans)):
            fem = self._compute_fem(i)
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            F_global[idx] -= fem
            F_fem_equiv[idx] += fem 

        # 3. Boundary Conditions
        constrained_dof = []
        for node_i, row in self.supports.iterrows():
            stype = row['type']
            if stype in ["Pin", "Roller"]:
                constrained_dof.append(2*node_i)
            elif stype == "Fixed":
                constrained_dof.extend([2*node_i, 2*node_i+1])
        
        free_dof = [d for d in range(self.dof) if d not in constrained_dof]
        
        if not free_dof: raise ValueError("Fully constrained.")
        
        # 4. Solve Displacements
        try:
            D_free = np.linalg.solve(K_global[np.ix_(free_dof, free_dof)], F_global[free_dof])
        except: raise ValueError("Unstable Structure.")
            
        D_total = np.zeros(self.dof)
        D_total[free_dof] = D_free
        
        # 5. Calculate Reactions
        # R = K * D + FEM_forces (Forces exerted by members on nodes)
        # Usually R = K*D - F_external_nodal. 
        # Since F_external_nodal = -FEM_equiv (we assumed 0 direct loads), 
        # R = K*D + FEM_equiv
        R_vec = np.dot(K_global, D_total) + F_fem_equiv
        
        reactions = []
        for i, row in self.supports.iterrows():
            if row['type'] != 'None':
                # Reaction Y
                Ry = R_vec[2*i]
                # Reaction Moment
                Mz = R_vec[2*i+1]
                reactions.append({
                    "Support": f"S{i+1} ({row['type']})",
                    "Ry (kg)": Ry,
                    "Mz (kg-m)": Mz
                })
        
        # 6. Internal Forces for Plotting
        final_x, final_v, final_m = [], [], []
        curr_x = 0
        
        for i, L in enumerate(self.spans):
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            u_el = D_total[idx]
            K_el = self._get_element_stiffness(L)
            f_stiff = np.dot(K_el, u_el)
            f_total = f_stiff + self._compute_fem(i)
            
            V_start, M_start = f_total[0], f_total[1]
            x_pts = np.linspace(0, L, 100)
            span_loads = [l for l in self.loads if l['span_idx'] == i]
            
            for x in x_pts:
                Vx, Mx = V_start, -M_start + V_start*x
                for l in span_loads:
                    if l['type'] == 'U' and x > 0:
                        Vx -= l['w']*x
                        Mx -= l['w']*x**2/2
                    elif l['type'] == 'P' and x > l['x']:
                        Vx -= l['P']
                        Mx -= l['P']*(x - l['x'])
                
                final_x.append(curr_x + x)
                final_v.append(Vx)
                final_m.append(Mx)
            curr_x += L
            
        return pd.DataFrame({'x': final_x, 'shear': final_v, 'moment': final_m}), pd.DataFrame(reactions)

# ==============================================================================
# 3. VISUALIZATION (FIXED: BEAM LINE & LABELS)
# ==============================================================================
class DiagramPlotter:
    def __init__(self, df, spans, supports, loads):
        self.df = df
        self.spans = spans
        self.supports = supports
        self.loads = loads
        self.cum_dist = [0] + list(np.cumsum(spans))
        self.total_len = self.cum_dist[-1]

    def _draw_sup(self, fig, x, y, stype, sw, sh):
        lc, fc = "#263238", "#CFD8DC"
        if stype == "Pin":
            path = f"M {x},{y} L {x-sw/2},{y-sh} L {x+sw/2},{y-sh} Z"
            fig.add_shape(type="path", path=path, fillcolor=fc, line=dict(color=lc, width=2), row=1, col=1)
            fig.add_shape(type="line", x0=x-sw, y0=y-sh, x1=x+sw, y1=y-sh, line=dict(color=lc, width=2), row=1, col=1)
        elif stype == "Roller":
            fig.add_shape(type="circle", x0=x-sw/2, y0=y-sh, x1=x+sw/2, y1=y, fillcolor=fc, line=dict(color=lc, width=2), row=1, col=1)
            fig.add_shape(type="line", x0=x-sw, y0=y-sh, x1=x+sw, y1=y-sh, line=dict(color=lc, width=2), row=1, col=1)
        elif stype == "Fixed":
            fig.add_shape(type="line", x0=x, y0=y-sh, x1=x, y1=y+sh, line=dict(color=lc, width=4), row=1, col=1)
            for k in range(5):
                dy = (2*sh)/5*k - sh
                fig.add_shape(type="line", x0=x, y0=y+dy, x1=x-sw/2, y1=y+dy+sw/3, line=dict(color=lc, width=1), row=1, col=1)

    def plot(self):
        # Scale Calculation
        load_vals = [l['w'] for l in self.loads if l['type']=='U'] + [l['P'] for l in self.loads if l['type']=='P']
        max_load = max(map(abs, load_vals)) if load_vals else 100
        vh = max_load * 1.5
        sw, sh = max(0.5, self.total_len*0.02), vh*0.15
        
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.3, 0.35, 0.35], vertical_spacing=0.08,
                            subplot_titles=("<b>1. Loading Diagram (Free Body)</b>", "<b>2. Shear Force Diagram (SFD)</b>", "<b>3. Bending Moment Diagram (BMD)</b>"))

        # --- Row 1: Beam & Loads ---
        # **FIXED: Beam Line Explicitly Added**
        fig.add_shape(type="line", x0=0, y0=0, x1=self.total_len, y1=0, 
                      line=dict(color="black", width=4), layer="above", row=1, col=1)
        
        for i, x in enumerate(self.cum_dist):
            if i < len(self.supports):
                self._draw_sup(fig, x, 0, self.supports.iloc[i]['type'], sw, sh)
                
        for l in self.loads:
            if l['type'] == 'U':
                x1, x2 = self.cum_dist[l['span_idx']], self.cum_dist[l['span_idx']+1]
                h = (l['w']/max_load)*vh*0.5
                fig.add_trace(go.Scatter(x=[x1,x2,x2,x1], y=[0,0,h,h], fill='toself', fillcolor='rgba(255,152,0,0.3)', line_width=0, showlegend=False, hoverinfo='skip'), row=1, col=1)
                fig.add_trace(go.Scatter(x=[x1,x2], y=[h,h], mode='lines', line=dict(color='#E65100', width=2), showlegend=False), row=1, col=1)
                fig.add_annotation(x=(x1+x2)/2, y=h, text=f"<b>w={l['w']:.0f}</b>", yshift=15, showarrow=False, font=dict(color='#E65100'), row=1, col=1)
            elif l['type'] == 'P':
                px = self.cum_dist[l['span_idx']] + l['x']
                h = (l['P']/max_load)*vh*0.7
                fig.add_annotation(x=px, y=0, ax=px, ay=h, xref="x1", yref="y1", axref="x1", ayref="y1", showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2, arrowcolor="#BF360C", text=f"<b>P={l['P']:.0f}</b>", row=1, col=1)
        
        fig.update_yaxes(visible=False, fixedrange=True, row=1, col=1)

        # --- Row 2 & 3: SFD / BMD ---
        for r, col_name, c, unit in [(2, 'shear', '#1976D2', 'kg'), (3, 'moment', '#D32F2F', 'kg-m')]:
            fig.add_trace(go.Scatter(x=self.df['x'], y=self.df[col_name], fill='tozeroy', line=dict(color=c, width=2), name=col_name.title()), row=r, col=1)
            
            # **FIXED: Max/Min Labels with Units**
            arr = self.df[col_name].values
            mx, mn = np.max(arr), np.min(arr)
            rn = mx - mn; pad = rn*0.2 if rn > 0 else 1.0
            fig.update_yaxes(range=[mn-pad, mx+pad], row=r, col=1)
            
            # Add labels for Max and Min
            for val in [mx, mn]:
                if abs(val) > 0.1: # Skip near-zero noise
                    idx = np.argmax(arr == val) if val == mx else np.argmin(arr == val)
                    x_pos = self.df['x'].iloc[idx]
                    ys = 15 if val >= 0 else -15
                    fig.add_annotation(
                        x=x_pos, y=val, text=f"<b>{val:,.2f} {unit}</b>",
                        bgcolor="rgba(255,255,255,0.85)", bordercolor=c, borderwidth=1,
                        showarrow=False, yshift=ys, font=dict(size=11, color=c),
                        row=r, col=1
                    )

        # Layout
        width_px = max(1000, len(self.spans)*280)
        fig.update_layout(width=width_px, height=900, template="plotly_white", margin=dict(l=40, r=40, t=40, b=40), showlegend=False, hovermode="x unified")
        for x in self.cum_dist: fig.add_vline(x=x, line_dash="dot", line_color="gray", opacity=0.4)
        
        return fig

# ==============================================================================
# 4. RC DESIGN MODULE
# ==============================================================================
class RCDesign:
    def __init__(self, Mu, Vu, fc, fy, b, h, cover, method):
        self.Mu, self.Vu = abs(Mu), abs(Vu)
        self.fc, self.fy, self.b, self.h = fc, fy, b, h
        self.d = h - cover
        self.method = method

    def check(self):
        Mu_kgcm = self.Mu * 100
        res_flex, res_shear = {}, {}
        
        if self.method == "SDM":
            phi = 0.9
            Rn = Mu_kgcm / (phi * self.b * self.d**2)
            term = 1 - (2*Rn)/(0.85*self.fc)
            if term < 0: return {"status": "Fail", "msg": "Section too small"}, {"status": "Check"}
            
            rho = (0.85*self.fc/self.fy)*(1 - np.sqrt(term))
            As = rho * self.b * self.d
            As_min = max(14/self.fy, 0.25*np.sqrt(self.fc)/self.fy) * self.b * self.d
            res_flex = {"As": max(As, As_min), "rho": rho, "msg": "OK" if rho < 0.025 else "Warning (High Rho)"}
            
            vc = 0.53 * np.sqrt(self.fc) * self.b * self.d
            phi_vc = 0.75 * vc
            res_shear = {"phiVc": phi_vc, "req": "Stirrups Req" if self.Vu > phi_vc/2 else "Min Stirrups"}
        else:
            n, k, j = 10, 0.378, 0.874 # Simplified
            fs = 0.5 * self.fy
            As = Mu_kgcm / (fs * j * self.d)
            res_flex = {"As": As, "msg": "WSD"}
            res_shear = {"req": "Check allowable shear stress"}
            
        return res_flex, res_shear

# ==============================================================================
# 5. MAIN UI
# ==============================================================================
def main():
    st.markdown('<div class="main-header"><h1>🏗️ RC Beam Pro: Platinum Edition</h1></div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        method = st.radio("Method", ["SDM", "WSD"])
        with st.expander("Properties", expanded=True):
            fc = st.number_input("fc' (ksc)", 240.0)
            fy = st.number_input("fy (ksc)", 4000.0)
            b = st.number_input("Width (cm)", 25.0)
            h = st.number_input("Depth (cm)", 50.0)
            cv = st.number_input("Cover (cm)", 3.0)
        if method=="SDM":
            fdl, fll = st.number_input("DL Factor", 1.4), st.number_input("LL Factor", 1.7)
        else: fdl, fll = 1.0, 1.0

    # Inputs
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.markdown('<div class="section-title">1. Geometry</div>', unsafe_allow_html=True)
        n_span = st.number_input("Spans", 1, 15, 2)
        spans, supports = [], []
        cols = st.columns(3)
        st.caption("Length (m)")
        for i in range(n_span): spans.append(cols[i%3].number_input(f"L{i+1}", 1.0, 50.0, 5.0, key=f"s{i}"))
        
        st.caption("Supports")
        cols = st.columns(3)
        for i in range(n_span+1):
            def_idx = 0 if i==0 else (1 if i<n_span else 1)
            supports.append(cols[i%3].selectbox(f"S{i+1}", ["Pin", "Roller", "Fixed", "None"], index=def_idx, key=f"sup{i}"))
        df_sup = pd.DataFrame({'type': supports})
        stable = len(df_sup[df_sup['type']!='None']) >= 2 or "Fixed" in supports

    with c2:
        st.markdown('<div class="section-title">2. Loads</div>', unsafe_allow_html=True)
        loads = []
        tabs = st.tabs([f"Span {i+1}" for i in range(n_span)])
        for i, tab in enumerate(tabs):
            with tab:
                cc1, cc2 = st.columns(2)
                with cc1:
                    wdl = st.number_input("UDL DL (kg/m)", key=f"d{i}")
                    wll = st.number_input("UDL LL (kg/m)", key=f"l{i}")
                    if wdl+wll > 0: loads.append({'span_idx':i, 'type':'U', 'w':wdl*fdl+wll*fll})
                with cc2:
                    p_qty = st.number_input("Point Loads", 0, 5, 0, key=f"pq{i}")
                    for j in range(p_qty):
                        c_p = st.columns([1,1,1.5])
                        pd = c_p[0].number_input(f"DL", key=f"pd{i}{j}")
                        pl = c_p[1].number_input(f"LL", key=f"pl{i}{j}")
                        px = c_p[2].number_input(f"x (m)", 0.0, spans[i], key=f"px{i}{j}")
                        if pd+pl > 0: loads.append({'span_idx':i, 'type':'P', 'P':pd*fdl+pl*fll, 'x':px})

    if not stable: st.error("⚠️ Structure Unstable!"); st.stop()

    if st.button("🚀 Analyze & Design", type="primary", use_container_width=True):
        solver = StructuralEngine(spans, df_sup, loads)
        df_res, df_reac = solver.solve()
        
        # --- RESULTS ---
        st.markdown('<div class="section-title">3. Analysis Results</div>', unsafe_allow_html=True)
        
        # 1. Plot
        plotter = DiagramPlotter(df_res, spans, df_sup, loads)
        st.markdown(f'<div class="plot-scroll-container">{plotter.plot().to_html(full_html=False, include_plotlyjs="cdn")}</div>', unsafe_allow_html=True)
        
        # 2. Reactions Table (NEW)
        st.markdown("### 🔹 Support Reactions")
        # Format columns nicely
        st.dataframe(
            df_reac.style.format({"Ry (kg)": "{:,.2f}", "Mz (kg-m)": "{:,.2f}"}), 
            use_container_width=True, 
            hide_index=True
        )

        # 3. Max Forces Summary (Fixed Units)
        m_max = df_res['moment'].abs().max()
        v_max = df_res['shear'].abs().max()
        
        st.markdown("### 🔹 Max Design Forces")
        col_res1, col_res2 = st.columns(2)
        col_res1.metric("Max Shear (Vu)", f"{v_max:,.2f} kg")
        col_res2.metric("Max Moment (Mu)", f"{m_max:,.2f} kg-m")
        
        # 4. Design
        st.markdown('<div class="section-title">4. RC Design</div>', unsafe_allow_html=True)
        des = RCDesign(m_max, v_max, fc, fy, b, h, cv, method)
        rf, rs = des.check()
        
        d1, d2 = st.columns(2)
        with d1:
            st.info(f"**Flexure ({rf.get('msg', '')})**")
            st.write(f"Required As: **{rf.get('As', 0):,.2f} cm²**")
            bars = [12, 16, 20, 25]
            txt = " | ".join([f"DB{d}: {math.ceil(rf.get('As',0)/(3.14*(d/20)**2))}เส้น" for d in bars])
            st.caption(txt)
            
        with d2:
            st.warning(f"**Shear**")
            st.write(f"Vu: {v_max:,.0f} kg | phiVc: {rs.get('phiVc',0):,.0f} kg")
            st.write(f"Result: **{rs.get('req', '')}**")

if __name__ == "__main__":
    main()
