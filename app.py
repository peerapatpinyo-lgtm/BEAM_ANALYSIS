import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# ==============================================================================
# 1. PAGE CONFIGURATION & CLEAN STYLING
# ==============================================================================
st.set_page_config(
    page_title="RC Beam Pro: Final Edition",
    layout="wide",
    page_icon="🏗️",
    initial_sidebar_state="expanded"
)

# Clean, Compact, Professional CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Sarabun', sans-serif;
        font-size: 14px;
    }
    
    /* Compact Headers */
    h1 { font-size: 1.8rem !important; font-weight: 700; color: #1565C0; margin-bottom: 0.5rem; }
    h2 { font-size: 1.4rem !important; font-weight: 600; border-bottom: 2px solid #E0E0E0; padding-bottom: 5px; margin-top: 20px; }
    h3 { font-size: 1.1rem !important; font-weight: 600; color: #424242; }
    
    /* Result Cards */
    .metric-card {
        background: #F5F5F5;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val { font-size: 1.5rem; font-weight: 700; color: #1976D2; }
    .metric-lbl { font-size: 0.9rem; color: #616161; }
    
    /* Table Styling */
    .stDataFrame { font-size: 0.9rem; }
    
    /* Input Compactness */
    .stNumberInput { margin-bottom: 0px; }
    div[data-testid="stExpander"] div[role="button"] p { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. ANALYSIS ENGINE (MATRIX STIFFNESS METHOD) - FULL LOGIC
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
        # 1. Global Stiffness
        K_global = np.zeros((self.dof, self.dof))
        for i, L in enumerate(self.spans):
            K_el = self._get_element_stiffness(L)
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            for r in range(4):
                for c in range(4):
                    K_global[idx[r], idx[c]] += K_el[r, c]

        # 2. Forces & FEM
        F_global = np.zeros(self.dof)
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
        
        if not free_dof: return None, None
        
        # 4. Solve Displacements
        try:
            D_free = np.linalg.solve(K_global[np.ix_(free_dof, free_dof)], F_global[free_dof])
        except: return None, None
            
        D_total = np.zeros(self.dof)
        D_total[free_dof] = D_free
        
        # 5. Reactions
        R_vec = np.dot(K_global, D_total) + F_fem_equiv
        reactions = []
        for i, row in self.supports.iterrows():
            if row['type'] != 'None':
                reactions.append({
                    "Node": f"{i+1}",
                    "Support": row['type'],
                    "Ry (kg)": R_vec[2*i],
                    "Mz (kg-m)": R_vec[2*i+1]
                })
        
        # 6. Internal Forces
        final_x, final_v, final_m = [], [], []
        curr_x = 0
        
        for i, L in enumerate(self.spans):
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            u_el = D_total[idx]
            K_el = self._get_element_stiffness(L)
            f_stiff = np.dot(K_el, u_el)
            f_total = f_stiff + self._compute_fem(i)
            
            V_start, M_start = f_total[0], f_total[1]
            x_pts = np.linspace(0, L, 50) # 50 points per span
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
# 3. VISUALIZATION (FIXED BEAM LINE & LAYOUT)
# ==============================================================================
class DiagramPlotter:
    def __init__(self, df, spans, supports, loads):
        self.df = df
        self.spans = spans
        self.supports = supports
        self.loads = loads
        self.cum_dist = [0] + list(np.cumsum(spans))
        self.total_len = self.cum_dist[-1]

    def plot(self):
        # Calculate scales
        load_vals = [l['w'] for l in self.loads if l['type']=='U'] + [l['P'] for l in self.loads if l['type']=='P']
        max_load = max(map(abs, load_vals)) if load_vals else 100
        vh = max_load * 1.5 if max_load > 0 else 10
        sw = max(0.4, self.total_len*0.02)
        sh = vh * 0.15

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, 
            row_heights=[0.25, 0.35, 0.40],
            vertical_spacing=0.08,
            subplot_titles=("<b>1. Loading Diagram (Free Body)</b>", "<b>2. Shear Force (SFD)</b>", "<b>3. Bending Moment (BMD)</b>")
        )

        # --- ROW 1: BEAM & LOADS ---
        # 1.1 Draw Beam Line (ใช้ Scatter แทน Shape เพื่อความชัวร์)
        fig.add_trace(go.Scatter(
            x=[0, self.total_len], y=[0, 0], 
            mode='lines', line=dict(color='black', width=6), 
            hoverinfo='skip', showlegend=False
        ), row=1, col=1)

        # 1.2 Draw Supports
        for i, x in enumerate(self.cum_dist):
            if i < len(self.supports):
                self._draw_sup(fig, x, 0, self.supports.iloc[i]['type'], sw, sh)

        # 1.3 Draw Loads
        for l in self.loads:
            if l['type'] == 'U':
                x1, x2 = self.cum_dist[l['span_idx']], self.cum_dist[l['span_idx']+1]
                h = (l['w']/max_load)*vh*0.5
                # Fill Area
                fig.add_trace(go.Scatter(
                    x=[x1,x2,x2,x1], y=[0,0,h,h], 
                    fill='toself', fillcolor='rgba(255,152,0,0.4)', 
                    line_width=0, showlegend=False, hoverinfo='skip'
                ), row=1, col=1)
                # Top Line
                fig.add_trace(go.Scatter(x=[x1,x2], y=[h,h], mode='lines', line=dict(color='#E65100', width=2), showlegend=False), row=1, col=1)
                # Label
                fig.add_annotation(x=(x1+x2)/2, y=h, text=f"w={l['w']:.0f}", yshift=10, showarrow=False, font=dict(color='#E65100', size=10), row=1, col=1)
            
            elif l['type'] == 'P':
                px = self.cum_dist[l['span_idx']] + l['x']
                h = (l['P']/max_load)*vh*0.7
                fig.add_annotation(
                    x=px, y=0, ax=px, ay=h if h>0 else 10, 
                    xref="x1", yref="y1", axref="x1", ayref="y1", 
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#BF360C", 
                    text=f"P={l['P']:.0f}", row=1, col=1
                )

        # Fix Y-Axis for Loading Diagram (Ensure 0 is centered/visible)
        fig.update_yaxes(range=[-sh*1.5, vh*1.2], visible=False, fixedrange=True, row=1, col=1)

        # --- ROW 2 & 3: SFD / BMD ---
        for r, col, c, unit in [(2, 'shear', '#1976D2', 'kg'), (3, 'moment', '#D32F2F', 'kg-m')]:
            fig.add_trace(go.Scatter(
                x=self.df['x'], y=self.df[col], 
                fill='tozeroy', line=dict(color=c, width=2), 
                name=col.title(), mode='lines'
            ), row=r, col=1)
            
            # Max/Min Labels
            arr = self.df[col].values
            mx, mn = np.max(arr), np.min(arr)
            rn = mx - mn; pad = rn*0.3 if rn > 0 else 1.0
            fig.update_yaxes(range=[mn-pad, mx+pad], row=r, col=1) # Auto zoom with padding
            
            # Label Max/Min
            for val in [mx, mn]:
                if abs(val) > 0.1:
                    idx = np.argmax(arr == val) if val == mx else np.argmin(arr == val)
                    fig.add_annotation(
                        x=self.df['x'].iloc[idx], y=val, 
                        text=f"<b>{val:,.2f}</b>",
                        bgcolor="rgba(255,255,255,0.7)", bordercolor=c, borderwidth=1,
                        showarrow=False, yshift=15 if val>=0 else -15, 
                        font=dict(size=10, color=c), row=r, col=1
                    )

        # Final Layout
        fig.update_layout(
            height=800, 
            template="plotly_white", 
            margin=dict(l=40, r=40, t=30, b=30), 
            showlegend=False, 
            hovermode="x unified"
        )
        # Grid lines at supports
        for x in self.cum_dist: 
            fig.add_vline(x=x, line_dash="dot", line_color="gray", opacity=0.3)
        
        return fig

    def _draw_sup(self, fig, x, y, stype, sw, sh):
        lc, fc = "#37474F", "#B0BEC5"
        if stype == "Pin":
            path = f"M {x},{y} L {x-sw/2},{y-sh} L {x+sw/2},{y-sh} Z"
            fig.add_shape(type="path", path=path, fillcolor=fc, line=dict(color=lc, width=2), row=1, col=1)
            fig.add_shape(type="line", x0=x-sw, y0=y-sh, x1=x+sw, y1=y-sh, line=dict(color=lc, width=2), row=1, col=1)
        elif stype == "Roller":
            fig.add_shape(type="circle", x0=x-sw/2, y0=y-sh, x1=x+sw/2, y1=y, fillcolor=fc, line=dict(color=lc, width=2), row=1, col=1)
            fig.add_shape(type="line", x0=x-sw, y0=y-sh, x1=x+sw, y1=y-sh, line=dict(color=lc, width=2), row=1, col=1)
        elif stype == "Fixed":
            fig.add_shape(type="line", x0=x, y0=y-sh, x1=x, y1=y+sh, line=dict(color=lc, width=4), row=1, col=1)

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
            if term < 0: return {"status": "Fail", "msg": "Increase Section"}, {"status": "Check"}
            
            rho = (0.85*self.fc/self.fy)*(1 - np.sqrt(term))
            As = max(rho * self.b * self.d, 14/self.fy * self.b * self.d)
            res_flex = {"As": As, "msg": "OK"}
            
            vc = 0.53 * np.sqrt(self.fc) * self.b * self.d
            phi_vc = 0.75 * vc
            res_shear = {"phiVc": phi_vc, "req": "Stirrups Req" if self.Vu > phi_vc/2 else "Min Stirrups"}
        else: # WSD
            n, k, j = 10, 0.378, 0.874 
            fs = 0.5 * self.fy
            As = Mu_kgcm / (fs * j * self.d)
            res_flex = {"As": As, "msg": "WSD"}
            res_shear = {"phiVc": 0.29*np.sqrt(self.fc)*self.b*self.d, "req": "Check Shear"}
            
        return res_flex, res_shear

# ==============================================================================
# 5. MAIN UI
# ==============================================================================
def main():
    st.title("🏗️ RC Beam Pro: Final Edition")
    st.write("Structural Analysis & Design System")

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Settings")
        method = st.radio("Method", ["SDM", "WSD"])
        with st.expander("Materials", expanded=True):
            fc = st.number_input("fc' (ksc)", 240.0)
            fy = st.number_input("fy (ksc)", 4000.0)
        with st.expander("Section (cm)", expanded=True):
            b = st.number_input("Width (b)", 25.0)
            h = st.number_input("Depth (h)", 50.0)
            cv = st.number_input("Cover", 3.0)
        fdl = 1.4 if method=="SDM" else 1.0
        fll = 1.7 if method=="SDM" else 1.0

    # --- INPUTS ---
    c1, c2 = st.columns([1, 1.5], gap="large")
    with c1:
        st.subheader("1. Geometry")
        n_span = st.number_input("Spans", 1, 10, 2)
        spans, supports = [], []
        
        # Compact Span Input
        cols = st.columns(2)
        for i in range(n_span): 
            spans.append(cols[i%2].number_input(f"L{i+1} (m)", 1.0, 50.0, 5.0, key=f"s{i}"))
        
        # Compact Support Input
        st.write("Supports:")
        cols = st.columns(3)
        for i in range(n_span+1):
            def_idx = 0 if i==0 else (1 if i<n_span else 1)
            supports.append(cols[i%3].selectbox(f"N{i+1}", ["Pin", "Roller", "Fixed", "None"], index=def_idx, key=f"sup{i}"))
        
        df_sup = pd.DataFrame({'type': supports})
        stable = len(df_sup[df_sup['type']!='None']) >= 2 or "Fixed" in supports

    with c2:
        st.subheader("2. Loads")
        loads = []
        tabs = st.tabs([f"Span {i+1}" for i in range(n_span)])
        for i, tab in enumerate(tabs):
            with tab:
                cc1, cc2 = st.columns(2)
                with cc1:
                    st.markdown("**Uniform Load (kg/m)**")
                    d = st.number_input("DL", key=f"d{i}")
                    l = st.number_input("LL", key=f"l{i}")
                    if d+l > 0: loads.append({'span_idx':i, 'type':'U', 'w':d*fdl+l*fll})
                with cc2:
                    st.markdown("**Point Load (kg)**")
                    if st.checkbox(f"Add Point Load?", key=f"cp{i}"):
                        pd = st.number_input("P(DL)", key=f"pd{i}")
                        pl = st.number_input("P(LL)", key=f"pl{i}")
                        px = st.number_input("x (m)", 0.0, spans[i], spans[i]/2, key=f"px{i}")
                        if pd+pl > 0: loads.append({'span_idx':i, 'type':'P', 'P':pd*fdl+pl*fll, 'x':px})

    if not stable: st.error("⚠️ Unstable Structure"); st.stop()

    st.markdown("---")
    if st.button("🚀 Analyze Beam", type="primary", use_container_width=True):
        solver = StructuralEngine(spans, df_sup, loads, E=2e6, I=(b*h**3/12)*1e-8)
        df_res, df_reac = solver.solve()
        
        if df_res is None: st.error("Calculation Error"); st.stop()

        # --- RESULTS ---
        # 1. Diagram
        st.subheader("Analysis Diagrams")
        plotter = DiagramPlotter(df_res, spans, df_sup, loads)
        # Use simple container width (Responsive)
        st.plotly_chart(plotter.plot(), use_container_width=True)

        # 2. Results Grid
        c_res1, c_res2 = st.columns([1, 1.5])
        
        with c_res1:
            st.subheader("🔹 Support Reactions")
            st.dataframe(df_reac.style.format({"Ry (kg)": "{:,.2f}", "Mz (kg-m)": "{:,.2f}"}), hide_index=True, use_container_width=True)
        
        with c_res2:
            st.subheader("🔹 Design Forces & RC Check")
            v_max = df_res['shear'].abs().max()
            m_max = df_res['moment'].abs().max()
            
            # Metric Cards
            mc1, mc2 = st.columns(2)
            mc1.markdown(f'<div class="metric-card"><div class="metric-lbl">Max Shear (Vu)</div><div class="metric-val">{v_max:,.0f} kg</div></div>', unsafe_allow_html=True)
            mc2.markdown(f'<div class="metric-card"><div class="metric-lbl">Max Moment (Mu)</div><div class="metric-val">{m_max:,.0f} kg-m</div></div>', unsafe_allow_html=True)
            
            st.write("") # Spacer
            des = RCDesign(m_max, v_max, fc, fy, b, h, cv, method)
            rf, rs = des.check()
            
            st.success(f"**Flexure Design:** Req As = {rf.get('As', 0):.2f} cm² ({rf.get('msg','')})")
            bar_txt = " | ".join([f"DB{d}: {math.ceil(rf.get('As',0)/(3.14*(d/20)**2))}" for d in [12,16,20,25]])
            st.caption(f"Suggestion: {bar_txt}")
            
            st.info(f"**Shear Design:** {rs.get('req','')}")

if __name__ == "__main__":
    main()
