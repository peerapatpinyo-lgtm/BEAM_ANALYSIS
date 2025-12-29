import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# ==============================================================================
# 1. SYSTEM CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="RC Beam Pro: Senior Edition",
    layout="wide",
    page_icon="🏗️",
    initial_sidebar_state="expanded"
)

# Professional CSS (Clean & Compact)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Sarabun', sans-serif;
    }
    
    /* Header */
    .main-header {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 6px solid #1565C0;
        margin-bottom: 20px;
    }
    .main-header h1 {
        color: #0D47A1;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .main-header p {
        color: #546E7A;
        font-size: 1rem;
        margin-top: 5px;
    }

    /* Cards */
    .metric-container {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 0.9rem;
        color: #616161;
        font-weight: 500;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1565C0;
    }

    /* Plot Container */
    .chart-box {
        background: white;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #ddd;
    }
    
    /* Input Tweaks */
    .stNumberInput div[data-baseweb="input"] {
        background-color: #fff;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. CORE CALCULATION ENGINE (MATRIX STIFFNESS METHOD)
# ==============================================================================
class BeamAnalysisEngine:
    """
    Core solver class using Direct Stiffness Method.
    Encapsulated to prevent scope pollution.
    """
    def __init__(self, spans, supports, loads, E=2e6, I=1e-3):
        self.spans = spans
        self.supports = supports
        self.loads = loads
        self.E = E
        self.I = I
        
        # Calculate Node Coordinates (Cumulative Distance)
        self.nodes = [0] + list(np.cumsum(spans))
        self.num_nodes = len(self.nodes)
        self.dof = 2 * self.num_nodes # 2 DOF per node (y, theta)

    def _element_stiffness(self, L):
        """Builds local stiffness matrix (4x4)"""
        k = (self.E * self.I) / (L**3)
        # Standard Beam Element Stiffness Matrix
        # [v1, theta1, v2, theta2]
        K = np.array([
            [12,      6*L,     -12,     6*L],
            [6*L,     4*L**2,  -6*L,    2*L**2],
            [-12,     -6*L,    12,      -6*L],
            [6*L,     2*L**2,  -6*L,    4*L**2]
        ])
        return K * k

    def _calculate_fem(self, span_idx):
        """Calculates Fixed End Moments (FEM) vector (4x1)"""
        L = self.spans[span_idx]
        fem = np.zeros(4)
        
        span_loads = [l for l in self.loads if l['span_idx'] == span_idx]
        
        for load in span_loads:
            if load['type'] == 'U':
                w = load['w']
                # FEM for Uniform Load: wL^2/12
                # Forces: wL/2
                fem += np.array([w*L/2, w*L**2/12, w*L/2, -w*L**2/12])
                
            elif load['type'] == 'P':
                P = load['P']
                a = load['x']
                b = L - a
                # FEM for Point Load
                fem += np.array([
                    (P * b**2 * (3*a + b)) / L**3,  # V1
                    (P * a * b**2) / L**2,          # M1
                    (P * a**2 * (a + 3*b)) / L**3,  # V2
                    -(P * a**2 * b) / L**2          # M2
                ])
        return fem

    def solve(self):
        """Executes the matrix solver"""
        try:
            # 1. Assemble Global Stiffness Matrix
            K_global = np.zeros((self.dof, self.dof))
            for i, L in enumerate(self.spans):
                K_el = self._element_stiffness(L)
                # Global indices: [2i, 2i+1, 2i+2, 2i+3]
                idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
                for r in range(4):
                    for c in range(4):
                        K_global[idx[r], idx[c]] += K_el[r, c]

            # 2. Assemble Force Vector (from FEM)
            F_nodes = np.zeros(self.dof)
            F_fem_equiv = np.zeros(self.dof) # Keep track for reactions
            
            for i in range(len(self.spans)):
                fem = self._calculate_fem(i)
                idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
                F_nodes[idx] -= fem  # Applied force is opposite to FEM
                F_fem_equiv[idx] += fem

            # 3. Apply Boundary Conditions
            constrained_dofs = []
            for i, row in self.supports.iterrows():
                stype = row['type']
                if stype in ['Pin', 'Roller']:
                    constrained_dofs.append(2*i) # Fix Vertical (y)
                elif stype == 'Fixed':
                    constrained_dofs.append(2*i)   # Fix Vertical (y)
                    constrained_dofs.append(2*i+1) # Fix Rotation (theta)
            
            free_dofs = [d for d in range(self.dof) if d not in constrained_dofs]
            
            if not free_dofs:
                return None, None # System is fully locked/invalid

            # 4. Solve for Displacements: [K]{D} = {F}
            K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
            F_reduced = F_nodes[free_dofs]
            
            D_free = np.linalg.solve(K_reduced, F_reduced)
            
            D_total = np.zeros(self.dof)
            D_total[free_dofs] = D_free

            # 5. Calculate Reactions: {R} = [K]{D} + {FEM}
            R_vec = np.dot(K_global, D_total) + F_fem_equiv
            
            reactions_list = []
            for i, row in self.supports.iterrows():
                if row['type'] != 'None':
                    reactions_list.append({
                        "Node": i+1,
                        "Support": row['type'],
                        "Ry (kg)": R_vec[2*i],
                        "Mz (kg-m)": R_vec[2*i+1]
                    })
            df_reactions = pd.DataFrame(reactions_list)

            # 6. Post-Processing (Internal Forces along beam)
            x_vals, shear_vals, moment_vals = [], [], []
            current_x = 0
            
            for i, L in enumerate(self.spans):
                # Element Displacements
                idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
                u_el = D_total[idx]
                
                # Forces at ends from stiffness
                f_stiff = np.dot(self._element_stiffness(L), u_el)
                # Total forces = stiffness + FEM
                f_total = f_stiff + self._calculate_fem(i)
                
                V_start, M_start = f_total[0], f_total[1]
                
                # Sampling points
                pts = np.linspace(0, L, 50)
                span_loads = [l for l in self.loads if l['span_idx'] == i]
                
                for x in pts:
                    # Equilibrium at distance x
                    Vx = V_start
                    Mx = -M_start + V_start * x
                    
                    for l in span_loads:
                        if l['type'] == 'U':
                            if x > 0:
                                Vx -= l['w'] * x
                                Mx -= l['w'] * x**2 / 2
                        elif l['type'] == 'P':
                            if x > l['x']:
                                Vx -= l['P']
                                Mx -= l['P'] * (x - l['x'])
                    
                    x_vals.append(current_x + x)
                    shear_vals.append(Vx)
                    moment_vals.append(Mx)
                
                current_x += L

            df_results = pd.DataFrame({
                'x': x_vals,
                'shear': shear_vals,
                'moment': moment_vals
            })
            
            return df_results, df_reactions
            
        except np.linalg.LinAlgError:
            return None, None # Singular matrix (Unstable)
        except Exception as e:
            st.error(f"Internal Solver Error: {str(e)}")
            return None, None


# ==============================================================================
# 3. VISUALIZATION ENGINE (GUARANTEED RENDERING)
# ==============================================================================
class DiagramVisualizer:
    def __init__(self, df, spans, supports, loads):
        self.df = df
        self.spans = spans
        self.supports = supports
        self.loads = loads
        self.cum_dist = [0] + list(np.cumsum(spans))
        self.total_len = self.cum_dist[-1]

    def create_plot(self):
        # Auto-scaling
        load_magnitudes = [l['w'] for l in self.loads if l['type']=='U'] + [l['P'] for l in self.loads if l['type']=='P']
        max_load = max(map(abs, load_magnitudes)) if load_magnitudes else 100
        
        # Visual dimensions
        viz_h = max_load * 1.5 if max_load > 0 else 10
        sup_w = max(0.3, self.total_len * 0.02)
        sup_h = viz_h * 0.15

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.3, 0.35, 0.35],
            subplot_titles=("<b>1. Loading Diagram</b>", "<b>2. Shear Force (SFD)</b>", "<b>3. Bending Moment (BMD)</b>")
        )

        # --- ROW 1: LOADING ---
        # 1.1 The Beam Line (Black & Thick - Using Scatter to guarantee it appears)
        fig.add_trace(go.Scatter(
            x=[0, self.total_len], y=[0, 0],
            mode='lines',
            line=dict(color='black', width=5),
            hoverinfo='skip',
            name='Beam'
        ), row=1, col=1)

        # 1.2 Supports
        for i, x in enumerate(self.cum_dist):
            if i < len(self.supports):
                self._add_support_icon(fig, x, 0, self.supports.iloc[i]['type'], sup_w, sup_h)

        # 1.3 Loads
        for l in self.loads:
            if l['type'] == 'U':
                x1, x2 = self.cum_dist[l['span_idx']], self.cum_dist[l['span_idx']+1]
                h = (l['w']/max_load) * viz_h * 0.6
                
                # Load block
                fig.add_trace(go.Scatter(
                    x=[x1, x2, x2, x1], y=[0, 0, h, h],
                    fill='toself', fillcolor='rgba(255, 111, 0, 0.25)',
                    line_width=0, showlegend=False, hoverinfo='skip'
                ), row=1, col=1)
                
                # Load line
                fig.add_trace(go.Scatter(
                    x=[x1, x2], y=[h, h], mode='lines',
                    line=dict(color='#E65100', width=2), showlegend=False
                ), row=1, col=1)
                
                # Label
                fig.add_annotation(
                    x=(x1+x2)/2, y=h, text=f"w={l['w']:,.0f}",
                    yshift=10, showarrow=False, font=dict(color='#E65100', size=11), row=1, col=1
                )

            elif l['type'] == 'P':
                x_pos = self.cum_dist[l['span_idx']] + l['x']
                h = (l['P']/max_load) * viz_h * 0.7
                
                # Arrow
                fig.add_annotation(
                    x=x_pos, y=0, ax=x_pos, ay=h if h>0 else 20,
                    xref="x1", yref="y1", axref="x1", ayref="y1",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#C62828",
                    text=f"P={l['P']:,.0f}", row=1, col=1
                )

        # Force Row 1 Y-axis to be invisible but correct scale
        fig.update_yaxes(visible=False, fixedrange=True, range=[-sup_h*1.5, viz_h*1.4], row=1, col=1)

        # --- ROW 2 & 3: SFD & BMD ---
        self._add_diagram_trace(fig, self.df['x'], self.df['shear'], 2, "Shear", "#1565C0", "kg")
        self._add_diagram_trace(fig, self.df['x'], self.df['moment'], 3, "Moment", "#C62828", "kg-m")

        # Layout Settings
        fig.update_layout(
            height=900,
            template="plotly_white",
            showlegend=False,
            hovermode="x unified",
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        # Grid lines
        for x in self.cum_dist:
            fig.add_vline(x=x, line_dash="dot", line_color="gray", opacity=0.3)

        return fig

    def _add_diagram_trace(self, fig, x, y, row, name, color, unit):
        # Trace
        fig.add_trace(go.Scatter(
            x=x, y=y, fill='tozeroy', mode='lines',
            line=dict(color=color, width=2), name=name
        ), row=row, col=1)
        
        # Scale & Label
        vals = y.values
        mx, mn = np.max(vals), np.min(vals)
        rng = mx - mn
        pad = rng * 0.2 if rng > 0 else 1.0
        fig.update_yaxes(range=[mn-pad, mx+pad], row=row, col=1)
        
        for val in [mx, mn]:
            if abs(val) > 0.01:
                idx = np.argmax(vals==val) if val==mx else np.argmin(vals==val)
                fig.add_annotation(
                    x=x.iloc[idx], y=val, text=f"<b>{val:,.1f}</b>",
                    bgcolor="rgba(255,255,255,0.8)", bordercolor=color,
                    showarrow=False, yshift=15 if val>=0 else -15,
                    font=dict(color=color, size=11), row=row, col=1
                )

    def _add_support_icon(self, fig, x, y, stype, w, h):
        lc, fc = "#37474F", "#CFD8DC"
        if stype == "Pin":
            # Triangle path
            path = f"M {x},{y} L {x-w/2},{y-h} L {x+w/2},{y-h} Z"
            fig.add_shape(type="path", path=path, fillcolor=fc, line=dict(color=lc, width=2), row=1, col=1)
            fig.add_shape(type="line", x0=x-w, y0=y-h, x1=x+w, y1=y-h, line=dict(color=lc, width=2), row=1, col=1)
        elif stype == "Roller":
            fig.add_shape(type="circle", x0=x-w/2, y0=y-h, x1=x+w/2, y1=y, fillcolor=fc, line=dict(color=lc, width=2), row=1, col=1)
            fig.add_shape(type="line", x0=x-w, y0=y-h, x1=x+w, y1=y-h, line=dict(color=lc, width=2), row=1, col=1)
        elif stype == "Fixed":
            fig.add_shape(type="line", x0=x, y0=y-h, x1=x, y1=y+h, line=dict(color=lc, width=4), row=1, col=1)


# ==============================================================================
# 4. RC DESIGN MODULE
# ==============================================================================
class RCDesigner:
    def __init__(self, Mu, Vu, fc, fy, b, h, cover, method):
        self.Mu = abs(Mu)
        self.Vu = abs(Vu)
        self.fc = fc
        self.fy = fy
        self.b = b
        self.h = h
        self.d = h - cover
        self.method = method
    
    def check(self):
        Mu_kgcm = self.Mu * 100
        res_flex = {}
        res_shear = {}
        
        if self.method == "SDM":
            phi = 0.9
            Rn = Mu_kgcm / (phi * self.b * self.d**2)
            limit = (2 * Rn) / (0.85 * self.fc)
            term = 1 - limit
            
            if term < 0:
                res_flex = {"As": 0, "status": "Fail", "msg": "❌ Section Too Small (Concrete Crush)"}
            else:
                rho = (0.85 * self.fc / self.fy) * (1 - np.sqrt(term))
                As_req = rho * self.b * self.d
                As_min = max(14/self.fy, 0.25*np.sqrt(self.fc)/self.fy) * self.b * self.d
                As_final = max(As_req, As_min)
                
                msg = "✅ Design OK" if As_req > As_min else "⚠️ Minimum Steel Governs"
                res_flex = {"As": As_final, "status": "Pass", "msg": msg}
            
            # Shear
            vc = 0.53 * np.sqrt(self.fc) * self.b * self.d
            phi_vc = 0.75 * vc
            
            if self.Vu > phi_vc:
                s_msg = "Stirrups Required"
            elif self.Vu > phi_vc/2:
                s_msg = "Min Stirrups"
            else:
                s_msg = "Theoretical None"
            res_shear = {"phiVc": phi_vc, "msg": s_msg}
            
        else: # WSD
            n = 135000/(15100*np.sqrt(self.fc))
            n = round(n) if n > 0 else 10
            r = (0.5*self.fy) / (0.45*self.fc)
            k = n / (n + r)
            j = 1 - k/3
            
            fs = 0.5 * self.fy
            As = Mu_kgcm / (fs * j * self.d)
            res_flex = {"As": As, "status": "Pass", "msg": "✅ WSD Method"}
            
            vc = 0.29 * np.sqrt(self.fc) * self.b * self.d
            res_shear = {"phiVc": vc, "msg": "Stirrups Req" if self.Vu > vc else "Concrete OK"}
            
        return res_flex, res_shear


# ==============================================================================
# 5. MAIN APP CONTROLLER
# ==============================================================================
def main():
    # --- Header ---
    st.markdown("""
        <div class="main-header">
            <h1>🏗️ RC Beam Pro: Senior Edition</h1>
            <p>Advanced Structural Analysis & Design System</p>
        </div>
    """, unsafe_allow_html=True)

    # --- Sidebar (Configuration) ---
    with st.sidebar:
        st.header("⚙️ Project Settings")
        method = st.radio("Design Method", ["SDM (Strength)", "WSD (Working)"])
        method_key = "SDM" if "SDM" in method else "WSD"
        
        st.markdown("---")
        with st.expander("🧱 Material Properties", expanded=True):
            fc = st.number_input("Concrete fc' (ksc)", value=240.0, step=10.0)
            fy = st.number_input("Rebar fy (ksc)", value=4000.0, step=100.0)
            
        with st.expander("📐 Section Dimensions", expanded=True):
            b = st.number_input("Width b (cm)", value=25.0)
            h = st.number_input("Depth h (cm)", value=50.0)
            cv = st.number_input("Covering (cm)", value=3.0)
            
        # Load Factors
        fdl = 1.4 if method_key == "SDM" else 1.0
        fll = 1.7 if method_key == "SDM" else 1.0
        if method_key == "SDM":
            st.info(f"Factors: DL={fdl}, LL={fll}")

    # --- INPUT SECTION (Robust Logic) ---
    col_geo, col_load = st.columns([1, 1.5], gap="large")

    # A. GEOMETRY INPUTS
    with col_geo:
        st.subheader("1. Geometry & Supports")
        n_spans = st.number_input("Number of Spans", min_value=1, max_value=20, value=2)
        
        # Initialize lists immediately to ensure scope validity
        spans_input = []
        supports_input_list = []
        
        # Grid layout for compactness
        st.write("<b>Span Lengths (m)</b>", unsafe_allow_html=True)
        cols_span = st.columns(2)
        for i in range(n_spans):
            with cols_span[i % 2]:
                val = st.number_input(f"Span {i+1}", min_value=0.5, value=5.0, key=f"span_len_{i}")
                spans_input.append(val)
        
        st.write("<b>Support Types</b>", unsafe_allow_html=True)
        cols_sup = st.columns(3)
        for i in range(n_spans + 1):
            with cols_sup[i % 3]:
                # Intelligent Defaults
                def_idx = 0 # Pin
                if i > 0 and i < n_spans: def_idx = 1 # Roller
                if i == n_spans: def_idx = 1 # Roller
                
                stype = st.selectbox(f"Node {i+1}", ["Pin", "Roller", "Fixed", "None"], index=def_idx, key=f"sup_type_{i}")
                supports_input_list.append(stype)

        # Create DataFrame here to catch errors early
        try:
            df_supports = pd.DataFrame({'type': supports_input_list})
        except Exception as e:
            st.error(f"Support Data Error: {e}")
            st.stop()

    # B. LOAD INPUTS
    with col_load:
        st.subheader("2. Applied Loads")
        loads_data = []
        
        tabs = st.tabs([f"Span {i+1}" for i in range(n_spans)])
        for i, tab in enumerate(tabs):
            with tab:
                c1, c2 = st.columns(2)
                
                # Uniform Load
                with c1:
                    st.markdown("##### 🟦 Uniform (kg/m)")
                    wdl = st.number_input("DL", key=f"udl_dl_{i}")
                    wll = st.number_input("LL", key=f"udl_ll_{i}")
                    wt = wdl*fdl + wll*fll
                    if wt != 0:
                        loads_data.append({'span_idx': i, 'type': 'U', 'w': wt})
                        st.caption(f"Total w = {wt:,.0f}")

                # Point Loads
                with c2:
                    st.markdown("##### 🔻 Point Loads (kg)")
                    count = st.number_input("Add Points", 0, 5, 0, key=f"p_cnt_{i}")
                    for j in range(count):
                        cc1, cc2, cc3 = st.columns([1,1,1.5])
                        pd = cc1.number_input("DL", key=f"pd_{i}_{j}")
                        pl = cc2.number_input("LL", key=f"pl_{i}_{j}")
                        px = cc3.number_input("x (m)", 0.0, spans_input[i], key=f"px_{i}_{j}")
                        
                        pt = pd*fdl + pl*fll
                        if pt != 0:
                            loads_data.append({'span_idx': i, 'type': 'P', 'P': pt, 'x': px})

    # --- EXECUTION ---
    st.markdown("---")
    
    # Stability Check
    valid_sups = [s for s in supports_input_list if s != 'None']
    is_stable = True
    if len(valid_sups) < 2 and "Fixed" not in supports_input_list:
        is_stable = False
        st.error("⚠️ Structure Unstable: Add at least 2 supports or 1 Fixed support.")

    if st.button("🚀 Analyze Structure", type="primary", disabled=not is_stable):
        with st.spinner("Calculating Stiffness & Forces..."):
            # 1. Run Analysis
            engine = BeamAnalysisEngine(spans_input, df_supports, loads_data, E=2e6, I=(b*h**3/12)*1e-8)
            df_res, df_reac = engine.solve()
            
            if df_res is None:
                st.error("❌ Singular Matrix: Structure is unstable or mechanism detected.")
                st.stop()
            
            # 2. Visualization
            st.subheader("Analysis Results")
            viz = DiagramVisualizer(df_res, spans_input, df_supports, loads_data)
            fig = viz.create_plot()
            st.plotly_chart(fig, use_container_width=True)
            
            # 3. Tables & Design
            c_res1, c_res2 = st.columns([1, 1.5])
            
            with c_res1:
                st.markdown("##### 🔹 Reactions")
                st.dataframe(
                    df_reac.style.format({"Ry (kg)": "{:,.2f}", "Mz (kg-m)": "{:,.2f}"}),
                    use_container_width=True, hide_index=True
                )
            
            with c_res2:
                st.markdown("##### 🔹 Design Check")
                v_max = df_res['shear'].abs().max()
                m_max = df_res['moment'].abs().max()
                
                # Summary Cards
                k1, k2 = st.columns(2)
                k1.markdown(f'<div class="metric-container"><div class="metric-label">Max Shear (Vu)</div><div class="metric-value">{v_max:,.0f}</div></div>', unsafe_allow_html=True)
                k2.markdown(f'<div class="metric-container"><div class="metric-label">Max Moment (Mu)</div><div class="metric-value">{m_max:,.0f}</div></div>', unsafe_allow_html=True)
                
                # Design Verification
                st.write("")
                des = RCDesigner(m_max, v_max, fc, fy, b, h, cv, method_key)
                rf, rs = des.check()
                
                # Flexure Result
                if rf['status'] == 'Fail':
                    st.error(rf['msg'])
                else:
                    st.success(f"{rf['msg']} (As req: {rf['As']:.2f} cm²)")
                    # Bar Suggestion
                    txt = " | ".join([f"DB{d}: {math.ceil(rf['As']/(3.14*(d/20)**2))}" for d in [12,16,20,25]])
                    st.caption(f"Suggestion: {txt}")
                
                # Shear Result
                st.info(f"Shear: {rs['msg']} (phiVc: {rs['phiVc']:,.0f} kg)")

if __name__ == "__main__":
    main()
