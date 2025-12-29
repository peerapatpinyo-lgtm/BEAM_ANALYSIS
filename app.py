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
    page_title="RC Beam Pro: Enterprise Gold",
    layout="wide",
    page_icon="🏗️",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Engineering Report
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Sarabun', sans-serif;
    }

    /* Header Styling */
    .main-header {
        background: linear-gradient(90deg, #1565C0 0%, #0D47A1 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        font-weight: 700;
        margin: 0;
        font-size: 2.2rem;
    }
    .main-header p {
        margin: 5px 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
    }

    /* Section Styling */
    .section-title {
        border-left: 5px solid #1976D2;
        padding-left: 15px;
        font-size: 1.4rem;
        font-weight: 700;
        color: #1565C0;
        margin-top: 30px;
        margin-bottom: 15px;
        background-color: #F5F5F5;
        padding-top: 10px;
        padding-bottom: 10px;
        border-radius: 0 5px 5px 0;
    }

    /* Result Box Styling */
    .result-box {
        background-color: white;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .result-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #D32F2F;
    }
    .result-label {
        color: #616161;
        font-weight: 500;
        font-size: 1rem;
    }

    /* Plot Container */
    .plot-container {
        background: white;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #ddd;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }

    /* Input Fields */
    .stNumberInput input {
        font-weight: 600;
        color: #0277BD;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. STRUCTURAL ANALYSIS ENGINE (FULL MATRIX METHOD)
# ==============================================================================
class StructuralEngine:
    """
    Engine for solving continuous beams using Direct Stiffness Method.
    Includes full 4x4 DOF Matrix formulation.
    """
    def __init__(self, spans, supports, loads, E=2e6, I=1e-3):
        self.spans = spans
        self.supports = supports
        self.loads = loads
        self.E = E
        self.I = I
        
        # Node Coordinates
        self.node_coords = [0] + list(np.cumsum(spans))
        self.n_nodes = len(self.node_coords)
        self.dof = 2 * self.n_nodes  # 2 DOF per node (Vertical, Rotation)

    def _get_element_stiffness(self, L):
        """Construct local stiffness matrix for a beam element."""
        k = (self.E * self.I) / (L**3)
        K = np.array([
            [12,    6*L,    -12,    6*L],
            [6*L,   4*L**2, -6*L,   2*L**2],
            [-12,   -6*L,   12,     -6*L],
            [6*L,   2*L**2, -6*L,   4*L**2]
        ])
        return K * k

    def _compute_fem(self, span_idx):
        """Calculate Fixed End Moments (FEM) and Forces."""
        L = self.spans[span_idx]
        fem_vec = np.zeros(4) # [Fy1, M1, Fy2, M2]
        
        span_loads = [l for l in self.loads if l['span_idx'] == span_idx]
        
        for load in span_loads:
            if load['type'] == 'U':
                w = load['w']
                # UDL Formulas
                fem_vec += np.array([
                    w*L/2,      # Fy1
                    w*L**2/12,  # M1
                    w*L/2,      # Fy2
                    -w*L**2/12  # M2
                ])
            elif load['type'] == 'P':
                P = load['P']
                a = load['x']
                b = L - a
                # Point Load Formulas
                fem_vec += np.array([
                    (P*b**2*(3*a+b))/L**3, # Fy1
                    (P*a*b**2)/L**2,       # M1
                    (P*a**2*(a+3*b))/L**3, # Fy2
                    -(P*a**2*b)/L**2       # M2
                ])
        return fem_vec

    def solve(self):
        """Main solver logic."""
        
        # 1. Global Stiffness Assembly
        K_global = np.zeros((self.dof, self.dof))
        for i, L in enumerate(self.spans):
            K_el = self._get_element_stiffness(L)
            # Map local to global indices
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            for r in range(4):
                for c in range(4):
                    K_global[idx[r], idx[c]] += K_el[r, c]

        # 2. Global Force Vector Assembly
        F_global = np.zeros(self.dof)
        F_fem_equiv = np.zeros(self.dof) # Store to calc reactions later
        
        for i in range(len(self.spans)):
            fem = self._compute_fem(i)
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            F_global[idx] -= fem  # Subtract FEM from nodes
            F_fem_equiv[idx] += fem 

        # 3. Apply Boundary Conditions
        constrained_dof = []
        for node_i, row in self.supports.iterrows():
            stype = row['type']
            if stype == "Pin" or stype == "Roller":
                constrained_dof.append(2*node_i) # Constrain Vertical (Y)
            elif stype == "Fixed":
                constrained_dof.extend([2*node_i, 2*node_i+1]) # Constrain Y and Rotation
        
        free_dof = [d for d in range(self.dof) if d not in constrained_dof]
        
        # Check stability
        if not free_dof or len(constrained_dof) < 2:
            if "Fixed" not in self.supports['type'].values and len(constrained_dof) < 3:
                 pass # Simple check
        
        # 4. Solve System of Equations
        try:
            K_reduced = K_global[np.ix_(free_dof, free_dof)]
            F_reduced = F_global[free_dof]
            D_free = np.linalg.solve(K_reduced, F_reduced)
        except np.linalg.LinAlgError:
            return None, None # Unstable
            
        D_total = np.zeros(self.dof)
        D_total[free_dof] = D_free
        
        # 5. Calculate Reactions
        # Reaction = K_global * Displacement + FEM_equivalent
        R_vec = np.dot(K_global, D_total) + F_fem_equiv
        
        reactions_data = []
        for i, row in self.supports.iterrows():
            if row['type'] != 'None':
                reactions_data.append({
                    "Node": f"{i+1}",
                    "Support Type": row['type'],
                    "Vertical Reaction (kg)": R_vec[2*i],
                    "Moment Reaction (kg-m)": R_vec[2*i+1]
                })
        
        # 6. Calculate Internal Forces (Post-Processing)
        final_x = []
        final_v = []
        final_m = []
        
        current_x = 0
        
        for i, L in enumerate(self.spans):
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            u_el = D_total[idx]
            
            # Element end forces from stiffness
            K_el = self._get_element_stiffness(L)
            f_stiff = np.dot(K_el, u_el)
            
            # Total end forces = stiffness forces + FEM
            f_total = f_stiff + self._compute_fem(i)
            
            V_start = f_total[0]
            M_start = f_total[1]
            
            # Discretize span for plotting
            x_pts = np.linspace(0, L, 50) 
            span_loads = [l for l in self.loads if l['span_idx'] == i]
            
            for x in x_pts:
                # Equilibrium at x
                Vx = V_start
                Mx = -M_start + V_start * x # Beam convention
                
                for l in span_loads:
                    if l['type'] == 'U':
                        if x > 0:
                            Vx -= l['w'] * x
                            Mx -= l['w'] * x**2 / 2
                    elif l['type'] == 'P':
                        if x > l['x']:
                            Vx -= l['P']
                            Mx -= l['P'] * (x - l['x'])
                
                final_x.append(current_x + x)
                final_v.append(Vx)
                final_m.append(Mx)
            
            current_x += L
            
        return pd.DataFrame({'x': final_x, 'shear': final_v, 'moment': final_m}), pd.DataFrame(reactions_data)


# ==============================================================================
# 3. VISUALIZATION MODULE (FIXED BEAM LINE & LABELS)
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
        # Scale Calculation for Aesthetics
        load_vals = [l['w'] for l in self.loads if l['type']=='U'] + [l['P'] for l in self.loads if l['type']=='P']
        max_load = max(map(abs, load_vals)) if load_vals else 100
        
        vh = max_load * 1.5 if max_load > 0 else 10
        sw = max(0.4, self.total_len * 0.02) # Support width
        sh = vh * 0.15 # Support height

        # Initialize Subplots
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True, 
            row_heights=[0.3, 0.35, 0.35],
            vertical_spacing=0.08,
            subplot_titles=(
                "<b>1. Loading Diagram (Free Body)</b>", 
                "<b>2. Shear Force Diagram (SFD)</b>", 
                "<b>3. Bending Moment Diagram (BMD)</b>"
            )
        )

        # --- ROW 1: BEAM LOADING ---
        
        # [FIX] Draw Beam Line using Scatter (Not Shape) to ensure visibility
        fig.add_trace(go.Scatter(
            x=[0, self.total_len], 
            y=[0, 0],
            mode='lines',
            line=dict(color='black', width=5), # Thick black line
            name='Beam',
            hoverinfo='none'
        ), row=1, col=1)

        # Draw Supports
        for i, x in enumerate(self.cum_dist):
            if i < len(self.supports):
                self._draw_support_shape(fig, x, 0, self.supports.iloc[i]['type'], sw, sh)

        # Draw Loads
        for l in self.loads:
            if l['type'] == 'U':
                x1 = self.cum_dist[l['span_idx']]
                x2 = self.cum_dist[l['span_idx']+1]
                h = (l['w'] / max_load) * vh * 0.6
                
                # Filled area for UDL
                fig.add_trace(go.Scatter(
                    x=[x1, x2, x2, x1], 
                    y=[0, 0, h, h],
                    fill='toself', 
                    fillcolor='rgba(255, 152, 0, 0.3)',
                    line_width=0,
                    showlegend=False,
                    hoverinfo='skip'
                ), row=1, col=1)
                
                # Top line for UDL
                fig.add_trace(go.Scatter(
                    x=[x1, x2], y=[h, h],
                    mode='lines',
                    line=dict(color='#E65100', width=2),
                    showlegend=False
                ), row=1, col=1)
                
                # Text Label
                fig.add_annotation(
                    x=(x1+x2)/2, y=h, 
                    text=f"<b>w={l['w']:,.0f}</b>", 
                    yshift=15, showarrow=False, 
                    font=dict(color='#E65100', size=11), 
                    row=1, col=1
                )
            
            elif l['type'] == 'P':
                px = self.cum_dist[l['span_idx']] + l['x']
                h = (l['P'] / max_load) * vh * 0.7
                
                # Arrow Annotation
                fig.add_annotation(
                    x=px, y=0, 
                    ax=px, ay=h if h > 0 else 20, 
                    xref="x1", yref="y1", axref="x1", ayref="y1",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#BF360C",
                    text=f"<b>P={l['P']:,.0f}</b>",
                    row=1, col=1
                )

        # Fix Y-Axis range for Row 1 to keep beam centered
        fig.update_yaxes(visible=False, fixedrange=True, range=[-sh*1.5, vh*1.3], row=1, col=1)


        # --- ROW 2 & 3: SFD & BMD ---
        
        # Helper to plot graph with labels
        def plot_diagram(row_idx, col_name, color, title, unit):
            fig.add_trace(go.Scatter(
                x=self.df['x'], y=self.df[col_name],
                fill='tozeroy',
                line=dict(color=color, width=2),
                name=title
            ), row=row_idx, col=1)
            
            # Find Min/Max
            vals = self.df[col_name].values
            mx = np.max(vals)
            mn = np.min(vals)
            
            # Padding
            rng = mx - mn
            pad = rng * 0.2 if rng > 0 else 1.0
            fig.update_yaxes(range=[mn-pad, mx+pad], row=row_idx, col=1)
            
            # Add Max/Min Labels
            for val in [mx, mn]:
                if abs(val) > 0.01: # Filter noise
                    idx = np.argmax(vals == val) if val == mx else np.argmin(vals == val)
                    x_pos = self.df['x'].iloc[idx]
                    
                    fig.add_annotation(
                        x=x_pos, y=val,
                        text=f"<b>{val:,.2f} {unit}</b>",
                        bgcolor="rgba(255,255,255,0.85)",
                        bordercolor=color, borderwidth=1,
                        showarrow=False,
                        yshift=15 if val >= 0 else -15,
                        font=dict(color=color, size=11),
                        row=row_idx, col=1
                    )

        plot_diagram(2, 'shear', '#1976D2', 'Shear', 'kg')
        plot_diagram(3, 'moment', '#D32F2F', 'Moment', 'kg-m')

        # Global Layout
        fig.update_layout(
            height=900,
            template="plotly_white",
            showlegend=False,
            hovermode="x unified",
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Add Vertical Grid Lines at Supports
        for x in self.cum_dist:
            fig.add_vline(x=x, line_dash="dot", line_color="gray", opacity=0.5)

        return fig

    def _draw_support_shape(self, fig, x, y, stype, w, h):
        """Helper to draw engineering symbols for supports."""
        lc, fc = "#37474F", "#B0BEC5"
        
        if stype == "Pin":
            # Triangle
            path = f"M {x},{y} L {x-w/2},{y-h} L {x+w/2},{y-h} Z"
            fig.add_shape(type="path", path=path, fillcolor=fc, line=dict(color=lc, width=2), row=1, col=1)
            # Base
            fig.add_shape(type="line", x0=x-w, y0=y-h, x1=x+w, y1=y-h, line=dict(color=lc, width=2), row=1, col=1)
            
        elif stype == "Roller":
            # Circle
            fig.add_shape(type="circle", x0=x-w/2, y0=y-h, x1=x+w/2, y1=y, fillcolor=fc, line=dict(color=lc, width=2), row=1, col=1)
            # Base
            fig.add_shape(type="line", x0=x-w, y0=y-h, x1=x+w, y1=y-h, line=dict(color=lc, width=2), row=1, col=1)
            
        elif stype == "Fixed":
            # Vertical Line
            fig.add_shape(type="line", x0=x, y0=y-h, x1=x, y1=y+h, line=dict(color=lc, width=4), row=1, col=1)
            # Hatching
            for k in range(5):
                dy = (2*h)/5 * k - h
                fig.add_shape(type="line", x0=x, y0=y+dy, x1=x-w/2, y1=y+dy+w/3, line=dict(color=lc, width=1), row=1, col=1)


# ==============================================================================
# 4. RC DESIGN MODULE
# ==============================================================================
class RCDesign:
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
            
            # Check Concrete Crushing
            limit = (2 * Rn) / (0.85 * self.fc)
            term = 1 - limit
            
            if term < 0:
                return {"status": "Fail", "msg": "Section Dimensions Too Small (Concrete Fail)"}, {"status": "Check"}
            
            rho_req = (0.85 * self.fc / self.fy) * (1 - np.sqrt(term))
            As_req = rho_req * self.b * self.d
            
            # Min Rebar
            As_min = max(14/self.fy, 0.25*np.sqrt(self.fc)/self.fy) * self.b * self.d
            As_final = max(As_req, As_min)
            
            res_flex = {
                "As": As_final,
                "msg": "Design OK" if As_req > As_min else "Minimum Steel Governs"
            }
            
            # Shear
            vc = 0.53 * np.sqrt(self.fc) * self.b * self.d
            phi_vc = 0.75 * vc
            
            req_shear = ""
            if self.Vu > phi_vc:
                req_shear = "Stirrups Required (Calculated)"
            elif self.Vu > phi_vc / 2:
                req_shear = "Minimum Stirrups Required"
            else:
                req_shear = "No Stirrups Required (Theoretical)"
                
            res_shear = {"phiVc": phi_vc, "req": req_shear}
            
        else:
            # WSD
            n = 135000 / (15100 * np.sqrt(self.fc)) # Approximate modular ratio
            n = round(n) if n > 0 else 10
            
            k = np.sqrt(2*0.01*n + (0.01*n)**2) - 0.01*n # Simplified k
            j = 1 - k/3
            # Use standard constants for simplicity if exact not needed
            j = 0.875 
            fs = 0.5 * self.fy
            
            As_req = Mu_kgcm / (fs * j * self.d)
            res_flex = {"As": As_req, "msg": "WSD Method"}
            
            vc_allow = 0.29 * np.sqrt(self.fc)
            Vc = vc_allow * self.b * self.d
            res_shear = {
                "phiVc": Vc, 
                "req": "Stirrups Required" if self.Vu > Vc else "Concrete OK"
            }
            
        return res_flex, res_shear


# ==============================================================================
# 5. MAIN APPLICATION (UI)
# ==============================================================================
def main():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🏗️ RC Beam Pro: Enterprise Gold</h1>
            <p>Complete Structural Analysis & Design Suite</p>
        </div>
    """, unsafe_allow_html=True)

    # --- SIDEBAR: SETTINGS ---
    with st.sidebar:
        st.header("⚙️ Design Parameters")
        method = st.radio("Design Method", ["SDM (Strength Design)", "WSD (Working Stress)"])
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
        if method_key == "SDM":
            st.markdown("---")
            st.write("**Load Factors:**")
            fdl = st.number_input("Dead Load Factor", 1.4)
            fll = st.number_input("Live Load Factor", 1.7)
        else:
            fdl, fll = 1.0, 1.0

    # --- INPUT SECTION ---
    c_geo, c_load = st.columns([1, 1.5], gap="large")

    # 1. Geometry Inputs
    with c_geo:
        st.markdown('<div class="section-title">1. Geometry & Supports</div>', unsafe_allow_html=True)
        
        n_spans = st.number_input("Number of Spans", min_value=1, max_value=15, value=2)
        
        # Collect Spans
        st.write("<b>Span Lengths (m)</b>", unsafe_allow_html=True)
        spans_input = []
        cols_spans = st.columns(2)
        for i in range(n_spans):
            val = cols_spans[i%2].number_input(f"L{i+1}", min_value=0.1, value=5.0, key=f"span_{i}")
            spans_input.append(val)
            
        # Collect Supports
        st.write("<b>Support Conditions</b>", unsafe_allow_html=True)
        supports_input = []
        cols_sup = st.columns(3)
        for i in range(n_spans + 1):
            # Logic for default support types
            default_idx = 0 # Pin
            if i > 0 and i < n_spans: default_idx = 1 # Roller
            if i == n_spans: default_idx = 1 # Roller
            
            stype = cols_sup[i%3].selectbox(
                f"Node {i+1}", 
                ["Pin", "Roller", "Fixed", "None"], 
                index=default_idx, 
                key=f"sup_{i}"
            )
            supports_input.append(stype)
            
        # Create DataFrame for Supports immediately
        df_supports = pd.DataFrame({'type': supports_input})

    # 2. Load Inputs
    with c_load:
        st.markdown('<div class="section-title">2. Applied Loads</div>', unsafe_allow_html=True)
        
        loads_data = []
        
        tabs = st.tabs([f"Span {i+1}" for i in range(n_spans)])
        for i, tab in enumerate(tabs):
            with tab:
                c1, c2 = st.columns(2)
                
                # UDL Input
                with c1:
                    st.info("Uniform Load (kg/m)")
                    wdl = st.number_input("Dead Load (DL)", key=f"wdl_{i}")
                    wll = st.number_input("Live Load (LL)", key=f"wll_{i}")
                    
                    w_total = wdl * fdl + wll * fll
                    if w_total != 0:
                        loads_data.append({'span_idx': i, 'type': 'U', 'w': w_total})
                        st.caption(f"Factored w = {w_total:,.2f} kg/m")
                
                # Point Load Input
                with c2:
                    st.warning("Point Loads (kg)")
                    qty_p = st.number_input("Add Point Load?", 0, 3, 0, key=f"qp_{i}")
                    for j in range(qty_p):
                        p_col = st.columns([1, 1, 1.5])
                        pd = p_col[0].number_input(f"DL", key=f"pd_{i}_{j}")
                        pl = p_col[1].number_input(f"LL", key=f"pl_{i}_{j}")
                        px = p_col[2].number_input(f"x (m)", 0.0, spans_input[i], key=f"px_{i}_{j}")
                        
                        p_total = pd * fdl + pl * fll
                        if p_total != 0:
                            loads_data.append({'span_idx': i, 'type': 'P', 'P': p_total, 'x': px})

    # --- PROCESS ---
    st.markdown("---")
    
    # Check Stability
    valid_sup = [s for s in supports_input if s != "None"]
    is_stable = True
    if len(valid_sup) < 2 and "Fixed" not in supports_input:
        is_stable = False
        st.error("⚠️ Structure Unstable: Please add at least 2 supports or 1 Fixed support.")

    if st.button("🚀 Run Analysis & Design", type="primary", disabled=not is_stable):
        with st.spinner("Analyzing Structure..."):
            # 1. Structural Analysis
            engine = StructuralEngine(spans_input, df_supports, loads_data, E=2e6, I=(b*h**3/12)*1e-8)
            df_res, df_reac = engine.solve()
            
            if df_res is None:
                st.error("Error: Matrix Singular. Structure is unstable.")
                st.stop()

            # 2. Display Plot
            st.markdown('<div class="section-title">3. Analysis Diagrams</div>', unsafe_allow_html=True)
            plotter = DiagramPlotter(df_res, spans_input, df_supports, loads_data)
            fig = plotter.plot()
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # 3. Results Tables
            col_tbl1, col_tbl2 = st.columns([1, 1.5])
            
            with col_tbl1:
                st.markdown('<div class="section-title">4. Support Reactions</div>', unsafe_allow_html=True)
                st.dataframe(
                    df_reac.style.format({
                        "Vertical Reaction (kg)": "{:,.2f}", 
                        "Moment Reaction (kg-m)": "{:,.2f}"
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
            with col_tbl2:
                st.markdown('<div class="section-title">5. Design Forces</div>', unsafe_allow_html=True)
                v_max = df_res['shear'].abs().max()
                m_max = df_res['moment'].abs().max()
                
                c_m1, c_m2 = st.columns(2)
                c_m1.markdown(f"""
                    <div class="result-box">
                        <div class="result-label">Max Shear (Vu)</div>
                        <div class="result-value">{v_max:,.0f} kg</div>
                    </div>
                """, unsafe_allow_html=True)
                c_m2.markdown(f"""
                    <div class="result-box">
                        <div class="result-label">Max Moment (Mu)</div>
                        <div class="result-value">{m_max:,.0f} kg-m</div>
                    </div>
                """, unsafe_allow_html=True)

            # 4. RC Design Logic
            st.markdown('<div class="section-title">6. RC Design Check</div>', unsafe_allow_html=True)
            designer = RCDesign(m_max, v_max, fc, fy, b, h, cv, method_key)
            res_flex, res_shear = designer.check()
            
            d1, d2 = st.columns(2)
            with d1:
                st.subheader("🧱 Flexural Design (Main Steel)")
                if res_flex.get('status') == 'Fail':
                    st.error(f"❌ {res_flex['msg']}")
                else:
                    st.success(f"✅ {res_flex['msg']}")
                    st.write(f"Required As: **{res_flex['As']:,.2f} cm²**")
                    
                    # Rebar Suggestion
                    st.caption("Suggested Reinforcement:")
                    bars = [12, 16, 20, 25]
                    bar_txt = ""
                    for d in bars:
                        area = 3.1416 * (d/20)**2
                        num = math.ceil(res_flex['As'] / area)
                        bar_txt += f"**DB{d}**: {num} bars | "
                    st.write(bar_txt.strip(" | "))

            with d2:
                st.subheader("⛓️ Shear Design (Stirrups)")
                st.info(f"Result: **{res_shear['req']}**")
                st.write(f"Design Vu: {v_max:,.0f} kg")
                st.write(f"Capacity phiVc: {res_shear['phiVc']:,.0f} kg")

if __name__ == "__main__":
    main()
