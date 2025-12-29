import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# ==============================================================================
# 1. PAGE CONFIGURATION & ADVANCED CSS STYLING
# ==============================================================================
st.set_page_config(
    page_title="RC Beam Pro: Titanium Edition",
    layout="wide",
    page_icon="🏗️",
    initial_sidebar_state="expanded"
)

# Inject High-Quality CSS for Professional Look
st.markdown("""
<style>
    /* Import Google Font: Sarabun for Thai Support */
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Sarabun', sans-serif;
    }

    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #01579B 0%, #0288D1 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-weight: 700;
        margin: 0;
        font-size: 2.2rem;
    }
    
    .main-header p {
        margin-top: 5px;
        opacity: 0.9;
        font-size: 1.1rem;
    }

    /* Section Headers */
    .section-title {
        border-left: 5px solid #0288D1;
        padding-left: 15px;
        font-size: 1.5rem;
        font-weight: 700;
        color: #01579B;
        margin-top: 30px;
        margin-bottom: 20px;
        background-color: #E1F5FE;
        padding-top: 10px;
        padding-bottom: 10px;
        border-radius: 0 8px 8px 0;
    }

    /* Card Styling for Inputs/Outputs */
    .stCard {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #E0E0E0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }

    /* Scrollable Container for Plotly */
    .plot-scroll-container {
        width: 100%;
        overflow-x: auto;
        white-space: nowrap;
        border: 1px solid #CFD8DC;
        border-radius: 8px;
        padding: 10px;
        background: white;
        box-shadow: inset 0 0 10px rgba(0,0,0,0.05);
    }
    
    /* Input Field Customization */
    .stNumberInput input {
        font-weight: 600;
        color: #0277BD;
    }
    
    /* Dataframe Styling */
    .dataframe {
        font-size: 0.9rem !important;
    }

    /* Success/Error Message Styling */
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. STRUCTURAL ANALYSIS CORE (MATRIX STIFFNESS METHOD)
# ==============================================================================
class StructuralEngine:
    """
    Core engine for Beam Analysis using Direct Stiffness Method.
    Handles unlimited spans, supports, and loading conditions.
    """
    def __init__(self, spans, supports, loads, E=2e6, I=1e-3):
        self.spans = spans         # List of span lengths [L1, L2, ...]
        self.supports = supports   # DataFrame of supports
        self.loads = loads         # List of load dictionaries
        self.E = E                 # Elastic Modulus
        self.I = I                 # Moment of Inertia
        
        # Generate Nodes
        # Node 0 is at x=0, Node i is at sum(L_0...L_i-1)
        self.node_coords = [0] + list(np.cumsum(spans))
        self.n_nodes = len(self.node_coords)
        self.dof = 2 * self.n_nodes # 2 DOF per node (Vertical Y, Rotation Theta)
        
    def _get_element_stiffness(self, L):
        """Calculate local stiffness matrix for a beam element."""
        k_const = self.E * self.I / L**3
        # Standard beam element (Bernoulli-Euler)
        # [Fy1, M1, Fy2, M2]
        K = np.array([
            [12,    6*L,    -12,    6*L],
            [6*L,   4*L**2, -6*L,   2*L**2],
            [-12,   -6*L,   12,     -6*L],
            [6*L,   2*L**2, -6*L,   4*L**2]
        ])
        return K * k_const

    def _compute_fem(self, span_idx):
        """Calculate Fixed End Moments (FEM) for a specific span."""
        L = self.spans[span_idx]
        fem_vec = np.zeros(4) # [Fy1, M1, Fy2, M2]
        
        # Filter loads on this span
        span_loads = [l for l in self.loads if l['span_idx'] == span_idx]
        
        for load in span_loads:
            if load['type'] == 'U': # Uniform Load
                w = load['w']
                # FEM for UDL
                # Fy = wL/2
                # M_left = wL^2/12 (CCW+)
                # M_right = -wL^2/12 (CW-)
                fem_vec += np.array([
                    w*L/2, 
                    w*L**2/12, 
                    w*L/2, 
                    -w*L**2/12
                ])
                
            elif load['type'] == 'P': # Point Load
                P = load['P']
                a = load['x']
                b = L - a
                # FEM for Point Load
                # Fy1 = Pb^2(3a+b)/L^3
                # M1  = Pab^2/L^2
                r1 = (P * b**2 * (3*a + b)) / L**3
                m1 = (P * a * b**2) / L**2
                r2 = (P * a**2 * (a + 3*b)) / L**3
                m2 = -(P * a**2 * b) / L**2
                
                fem_vec += np.array([r1, m1, r2, m2])
                
        return fem_vec

    def solve(self):
        """Main execution method to solve for displacements and forces."""
        
        # 1. Assemble Global Stiffness Matrix (K)
        K_global = np.zeros((self.dof, self.dof))
        
        for i, L in enumerate(self.spans):
            K_el = self._get_element_stiffness(L)
            
            # Map element DOF to global DOF
            # Element i connects Node i and Node i+1
            # DOFs: 2*i, 2*i+1, 2*i+2, 2*i+3
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            
            for r in range(4):
                for c in range(4):
                    K_global[idx[r], idx[c]] += K_el[r, c]
                    
        # 2. Assemble Global Force Vector (F)
        # F_global = F_nodal_loads - F_equivalent_member_loads (FEM)
        F_global = np.zeros(self.dof)
        
        # We assume no direct nodal loads in this UI (only member loads), 
        # so F_nodes = 0. We just subtract FEM.
        for i in range(len(self.spans)):
            fem = self._compute_fem(i)
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            F_global[idx] -= fem # Subtract FEM to get equivalent nodal forces
            
        # 3. Apply Boundary Conditions
        constrained_dof = []
        for node_i, row in self.supports.iterrows():
            stype = row['type']
            if stype == "Pin" or stype == "Roller":
                constrained_dof.append(2*node_i) # Constrain Vertical (Y)
            elif stype == "Fixed":
                constrained_dof.append(2*node_i)   # Constrain Vertical (Y)
                constrained_dof.append(2*node_i+1) # Constrain Rotation (Theta)
        
        free_dof = [d for d in range(self.dof) if d not in constrained_dof]
        
        if not free_dof:
            raise ValueError("Structure is fully constrained/rigid (No free DOF).")
            
        # 4. Partition and Solve
        # K_ff * D_f = F_f
        K_ff = K_global[np.ix_(free_dof, free_dof)]
        F_f = F_global[free_dof]
        
        # Solve for Displacements
        try:
            D_f = np.linalg.solve(K_ff, F_f)
        except np.linalg.LinAlgError:
            raise ValueError("Structure is unstable (Singular Matrix). Check supports.")
            
        # Reconstruct Full Displacement Vector
        D_total = np.zeros(self.dof)
        D_total[free_dof] = D_f
        
        # 5. Post-Processing (Internal Forces Calculation)
        # We calculate forces along the beam by superposition:
        # Internal(x) = Statics_from_End_Forces(x) + Local_Load_Effects(x)
        
        final_x_points = []
        final_shear = []
        final_moment = []
        
        current_x_offset = 0
        
        for i, L in enumerate(self.spans):
            # Get Displacements for this element
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            u_el = D_total[idx]
            
            # 1. Member End Forces from Stiffness (f = k*u)
            K_el = self._get_element_stiffness(L)
            f_stiff = np.dot(K_el, u_el)
            
            # 2. Add FEM to get total End Forces
            fem = self._compute_fem(i)
            f_total = f_stiff + fem # [V_left, M_left, V_right, M_right]
            
            # Start Statics from Left Node
            V_start = f_total[0]
            M_start = f_total[1] # Note: In Matrix, CCW is positive
            
            # Discretize span for plotting
            num_points = 100
            x_local = np.linspace(0, L, num_points)
            
            span_loads = [l for l in self.loads if l['span_idx'] == i]
            
            for x in x_local:
                # Basic Statics from left end
                # V(x) = V_start - sum(loads)
                # M(x) = -M_start + V_start*x - sum(load_moments)
                # Note: Beam sign convention vs Matrix sign convention
                # Matrix: CCW M is positive. Beam: Sagging M is positive.
                # Usually M_beam = -M_matrix_left + ...
                
                Vx = V_start
                Mx = -M_start + V_start * x
                
                # Apply local loads
                for l in span_loads:
                    if l['type'] == 'U':
                        # UDL starts at 0 for now (full span)
                        if x > 0:
                            wx = l['w'] * x
                            Vx -= wx
                            Mx -= wx * (x/2)
                    elif l['type'] == 'P':
                        if x > l['x']:
                            P = l['P']
                            dist = x - l['x']
                            Vx -= P
                            Mx -= P * dist
                
                final_x_points.append(current_x_offset + x)
                final_shear.append(Vx)
                final_moment.append(Mx)
                
            current_x_offset += L
            
        results_df = pd.DataFrame({
            'x': final_x_points,
            'shear': final_shear,
            'moment': final_moment
        })
        
        return results_df


# ==============================================================================
# 3. VISUALIZATION ENGINE (PLOTLY & SVG)
# ==============================================================================
class DiagramPlotter:
    def __init__(self, df_res, spans, supports, loads):
        self.df = df_res
        self.spans = spans
        self.supports = supports
        self.loads = loads
        self.cum_dist = [0] + list(np.cumsum(spans))
        self.total_len = self.cum_dist[-1]
        
    def _draw_support(self, fig, x, y, sup_type, size_w, size_h):
        """Draws engineering symbols using SVG paths for perfect scaling."""
        line_color = "#37474F"
        fill_color = "#CFD8DC"
        
        if sup_type == "Pin":
            # Triangle
            path = f"M {x},{y} L {x-size_w/2},{y-size_h} L {x+size_w/2},{y-size_h} Z"
            fig.add_shape(type="path", path=path, fillcolor=fill_color, line=dict(color=line_color, width=2), row=1, col=1)
            # Base Hatch
            fig.add_shape(type="line", x0=x-size_w, y0=y-size_h, x1=x+size_w, y1=y-size_h, line=dict(color=line_color, width=2), row=1, col=1)
            
        elif sup_type == "Roller":
            # Circle
            fig.add_shape(type="circle", x0=x-size_w/2, y0=y-size_h, x1=x+size_w/2, y1=y, fillcolor=fill_color, line=dict(color=line_color, width=2), row=1, col=1)
            # Base Line
            fig.add_shape(type="line", x0=x-size_w, y0=y-size_h, x1=x+size_w, y1=y-size_h, line=dict(color=line_color, width=2), row=1, col=1)
            
        elif sup_type == "Fixed":
            # Vertical thick line
            fig.add_shape(type="line", x0=x, y0=y-size_h, x1=x, y1=y+size_h, line=dict(color=line_color, width=4), row=1, col=1)
            # Hatching
            for i in range(5):
                dy = (2*size_h)/5 * i - size_h
                fig.add_shape(type="line", x0=x, y0=y+dy, x1=x-size_w/2, y1=y+dy+size_w/3, line=dict(color=line_color, width=1), row=1, col=1)

    def generate_plot(self):
        # 1. Calculate Scaling Factors to prevent distortion
        # Get Max Load for Y-axis normalization
        all_loads = [l['w'] for l in self.loads if l['type']=='U'] + [l['P'] for l in self.loads if l['type']=='P']
        max_load = max(map(abs, all_loads)) if all_loads else 100.0
        
        # Scale Parameters
        viz_height = max_load * 1.5
        sup_h = viz_height * 0.15 # Support height is 15% of view height
        sup_w = max(0.5, self.total_len * 0.02) # Support width relative to length but clamped
        
        # 2. Setup Subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.3, 0.35, 0.35],
            subplot_titles=(
                "<b>1. Loading Diagram (Free Body)</b>",
                "<b>2. Shear Force Diagram (V)</b>", 
                "<b>3. Bending Moment Diagram (M)</b>"
            )
        )
        
        # --- ROW 1: LOADING ---
        # Beam Axis
        fig.add_shape(type="line", x0=0, y0=0, x1=self.total_len, y1=0, line=dict(color="black", width=3), row=1, col=1)
        
        # Draw Supports
        for i, x in enumerate(self.cum_dist):
            if i < len(self.supports):
                stype = self.supports.iloc[i]['type']
                self._draw_support(fig, x, 0, stype, sup_w, sup_h)
        
        # Draw Loads
        for l in self.loads:
            if l['type'] == 'U':
                x1 = self.cum_dist[l['span_idx']]
                x2 = self.cum_dist[l['span_idx']+1]
                h = (l['w'] / max_load) * (viz_height * 0.5)
                
                # UDL Rect
                fig.add_trace(go.Scatter(
                    x=[x1, x2, x2, x1], y=[0, 0, h, h],
                    fill="toself", fillcolor="rgba(255, 152, 0, 0.3)",
                    line=dict(width=0), showlegend=False, hoverinfo="skip"
                ), row=1, col=1)
                
                # Top Line
                fig.add_trace(go.Scatter(x=[x1, x2], y=[h, h], mode='lines', line=dict(color='#EF6C00', width=2), showlegend=False), row=1, col=1)
                
                # Arrows
                num_arrows = max(3, int((x2-x1)*2))
                for xa in np.linspace(x1, x2, num_arrows+2)[1:-1]:
                    fig.add_annotation(
                        x=xa, y=0, ax=xa, ay=h, xref="x1", yref="y1", axref="x1", ayref="y1",
                        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor="#EF6C00",
                        row=1, col=1
                    )
                # Label
                fig.add_annotation(x=(x1+x2)/2, y=h*1.2, text=f"<b>w={l['w']:.0f}</b>", showarrow=False, font=dict(color="#EF6C00"), row=1, col=1)
                
            elif l['type'] == 'P':
                px = self.cum_dist[l['span_idx']] + l['x']
                h = (l['P'] / max_load) * (viz_height * 0.7)
                
                fig.add_annotation(
                    x=px, y=0, ax=px, ay=h, xref="x1", yref="y1", axref="x1", ayref="y1",
                    showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=2, arrowcolor="#D84315",
                    text=f"<b>P={l['P']:.0f}</b>", row=1, col=1
                )
        
        # --- ROW 2: SHEAR ---
        fig.add_trace(go.Scatter(
            x=self.df['x'], y=self.df['shear'],
            fill='tozeroy', line=dict(color='#0277BD', width=2),
            name="Shear"
        ), row=2, col=1)
        
        # --- ROW 3: MOMENT ---
        fig.add_trace(go.Scatter(
            x=self.df['x'], y=self.df['moment'],
            fill='tozeroy', line=dict(color='#C62828', width=2),
            name="Moment"
        ), row=3, col=1)
        
        # --- ANNOTATIONS (Max Values) ---
        for col, row, color in [('shear', 2, '#0277BD'), ('moment', 3, '#C62828')]:
            arr = self.df[col].values
            mx, mn = np.max(arr), np.min(arr)
            
            # Pad Y-axis
            rng = mx - mn
            pad = rng * 0.2 if rng > 0 else 1.0
            fig.update_yaxes(range=[mn-pad, mx+pad], row=row, col=1)
            
            # Annotate peaks
            for val in [mx, mn]:
                if abs(val) > 0.1:
                    idx = np.where(arr == val)[0][0]
                    x_pos = self.df['x'].iloc[idx]
                    fig.add_annotation(
                        x=x_pos, y=val, text=f"<b>{val:,.2f}</b>",
                        bgcolor="rgba(255,255,255,0.9)", bordercolor=color,
                        showarrow=False, yshift=10 if val>0 else -10,
                        row=row, col=1
                    )

        # --- LAYOUT FOR SCROLLING ---
        # Calculate width based on number of spans to force scrolling if needed
        # 250px per span, min 800px
        plot_width = max(1000, len(self.spans) * 300)
        
        fig.update_layout(
            width=plot_width, # Explicit width forces the container to scroll
            height=900,
            template="plotly_white",
            showlegend=False,
            margin=dict(l=40, r=40, t=40, b=40),
            hovermode="x unified"
        )
        
        # Add light gridlines at supports
        for x in self.cum_dist:
            fig.add_vline(x=x, line_dash="dot", line_color="gray", opacity=0.4)
            
        return fig


# ==============================================================================
# 4. RC DESIGN MODULE (DETAILED)
# ==============================================================================
class RCDesign:
    def __init__(self, Mu, Vu, fc, fy, b, h, cover, method="SDM"):
        self.Mu = abs(Mu)
        self.Vu = abs(Vu)
        self.fc = fc
        self.fy = fy
        self.b = b
        self.h = h
        self.d = h - cover
        self.method = method
        
    def check_flexure(self):
        """Perform Flexural Design (Reinforcement Calculation)."""
        # Convert units to kg-cm for calculation (assuming inputs are kg, m, ksc)
        # Mu is in kg-m -> *100 -> kg-cm
        Mu_calc = self.Mu * 100 
        
        if self.method == "SDM":
            phi = 0.90
            beta1 = 0.85 if self.fc <= 280 else max(0.65, 0.85 - 0.05*(self.fc-280)/70)
            
            # Required Strength Check (Rn)
            # Mu = phi * Mn
            # Mn = Rn * b * d^2
            Rn = Mu_calc / (phi * self.b * self.d**2)
            
            # Rho calculation
            term = 1 - (2 * Rn) / (0.85 * self.fc)
            
            if term < 0:
                return {
                    "status": "Fail",
                    "msg": "❌ Section too small (Compression Fail)",
                    "As": 0,
                    "rho": 0
                }
                
            rho = (0.85 * self.fc / self.fy) * (1 - np.sqrt(term))
            
            # Limits
            rho_min = max(14/self.fy, 0.25*np.sqrt(self.fc)/self.fy) # ACI Metric
            rho_bal = 0.85 * beta1 * (self.fc/self.fy) * (6120 / (6120 + self.fy))
            rho_max = 0.75 * rho_bal # Simplified ductilty limit
            
            As_req = rho * self.b * self.d
            As_min = rho_min * self.b * self.d
            
            final_As = max(As_req, As_min)
            
            status_msg = "✅ OK"
            if rho > rho_max:
                status_msg = "⚠️ Over Reinforced (Ductility Warning)"
            elif rho < rho_min:
                status_msg = "ℹ️ Minimum Steel Governs"
                
            return {
                "status": "Pass" if rho <= rho_max else "Warning",
                "msg": status_msg,
                "As_req": As_req,
                "As_min": As_min,
                "As_final": final_As,
                "rho": rho,
                "rho_max": rho_max
            }
            
        else: # WSD (Simplified)
            # n = Es/Ec ~ 135000 / (15100 sqrt(fc))
            n = 10 # approximate
            k = np.sqrt(2*0.45*n + (0.45*n)**2) - 0.45*n # Simplified k for balanced
            j = 1 - k/3
            fs = 0.5 * self.fy # Allowable
            
            As_wsd = Mu_calc / (fs * j * self.d)
            return {
                "status": "Pass",
                "msg": "✅ WSD Design",
                "As_final": As_wsd,
                "rho": As_wsd/(self.b*self.d)
            }

    def check_shear(self):
        """Perform Shear Design (Stirrup check)."""
        # Vu in kg
        if self.method == "SDM":
            phi_v = 0.75 # ACI Shear
            # Vc = 0.53 sqrt(fc) b d
            vc = 0.53 * np.sqrt(self.fc) * self.b * self.d
            phi_vc = phi_v * vc
            
            req = ""
            if self.Vu <= phi_vc / 2:
                req = "No Stirrups theoretically required"
            elif self.Vu <= phi_vc:
                req = "Minimum Stirrups required"
            else:
                req = "Design Stirrups required (Vs)"
                
            if self.Vu > phi_vc + (2.1 * np.sqrt(self.fc) * self.b * self.d):
                 req = "❌ Section Dimensions too small for Shear!"
                 
            return {"Vu": self.Vu, "phiVc": phi_vc, "Result": req}
        else:
            return {"Vu": self.Vu, "Result": "WSD Shear Check omitted"}

# ==============================================================================
# 5. UI COMPONENTS (SIDEBAR & INPUTS)
# ==============================================================================
def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Design Standards")
        
        # Design Method
        method = st.radio("Method", ["SDM (Strength Design)", "WSD (Working Stress)"], index=0)
        
        st.markdown("---")
        st.markdown("### 🧱 Material Properties")
        
        with st.expander("Concrete & Steel", expanded=True):
            fc = st.number_input("fc' (ksc)", value=240.0, step=10.0, format="%.1f")
            fy = st.number_input("fy Main (ksc)", value=4000.0, step=100.0, format="%.0f")
            fys = st.number_input("fy Stirrup (ksc)", value=2400.0, step=100.0, format="%.0f")
            
        with st.expander("Section Dimensions", expanded=True):
            b = st.number_input("Width b (cm)", value=25.0, step=5.0)
            h = st.number_input("Depth h (cm)", value=50.0, step=5.0)
            cv = st.number_input("Covering (cm)", value=3.0, step=0.5)
            
        st.markdown("### ⚖️ Safety Factors")
        if method == "SDM":
            c1, c2 = st.columns(2)
            fdl = c1.number_input("DL Factor", value=1.4)
            fll = c2.number_input("LL Factor", value=1.7)
            st.caption("Common ACI: 1.2D+1.6L or 1.4D+1.7L")
        else:
            fdl, fll = 1.0, 1.0
            st.info("Using Unfactored Loads (1.0)")
            
    return {
        'fc': fc, 'fy': fy, 'fys': fys, 
        'b': b, 'h': h, 'cv': cv,
        'fdl': fdl, 'fll': fll, 'method': method
    }

def render_inputs_grid(params):
    c1, c2 = st.columns([1, 1.4])
    
    # --- GEOMETRY INPUT ---
    with c1:
        st.markdown('<div class="section-title">1. Geometry & Supports</div>', unsafe_allow_html=True)
        with st.container():
            n_span = st.number_input("Number of Spans", min_value=1, max_value=20, value=2)
            
            # 1. Spans
            st.markdown("<b>Span Lengths (m)</b>", unsafe_allow_html=True)
            spans = []
            cols_s = st.columns(3)
            for i in range(n_span):
                val = cols_s[i%3].number_input(f"L{i+1}", min_value=0.5, value=6.0, key=f"s{i}")
                spans.append(val)
                
            # 2. Supports
            st.markdown("<b>Support Types (Left to Right)</b>", unsafe_allow_html=True)
            supports = []
            cols_sup = st.columns(3)
            opts = ["Pin", "Roller", "Fixed", "None"]
            for i in range(n_span + 1):
                # Intelligent default
                def_idx = 0 if i==0 else (1 if i < n_span else 1)
                val = cols_sup[i%3].selectbox(f"S{i+1}", opts, index=def_idx, key=f"sup{i}")
                supports.append(val)
                
            df_sup = pd.DataFrame({'type': supports})
            
            # Check stability basics
            active_sups = df_sup[df_sup['type'] != 'None']
            is_stable = True
            msg = "Stable"
            if len(active_sups) < 2 and "Fixed" not in active_sups['type'].values:
                is_stable = False
                msg = "Unstable (Need 2+ supports or Fixed)"
            elif len(active_sups) == 0:
                is_stable = False
                msg = "Floating Structure"
                
            if not is_stable:
                st.error(f"⚠️ Structure Error: {msg}")

    # --- LOADS INPUT ---
    with c2:
        st.markdown('<div class="section-title">2. Loading</div>', unsafe_allow_html=True)
        loads = []
        
        tabs = st.tabs([f"Span {i+1}" for i in range(n_span)])
        
        for i, tab in enumerate(tabs):
            with tab:
                st.info(f"Adding loads to Span {i+1} (Length: {spans[i]} m)")
                
                cc1, cc2 = st.columns(2)
                
                # UDL Section
                with cc1:
                    st.markdown("##### Uniform Load (kg/m)")
                    with st.expander("Add UDL", expanded=True):
                        wdl = st.number_input("Dead Load (w_dl)", value=1000.0, step=100.0, key=f"wdl{i}")
                        wll = st.number_input("Live Load (w_ll)", value=500.0, step=100.0, key=f"wll{i}")
                        
                        wu = wdl * params['fdl'] + wll * params['fll']
                        if wu != 0:
                            loads.append({'span_idx': i, 'type': 'U', 'w': wu})
                            st.metric("Factored w_u", f"{wu:,.0f} kg/m")
                            
                # Point Load Section
                with cc2:
                    st.markdown("##### Point Loads (kg)")
                    qty = st.number_input("Quantity", 0, 5, 0, key=f"q{i}")
                    for j in range(qty):
                        st.markdown(f"**Point Load #{j+1}**")
                        r1, r2, r3 = st.columns([1,1,1])
                        pd = r1.number_input(f"DL", key=f"pd{i}{j}")
                        pl = r2.number_input(f"LL", key=f"pl{i}{j}")
                        px = r3.number_input(f"x (m)", 0.0, spans[i], spans[i]/2.0, key=f"px{i}{j}")
                        
                        pu = pd * params['fdl'] + pl * params['fll']
                        if pu != 0:
                            loads.append({'span_idx': i, 'type': 'P', 'P': pu, 'x': px})

    return spans, df_sup, loads, is_stable

# ==============================================================================
# 6. MAIN APPLICATION LOGIC
# ==============================================================================
def main():
    st.markdown('<div class="main-header"><h1>🏗️ RC Beam Pro: Titanium Edition</h1><p>Professional Structural Analysis & Design Suite</p></div>', unsafe_allow_html=True)

    # 1. Inputs
    params = render_sidebar()
    spans, supports, loads, stable = render_inputs_grid(params)
    
    st.markdown("---")
    
    # 2. Run Button
    col_btn_1, col_btn_2, col_btn_3 = st.columns([1, 2, 1])
    with col_btn_2:
        run_analysis = st.button("🚀 RUN ANALYSIS & DESIGN", type="primary", use_container_width=True, disabled=not stable)
        
    if run_analysis and stable:
        try:
            with st.spinner("Solving Global Stiffness Matrix..."):
                # A. Analysis Phase
                solver = StructuralEngine(spans, supports, loads)
                df_results = solver.solve()
                
                # B. Visualization Phase
                plotter = DiagramPlotter(df_results, spans, supports, loads)
                fig = plotter.generate_plot()
                
                # C. Display Analysis Results
                st.markdown('<div class="section-title">3. Analysis Results (SFD & BMD)</div>', unsafe_allow_html=True)
                
                # --- SCROLLABLE PLOT CONTAINER ---
                # This HTML wrapper enables horizontal scrolling for large plots
                st.markdown(f"""
                <div class="plot-scroll-container">
                    {fig.to_html(full_html=False, include_plotlyjs='cdn')}
                </div>
                """, unsafe_allow_html=True)
                
                st.caption("💡 Tip: Scroll horizontally on the graph to see long beams.")
                
                # Extract Max Forces
                max_shear = df_results['shear'].abs().max()
                max_moment = df_results['moment'].abs().max()
                min_moment = df_results['moment'].min() # Negative moment
                
                c_res1, c_res2, c_res3 = st.columns(3)
                c_res1.metric("Max Shear (Vu)", f"{max_shear:,.2f} kg", delta_color="off")
                c_res2.metric("Max Positive Moment", f"{df_results['moment'].max():,.2f} kg-m", delta_color="off")
                c_res3.metric("Max Negative Moment", f"{min_moment:,.2f} kg-m", delta_color="off")
                
                # D. Design Phase
                st.markdown('<div class="section-title">4. RC Design Verification</div>', unsafe_allow_html=True)
                
                # Perform Design Check
                designer = RCDesign(
                    max_moment, max_shear, 
                    params['fc'], params['fy'], 
                    params['b'], params['h'], params['cv'], 
                    params['method']
                )
                
                flex_res = designer.check_flexure()
                shear_res = designer.check_shear()
                
                # Design Results Display
                d1, d2 = st.columns(2)
                
                with d1:
                    st.markdown("#### 🧱 Flexural Design (Moment)")
                    st.markdown(f"""
                    <div class="stCard">
                        <h3 style="color:{'#2E7D32' if 'Pass' in flex_res['status'] else '#C62828'}">{flex_res['msg']}</h3>
                        <table style="width:100%">
                            <tr><td><b>Design Mu:</b></td><td>{designer.Mu:,.2f} kg-m</td></tr>
                            <tr><td><b>Effective d:</b></td><td>{designer.d} cm</td></tr>
                            <tr><td><hr></td><td><hr></td></tr>
                            <tr><td><b>Required As:</b></td><td><b>{flex_res.get('As_final',0):.2f} cm²</b></td></tr>
                            <tr><td><small>Calculated As:</small></td><td><small>{flex_res.get('As_req',0):.2f}</small></td></tr>
                            <tr><td><small>Minimum As:</small></td><td><small>{flex_res.get('As_min',0):.2f}</small></td></tr>
                        </table>
                        <br>
                        <b>Rebar Suggestion:</b><br>
                        {'DB12'}: {math.ceil(flex_res.get('As_final',0)/1.13)} bars<br>
                        {'DB16'}: {math.ceil(flex_res.get('As_final',0)/2.01)} bars<br>
                        {'DB20'}: {math.ceil(flex_res.get('As_final',0)/3.14)} bars<br>
                        {'DB25'}: {math.ceil(flex_res.get('As_final',0)/4.91)} bars
                    </div>
                    """, unsafe_allow_html=True)
                    
                with d2:
                    st.markdown("#### ⛓️ Shear Design (Stirrup)")
                    st.markdown(f"""
                    <div class="stCard">
                        <h3>{shear_res['Result']}</h3>
                        <table style="width:100%">
                            <tr><td><b>Design Vu:</b></td><td>{shear_res['Vu']:,.2f} kg</td></tr>
                            <tr><td><b>Capacity (phi Vc):</b></td><td>{shear_res.get('phiVc', 0):,.2f} kg</td></tr>
                        </table>
                        <br>
                        <p><i>Note: Based on ACI 318 Simplified Shear Method. If Vu > phi*Vc/2, minimum stirrups are required.</i></p>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Calculation Error: {str(e)}")
            st.warning("Please check your inputs (Zero lengths, Floating structures, or Singular Matrix).")
            # For debugging
            # st.exception(e)

if __name__ == "__main__":
    main()
