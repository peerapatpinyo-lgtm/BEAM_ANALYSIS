import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# ==============================================================================
# 1. PAGE CONFIGURATION & CSS STYLING (SYSTEM SETUP)
# ==============================================================================
st.set_page_config(
    page_title="RC Beam Pro: Enterprise Edition",
    layout="wide",
    page_icon="🏗️",
    initial_sidebar_state="expanded"
)

# Advanced CSS for Professional Engineering Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Sarabun', sans-serif;
    }

    /* Main Header Styling */
    .header-container {
        background: linear-gradient(135deg, #0D47A1 0%, #1976D2 100%);
        color: white;
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    .header-container h1 {
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 10px;
    }
    .header-container p {
        font-size: 1.1rem;
        opacity: 0.9;
    }

    /* Section Headings */
    .section-header {
        border-left: 6px solid #1565C0;
        padding-left: 15px;
        font-size: 1.6rem;
        font-weight: 700;
        color: #0D47A1;
        margin-top: 40px;
        margin-bottom: 20px;
        background-color: #E3F2FD;
        padding-top: 10px;
        padding-bottom: 10px;
        border-radius: 0 8px 8px 0;
    }

    /* Plot Container with Horizontal Scroll */
    .scrollable-plot-container {
        width: 100%;
        overflow-x: auto;
        white-space: nowrap;
        border: 1px solid #CFD8DC;
        border-radius: 10px;
        padding: 15px;
        background: white;
        box-shadow: inset 0 0 15px rgba(0,0,0,0.03);
        margin-bottom: 20px;
    }
    
    /* Result Cards */
    .result-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #E0E0E0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        text-align: center;
    }
    .result-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1565C0;
    }
    .result-label {
        font-size: 1rem;
        color: #616161;
        font-weight: 500;
    }

    /* Input Fields Customization */
    .stNumberInput input {
        font-weight: 600;
        color: #0277BD;
    }
    
    /* Tables */
    .dataframe {
        font-size: 1rem !important;
        border-radius: 5px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. ANALYSIS CORE: FINITE ELEMENT / MATRIX STIFFNESS METHOD
# ==============================================================================
class BeamAnalyzer:
    """
    Advanced Beam Analysis Engine using Direct Stiffness Method.
    Features:
    - Exact solution for multi-span beams
    - Handle Pin, Roller, Fixed supports
    - Point Loads and Uniform Loads
    - Reaction Force Calculation
    """
    def __init__(self, spans, supports, loads, E=2e6, I=1e-3):
        self.spans = spans  # List of span lengths
        self.supports = supports  # DataFrame of support types
        self.loads = loads  # List of load dictionaries
        self.E = E  # Modulus of Elasticity (ksc) -> Consistent units required
        self.I = I  # Moment of Inertia (m4)
        
        # Calculate Node Coordinates
        # Node 0 is at x=0
        self.nodes = [0] + list(np.cumsum(spans))
        self.num_nodes = len(self.nodes)
        self.dof_total = 2 * self.num_nodes  # 2 DOF per node (Deflection Y, Rotation Z)
        
    def _element_stiffness_matrix(self, L):
        """Constructs the local stiffness matrix for a beam element."""
        k = (self.E * self.I) / (L**3)
        
        # Matrix 4x4 for [v1, theta1, v2, theta2]
        # v = vertical displacement, theta = rotation
        K_local = np.array([
            [12,      6*L,     -12,     6*L],
            [6*L,     4*L**2,  -6*L,    2*L**2],
            [-12,     -6*L,    12,      -6*L],
            [6*L,     2*L**2,  -6*L,    4*L**2]
        ])
        
        return K_local * k

    def _calculate_fem(self, span_idx):
        """Calculates Fixed End Moments (FEM) for loads on a span."""
        L = self.spans[span_idx]
        fem_vector = np.zeros(4)  # [Fy_L, M_L, Fy_R, M_R]
        
        # Filter loads applied to this specific span
        current_span_loads = [l for l in self.loads if l['span_idx'] == span_idx]
        
        for load in current_span_loads:
            if load['type'] == 'U':
                # Uniform Distributed Load (w)
                w = load['w']
                # Formulas: Fy = wL/2, M = wL^2/12
                # Left Node (CCW +)
                fem_vector[0] += w * L / 2
                fem_vector[1] += w * L**2 / 12
                # Right Node
                fem_vector[2] += w * L / 2
                fem_vector[3] += -w * L**2 / 12  # Clockwise is negative in matrix math
                
            elif load['type'] == 'P':
                # Point Load (P) at distance 'a' from left
                P = load['P']
                a = load['x']
                b = L - a
                
                # Formulas for Point Load FEM
                # Left: Fy = Pb^2(3a+b)/L^3, M = Pab^2/L^2
                # Right: Fy = Pa^2(a+3b)/L^3, M = -Pa^2b/L^2
                
                # Force Left
                fem_vector[0] += (P * (b**2) * (3*a + b)) / (L**3)
                # Moment Left
                fem_vector[1] += (P * a * (b**2)) / (L**2)
                # Force Right
                fem_vector[2] += (P * (a**2) * (a + 3*b)) / (L**3)
                # Moment Right
                fem_vector[3] += -(P * (a**2) * b) / (L**2)
                
        return fem_vector

    def analyze(self):
        """Main execution function to solve the system."""
        
        # 1. Initialize Global Stiffness Matrix (K)
        K_global = np.zeros((self.dof_total, self.dof_total))
        
        # 2. Assemble K Global
        for i, length in enumerate(self.spans):
            K_element = self._element_stiffness_matrix(length)
            
            # Map element DOF indices to Global DOF indices
            # Node i -> Indices 2*i, 2*i+1
            # Node i+1 -> Indices 2*(i+1), 2*(i+1)+1
            # Combined for element i: [2i, 2i+1, 2i+2, 2i+3]
            indices = [2*i, 2*i+1, 2*i+2, 2*i+3]
            
            for row in range(4):
                for col in range(4):
                    K_global[indices[row], indices[col]] += K_element[row, col]
                    
        # 3. Assemble Force Vector (F)
        # In Matrix method: F_nodes = K * D + FEM
        # So, K * D = F_nodes - FEM
        # We assume no external nodal loads (only member loads), so F_nodes = 0
        # Thus, F_solving = -FEM
        
        F_solving = np.zeros(self.dof_total)
        F_accumulated_fem = np.zeros(self.dof_total) # Store for reaction calc
        
        for i in range(len(self.spans)):
            fem_vec = self._calculate_fem(i)
            indices = [2*i, 2*i+1, 2*i+2, 2*i+3]
            
            F_solving[indices] -= fem_vec
            F_accumulated_fem[indices] += fem_vec
            
        # 4. Apply Boundary Conditions
        constrained_dofs = []
        
        for i, row in self.supports.iterrows():
            support_type = row['type']
            node_idx_y = 2 * i
            node_idx_theta = 2 * i + 1
            
            if support_type == "Pin":
                constrained_dofs.append(node_idx_y) # Fix Y
            elif support_type == "Roller":
                constrained_dofs.append(node_idx_y) # Fix Y
            elif support_type == "Fixed":
                constrained_dofs.append(node_idx_y) # Fix Y
                constrained_dofs.append(node_idx_theta) # Fix Rotation
        
        # Identify Free DOFs
        all_dofs = list(range(self.dof_total))
        free_dofs = [d for d in all_dofs if d not in constrained_dofs]
        
        # Check for Stability
        if len(free_dofs) == 0 or len(constrained_dofs) == 0:
            return None, None  # Unstable or fully rigid without freedom
            
        # 5. Solve for Displacements (D)
        # K_free * D_free = F_free
        K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
        F_reduced = F_solving[free_dofs]
        
        try:
            D_calculated = np.linalg.solve(K_reduced, F_reduced)
        except np.linalg.LinAlgError:
            return None, None # Matrix Singular (Unstable structure)
            
        # Map back to full displacement vector
        D_global = np.zeros(self.dof_total)
        D_global[free_dofs] = D_calculated
        
        # 6. Calculate Reactions
        # R = K_global * D_global + FEM_accumulated
        # (Force exerted by the structure on the nodes)
        Reactions_vector = np.dot(K_global, D_global) + F_accumulated_fem
        
        reaction_results = []
        for i, row in self.supports.iterrows():
            if row['type'] != 'None':
                ry = Reactions_vector[2*i]
                mz = Reactions_vector[2*i+1]
                reaction_results.append({
                    "Node": i+1,
                    "Support Type": row['type'],
                    "Vertical Reaction (Ry)": ry,
                    "Moment Reaction (Mz)": mz
                })
        
        df_reactions = pd.DataFrame(reaction_results)

        # 7. Post-Processing: Calculate Internal Forces (Shear & Moment) along the beam
        # We need to cut sections along each span
        
        plot_x = []
        plot_shear = []
        plot_moment = []
        
        current_x_global = 0
        
        for i, length in enumerate(self.spans):
            # Get displacements for this element
            indices = [2*i, 2*i+1, 2*i+2, 2*i+3]
            u_element = D_global[indices]
            
            # Element stiffness
            K_element = self._element_stiffness_matrix(length)
            
            # End forces due to displacement: f = k * u
            f_disp = np.dot(K_element, u_element)
            
            # End forces due to loads (FEM)
            f_fem = self._calculate_fem(i)
            
            # Total end forces
            f_total = f_disp + f_fem
            
            # Extract Start Forces of the span
            V_start = f_total[0]
            M_start = f_total[1] # Matrix sign convention (CCW+)
            
            # Beam Sign Convention:
            # Shear: Up is positive
            # Moment: Sagging is positive (Smile) -> M_beam = -M_matrix_left
            
            # Create sampling points
            x_samples = np.linspace(0, length, 100)
            
            # Get loads on this span
            span_loads = [l for l in self.loads if l['span_idx'] == i]
            
            for x in x_samples:
                # Equilibrium at distance x from left node
                Vx = V_start
                Mx = -M_start + V_start * x
                
                # Subtract effect of loads within the cut [0, x]
                for load in span_loads:
                    if load['type'] == 'U':
                        # Uniform load applies from 0 to x (if x > 0)
                        if x > 0:
                            w = load['w']
                            Vx -= w * x
                            Mx -= (w * x**2) / 2
                    elif load['type'] == 'P':
                        # Point load applies if we passed it
                        if x > load['x']:
                            P = load['P']
                            dist = x - load['x']
                            Vx -= P
                            Mx -= P * dist
                
                plot_x.append(current_x_global + x)
                plot_shear.append(Vx)
                plot_moment.append(Mx)
                
            current_x_global += length
            
        df_forces = pd.DataFrame({
            'x': plot_x,
            'shear': plot_shear,
            'moment': plot_moment
        })
        
        return df_forces, df_reactions

# ==============================================================================
# 3. VISUALIZATION ENGINE (PLOTLY)
# ==============================================================================
class ResultVisualizer:
    def __init__(self, df_forces, spans, supports, loads):
        self.df = df_forces
        self.spans = spans
        self.supports = supports
        self.loads = loads
        self.cum_dist = [0] + list(np.cumsum(spans))
        self.total_length = self.cum_dist[-1]
        
    def create_diagrams(self):
        """Generates the main Plotly Figure containing Loading, SFD, and BMD."""
        
        # 1. Determine Scaling Factors (To prevent distortion)
        # Find maximum load value to normalize height
        load_values = [l['w'] for l in self.loads if l['type']=='U'] + [l['P'] for l in self.loads if l['type']=='P']
        max_load_val = max(map(abs, load_values)) if load_values else 100.0
        
        viz_height = max_load_val * 1.5
        
        # Support Dimensions
        sup_width = max(0.4, self.total_length * 0.02) # Min 0.4m or 2% of length
        sup_height = viz_height * 0.15 # 15% of visual height
        
        # Create Subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.3, 0.35, 0.35],
            subplot_titles=(
                "<b>1. Free Body Diagram (Loads & Structure)</b>",
                "<b>2. Shear Force Diagram (SFD)</b>",
                "<b>3. Bending Moment Diagram (BMD)</b>"
            )
        )
        
        # --- ROW 1: LOADING DIAGRAM ---
        
        # ** FIX: Draw Main Beam Line (Black & Thick) **
        fig.add_shape(
            type="line",
            x0=0, y0=0, x1=self.total_length, y1=0,
            line=dict(color="black", width=5),
            layer="above", # Ensure it sits on top of supports if needed
            row=1, col=1
        )
        
        # Draw Supports (Engineering Symbols)
        for i, x_loc in enumerate(self.cum_dist):
            if i < len(self.supports):
                sup_type = self.supports.iloc[i]['type']
                self._draw_single_support(fig, x_loc, 0, sup_type, sup_width, sup_height)

        # Draw Loads
        for load in self.loads:
            if load['type'] == 'U':
                self._draw_udl(fig, load, max_load_val, viz_height)
            elif load['type'] == 'P':
                self._draw_point_load(fig, load, max_load_val, viz_height)
                
        # Hide Y-axis for Loading Diagram
        fig.update_yaxes(visible=False, fixedrange=True, row=1, col=1)
        
        # --- ROW 2 & 3: SFD & BMD ---
        self._plot_graph_with_annotations(fig, self.df['x'], self.df['shear'], 
                                          "Shear Force", "#D32F2F", "kg", 2)
        self._plot_graph_with_annotations(fig, self.df['x'], self.df['moment'], 
                                          "Bending Moment", "#1976D2", "kg-m", 3)
        
        # Global Layout Settings
        # Calculate width dynamically to force horizontal scroll if beam is long
        pixel_per_span = 300
        chart_width = max(1000, len(self.spans) * pixel_per_span)
        
        fig.update_layout(
            width=chart_width, # Important for scroll
            height=950,
            template="plotly_white",
            showlegend=False,
            hovermode="x unified",
            margin=dict(l=50, r=50, t=50, b=50),
            font=dict(family="Sarabun")
        )
        
        # Add grid lines at supports
        for x in self.cum_dist:
            fig.add_vline(x=x, line_dash="dot", line_color="gray", opacity=0.5)
            
        return fig

    def _draw_single_support(self, fig, x, y, stype, w, h):
        """Helper to draw specific SVG paths for supports."""
        line_col = "#37474F"
        fill_col = "#CFD8DC"
        
        if stype == "Pin":
            # Triangle
            path = f"M {x},{y} L {x-w/2},{y-h} L {x+w/2},{y-h} Z"
            fig.add_shape(type="path", path=path, fillcolor=fill_col, line=dict(color=line_col, width=2), row=1, col=1)
            # Base
            fig.add_shape(type="line", x0=x-w, y0=y-h, x1=x+w, y1=y-h, line=dict(color=line_col, width=2), row=1, col=1)
            
        elif stype == "Roller":
            # Circle
            fig.add_shape(type="circle", x0=x-w/2, y0=y-h, x1=x+w/2, y1=y, fillcolor=fill_col, line=dict(color=line_col, width=2), row=1, col=1)
            # Base line
            fig.add_shape(type="line", x0=x-w, y0=y-h, x1=x+w, y1=y-h, line=dict(color=line_col, width=2), row=1, col=1)
            
        elif stype == "Fixed":
            # Vertical Line
            fig.add_shape(type="line", x0=x, y0=y-h, x1=x, y1=y+h, line=dict(color=line_col, width=4), row=1, col=1)
            # Hatching
            for k in range(5):
                dy = (2*h)/5 * k - h
                fig.add_shape(type="line", x0=x, y0=y+dy, x1=x-w/2, y1=y+dy+w/3, line=dict(color=line_col, width=1), row=1, col=1)

    def _draw_udl(self, fig, load, max_val, viz_h):
        start_x = self.cum_dist[load['span_idx']]
        end_x = self.cum_dist[load['span_idx']+1]
        
        # Scale height relative to load magnitude
        h = (load['w'] / max_val) * (viz_h * 0.5)
        
        # Filled Rectangle
        fig.add_trace(go.Scatter(
            x=[start_x, end_x, end_x, start_x],
            y=[0, 0, h, h],
            fill="toself",
            fillcolor="rgba(255, 111, 0, 0.2)",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False
        ), row=1, col=1)
        
        # Top Line
        fig.add_trace(go.Scatter(
            x=[start_x, end_x], y=[h, h],
            mode="lines",
            line=dict(color="#E65100", width=2)
        ), row=1, col=1)
        
        # Label
        fig.add_annotation(
            x=(start_x + end_x)/2, y=h,
            text=f"<b>w = {load['w']:,.0f}</b>",
            yshift=15, showarrow=False,
            font=dict(color="#E65100"),
            row=1, col=1
        )

    def _draw_point_load(self, fig, load, max_val, viz_h):
        pos_x = self.cum_dist[load['span_idx']] + load['x']
        h = (load['P'] / max_val) * (viz_h * 0.7)
        
        # Arrow
        fig.add_annotation(
            x=pos_x, y=0,
            ax=pos_x, ay=h,
            xref="x1", yref="y1", axref="x1", ayref="y1",
            showarrow=True,
            arrowhead=2, arrowsize=1.2, arrowwidth=2.5,
            arrowcolor="#B71C1C",
            text=f"<b>P = {load['P']:,.0f}</b>",
            yshift=10,
            font=dict(color="#B71C1C"),
            row=1, col=1
        )

    def _plot_graph_with_annotations(self, fig, x_data, y_data, name, color, unit, row_idx):
        # Main Line Plot
        fig.add_trace(go.Scatter(
            x=x_data, y=y_data,
            fill='tozeroy',
            line=dict(color=color, width=2.5),
            name=name
        ), row=row_idx, col=1)
        
        # Find Min/Max
        vals = y_data.values
        v_max = np.max(vals)
        v_min = np.min(vals)
        
        # Set Y-Range with padding
        v_range = v_max - v_min
        pad = v_range * 0.25 if v_range > 0 else 1.0
        fig.update_yaxes(range=[v_min - pad, v_max + pad], row=row_idx, col=1)
        
        # Add Labels for Max and Min
        # Use simple logic: label the global max and global min points
        for val, label_pos in [(v_max, 'top center'), (v_min, 'bottom center')]:
            if abs(val) > 0.01: # Avoid labeling strict zeros unless relevant
                # Find index (first occurrence)
                idx = np.where(vals == val)[0][0]
                x_pos = x_data.iloc[idx]
                
                # Determine vertical shift
                shift = 15 if val >= 0 else -15
                
                fig.add_annotation(
                    x=x_pos, y=val,
                    text=f"<b>{val:,.2f} {unit}</b>",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor=color,
                    borderwidth=1,
                    showarrow=False,
                    yshift=shift,
                    font=dict(color=color, size=11),
                    row=row_idx, col=1
                )


# ==============================================================================
# 4. DESIGN MODULE: RC DESIGN (SDM / WSD)
# ==============================================================================
class ReinforcedConcreteDesigner:
    """
    Handles RC Design calculations based on Analysis results.
    Supports both Strength Design Method (SDM) and Working Stress Design (WSD).
    """
    def __init__(self, Mu, Vu, fc, fy, fys, b, h, cover, method="SDM"):
        self.Mu = abs(Mu)
        self.Vu = abs(Vu)
        self.fc = fc
        self.fy = fy
        self.fys = fys
        self.b = b
        self.h = h
        self.cover = cover
        self.d = h - cover
        self.method = method
        
    def perform_flexural_design(self):
        """Calculates Required Steel Area (As)."""
        # Convert Moment to kg-cm
        Mu_kgcm = self.Mu * 100.0
        
        results = {}
        
        if self.method == "SDM":
            # 1. Strength Design Method
            phi = 0.90
            
            # Check Max Reinforcement Ratio (Ductility)
            # Beta1
            if self.fc <= 280:
                beta1 = 0.85
            else:
                beta1 = max(0.65, 0.85 - 0.05 * (self.fc - 280) / 70)
                
            # Rn = Mu / (phi * b * d^2)
            Rn = Mu_kgcm / (phi * self.b * (self.d**2))
            
            # Calculate Rho Required
            # rho = (0.85fc/fy) * [1 - sqrt(1 - 2Rn/0.85fc)]
            term_under_sqrt = 1 - (2 * Rn) / (0.85 * self.fc)
            
            if term_under_sqrt < 0:
                return {
                    "status": "Fail",
                    "message": "❌ Cross Section Too Small (Concrete Crushing)",
                    "As_required": 0.0
                }
                
            rho_req = (0.85 * self.fc / self.fy) * (1 - np.sqrt(term_under_sqrt))
            
            # Check Minimum Rho
            # As_min = max(14/fy, 0.25*sqrt(fc)/fy) * b * d (ACI Metric)
            rho_min = max(14/self.fy, (0.25 * np.sqrt(self.fc)) / self.fy)
            As_min = rho_min * self.b * self.d
            
            As_calc = rho_req * self.b * self.d
            As_final = max(As_calc, As_min)
            
            # Check Maximum Rho (Balanced)
            rho_bal = 0.85 * beta1 * (self.fc / self.fy) * (6120 / (6120 + self.fy))
            rho_max = 0.75 * rho_bal
            
            status_msg = "✅ Design OK"
            if rho_req > rho_max:
                status_msg = "⚠️ Warning: Over Reinforced (Brittle Failure Risk)"
            elif As_calc < As_min:
                status_msg = "ℹ️ Note: Minimum Steel Governs"
                
            results = {
                "status": "Pass" if rho_req <= rho_max else "Warning",
                "message": status_msg,
                "As_required": As_final,
                "Rho_actual": As_final / (self.b * self.d),
                "As_min": As_min
            }
            
        else:
            # 2. Working Stress Design
            # Modular ratio n
            Ec = 15100 * np.sqrt(self.fc)
            Es = 2.04e6
            n = round(Es / Ec)
            
            # Allowable Stresses
            fc_allow = 0.45 * self.fc
            fs_allow = 0.50 * self.fy
            
            # Design Constants (k, j, R)
            r = fs_allow / fc_allow
            k = n / (n + r)
            j = 1 - (k / 3)
            R = 0.5 * fc_allow * k * j
            
            # Check Moment Capacity of Concrete
            M_conc_cap = R * self.b * (self.d**2)
            
            if Mu_kgcm > M_conc_cap:
                return {
                    "status": "Fail",
                    "message": "❌ Concrete Compressive Stress Exceeded (Increase Depth)",
                    "As_required": 0.0
                }
            
            # Calculate Steel
            As_req = Mu_kgcm / (fs_allow * j * self.d)
            
            results = {
                "status": "Pass",
                "message": "✅ WSD Design OK",
                "As_required": As_req,
                "Rho_actual": As_req / (self.b * self.d)
            }
            
        return results

    def perform_shear_design(self):
        """Calculates Stirrup Requirements."""
        results = {}
        
        if self.method == "SDM":
            phi_v = 0.75
            
            # Vc = 0.53 * sqrt(fc) * b * d
            Vc = 0.53 * np.sqrt(self.fc) * self.b * self.d
            phi_Vc = phi_v * Vc
            
            check_msg = ""
            if self.Vu <= phi_Vc / 2:
                check_msg = "No Stirrups Required (Theoretical)"
            elif self.Vu <= phi_Vc:
                check_msg = "Minimum Stirrups Required"
            else:
                check_msg = "Stirrups Required for Strength"
                
            # Max Shear Check (Section limit)
            # Vs limit = 2.1 * sqrt(fc) * b * d
            Vs_max_limit = 2.1 * np.sqrt(self.fc) * self.b * self.d
            phi_Vn_max = phi_Vc + (phi_v * Vs_max_limit)
            
            if self.Vu > phi_Vn_max:
                check_msg = "❌ Section Too Small for Shear! (Increase Dimensions)"
                
            results = {
                "Vu": self.Vu,
                "phiVc": phi_Vc,
                "result": check_msg
            }
            
        else:
            # WSD Shear
            vc_allow = 0.29 * np.sqrt(self.fc)
            Vc = vc_allow * self.b * self.d
            
            msg = "Check Shear Stress"
            if self.Vu > Vc:
                msg = "Stirrups Required"
            else:
                msg = "Concrete Carries Shear OK"
                
            results = {
                "Vu": self.Vu,
                "phiVc": Vc, # Use this key for compatibility
                "result": msg
            }
            
        return results


# ==============================================================================
# 5. USER INTERFACE LAYER (STREAMLIT)
# ==============================================================================
def main():
    # --- HEADER ---
    st.markdown("""
        <div class="header-container">
            <h1>🏗️ RC Beam Pro: Enterprise Edition</h1>
            <p>High-Precision Structural Analysis & Reinforced Concrete Design Suite</p>
        </div>
    """, unsafe_allow_html=True)

    # --- SIDEBAR CONFIGURATION ---
    with st.sidebar:
        st.header("⚙️ Project Settings")
        
        # Design Method Selection
        design_method = st.radio("Design Code", ["SDM (Strength Design)", "WSD (Working Stress)"])
        method_key = "SDM" if "SDM" in design_method else "WSD"
        
        st.markdown("---")
        
        with st.expander("🧱 Material Properties", expanded=True):
            fc_input = st.number_input("Concrete fc' (ksc)", value=240.0, step=10.0, format="%.1f")
            fy_input = st.number_input("Main Steel fy (ksc)", value=4000.0, step=100.0, format="%.0f")
            fys_input = st.number_input("Stirrup Steel fys (ksc)", value=2400.0, step=100.0, format="%.0f")
            
        with st.expander("📐 Section Dimensions", expanded=True):
            b_input = st.number_input("Width b (cm)", value=25.0, step=5.0)
            h_input = st.number_input("Depth h (cm)", value=50.0, step=5.0)
            cover_input = st.number_input("Covering (cm)", value=3.0, step=0.5)
            
        st.markdown("---")
        st.write("🛡️ **Load Factors**")
        if method_key == "SDM":
            col_fac1, col_fac2 = st.columns(2)
            factor_dl = col_fac1.number_input("DL Factor", value=1.4)
            factor_ll = col_fac2.number_input("LL Factor", value=1.7)
        else:
            factor_dl = 1.0
            factor_ll = 1.0
            st.info("Using Unfactored Loads (1.0)")

    # --- MAIN INPUT AREA ---
    
    # 1. Geometry Section
    st.markdown('<div class="section-header">1. Geometry & Supports</div>', unsafe_allow_html=True)
    
    col_geo1, col_geo2 = st.columns([1, 2])
    
    with col_geo1:
        num_spans = st.number_input("Number of Spans", min_value=1, max_value=20, value=2)
    
    # Dynamic Input Grids
    spans_data = []
    supports_data = []
    
    with st.container():
        st.subheader("Span Lengths (m)")
        cols_spans = st.columns(4)
        for i in range(num_spans):
            with cols_spans[i % 4]:
                val = st.number_input(f"Span {i+1}", min_value=0.1, value=6.0, step=0.5, key=f"span_{i}")
                spans_data.append(val)
                
        st.subheader("Support Conditions")
        cols_supports = st.columns(4)
        support_options = ["Pin", "Roller", "Fixed", "None"]
        
        # Logic for default supports: Pin-Roller-Roller...
        for i in range(num_spans + 1):
            with cols_supports[i % 4]:
                default_idx = 0 if i == 0 else (1 if i < num_spans else 1)
                stype = st.selectbox(f"Node {i+1}", support_options, index=default_idx, key=f"sup_{i}")
                supports_data.append(stype)

    df_supports = pd.DataFrame({'type': supports_data})

    # Validation
    valid_supports = [s for s in supports_data if s != 'None']
    is_stable = True
    if len(valid_supports) < 2 and "Fixed" not in valid_supports:
        is_stable = False
        st.error("⚠️ Structure is Unstable: Requires at least 2 supports or 1 Fixed support.")

    # 2. Loading Section
    st.markdown('<div class="section-header">2. Applied Loads</div>', unsafe_allow_html=True)
    
    load_list = []
    
    # Use Tabs for cleaner interface per span
    span_tabs = st.tabs([f"Span {i+1}" for i in range(num_spans)])
    
    for i, tab in enumerate(span_tabs):
        with tab:
            c_load1, c_load2 = st.columns(2)
            
            # UDL
            with c_load1:
                st.markdown("##### 🟦 Uniform Load (kg/m)")
                udl_dl = st.number_input(f"Dead Load (DL)", value=0.0, key=f"udl_dl_{i}")
                udl_ll = st.number_input(f"Live Load (LL)", value=0.0, key=f"udl_ll_{i}")
                
                # Combine Load
                w_u = (udl_dl * factor_dl) + (udl_ll * factor_ll)
                if w_u != 0:
                    load_list.append({
                        'span_idx': i,
                        'type': 'U',
                        'w': w_u
                    })
                    st.caption(f"Factored w = {w_u:,.2f} kg/m")
            
            # Point Load
            with c_load2:
                st.markdown("##### 🔻 Point Loads (kg)")
                num_points = st.number_input("Add Point Loads", min_value=0, max_value=5, value=0, key=f"np_{i}")
                
                if num_points > 0:
                    for j in range(num_points):
                        cc1, cc2, cc3 = st.columns([1, 1, 1.5])
                        p_dl = cc1.number_input(f"P{j+1} DL", key=f"pdl_{i}_{j}")
                        p_ll = cc2.number_input(f"P{j+1} LL", key=f"pll_{i}_{j}")
                        p_x = cc3.number_input(f"Dist x (m)", min_value=0.0, max_value=spans_data[i], key=f"px_{i}_{j}")
                        
                        P_u = (p_dl * factor_dl) + (p_ll * factor_ll)
                        if P_u != 0:
                            load_list.append({
                                'span_idx': i,
                                'type': 'P',
                                'P': P_u,
                                'x': p_x
                            })
                            
    # --- ACTION BUTTON ---
    st.markdown("---")
    col_center = st.columns([1, 2, 1])
    with col_center[1]:
        analyze_btn = st.button("🚀 RUN FULL ANALYSIS", type="primary", use_container_width=True, disabled=not is_stable)

    # ==========================================================================
    # 6. PROCESSING & RESULTS DISPLAY
    # ==========================================================================
    if analyze_btn and is_stable:
        try:
            with st.spinner("Calculating Stiffness Matrices & Displacements..."):
                # A. Run Analysis
                engine = BeamAnalyzer(spans_data, df_supports, load_list)
                df_results, df_reactions = engine.analyze()
                
                if df_results is None:
                    st.error("Error: Structure Unstable (Singular Matrix). Please check supports.")
                    st.stop()
                
                # B. Visualization
                plotter = ResultVisualizer(df_results, spans_data, df_supports, load_list)
                fig_diagrams = plotter.create_diagrams()
                
                # C. Display Graphs
                st.markdown('<div class="section-header">3. Force Diagrams (SFD & BMD)</div>', unsafe_allow_html=True)
                
                # Insert Plotly Chart into Scrollable Container (HTML Wrapper)
                st.markdown(f"""
                <div class="scrollable-plot-container">
                    {fig_diagrams.to_html(full_html=False, include_plotlyjs='cdn')}
                </div>
                """, unsafe_allow_html=True)
                st.caption("💡 Tip: If the beam is long, scroll horizontally inside the box to see the full length.")

                # D. Reaction Table (NEW REQUEST)
                st.markdown('<div class="section-header">4. Support Reactions</div>', unsafe_allow_html=True)
                
                # Format the dataframe for display
                st.dataframe(
                    df_reactions.style.format({
                        "Vertical Reaction (Ry)": "{:,.2f} kg",
                        "Moment Reaction (Mz)": "{:,.2f} kg-m"
                    }),
                    use_container_width=True,
                    hide_index=True
                )

                # E. Max Forces Summary
                max_shear = df_results['shear'].abs().max()
                max_moment = df_results['moment'].abs().max()
                
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-label">Maximum Design Shear (Vu)</div>
                        <div class="result-value">{max_shear:,.2f} kg</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_res2:
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-label">Maximum Design Moment (Mu)</div>
                        <div class="result-value">{max_moment:,.2f} kg-m</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # F. Design Verification
                st.markdown('<div class="section-header">5. RC Design Check</div>', unsafe_allow_html=True)
                
                designer = ReinforcedConcreteDesigner(
                    max_moment, max_shear,
                    fc_input, fy_input, fys_input,
                    b_input, h_input, cover_input,
                    method=method_key
                )
                
                flex_res = designer.perform_flexural_design()
                shear_res = designer.perform_shear_design()
                
                col_des1, col_des2 = st.columns(2)
                
                # Flexure Card
                with col_des1:
                    st.subheader("🧱 Flexural Design (Main Steel)")
                    if flex_res['status'] == 'Fail':
                        st.error(flex_res['message'])
                    else:
                        if "Warning" in flex_res['message']:
                            st.warning(flex_res['message'])
                        else:
                            st.success(flex_res['message'])
                            
                        st.write(f"**Required As:** {flex_res['As_required']:.2f} cm²")
                        
                        # Bar Selection Helper
                        st.markdown("---")
                        st.caption("Suggested Reinforcement:")
                        bar_opts = [12, 16, 20, 25]
                        cols_bar = st.columns(len(bar_opts))
                        for idx, dia in enumerate(bar_opts):
                            area = 3.1416 * (dia/20)**2
                            num_bars = math.ceil(flex_res['As_required'] / area)
                            cols_bar[idx].metric(f"DB{dia}mm", f"{num_bars} bars")

                # Shear Card
                with col_des2:
                    st.subheader("⛓️ Shear Design (Stirrups)")
                    st.info(f"Result: **{shear_res['result']}**")
                    st.write(f"Design Vu: {shear_res['Vu']:,.2f} kg")
                    st.write(f"Capacity (phi Vc): {shear_res['phiVc']:,.2f} kg")

        except Exception as e:
            st.error(f"❌ An error occurred during calculation: {str(e)}")
            st.warning("Please verify your inputs. Ensure span lengths are not zero and structure is stable.")

if __name__ == "__main__":
    main()
