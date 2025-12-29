import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# ==============================================================================
# 1. CONFIGURATION & CSS (ENGINEERING THEME)
# ==============================================================================
st.set_page_config(page_title="RC Beam Pro: Ultimate", layout="wide", page_icon="🏗️")

st.markdown("""
<style>
    /* Global Font */
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Sarabun', sans-serif; }
    
    /* Header Styling */
    .header-box {
        background: linear-gradient(135deg, #004d40 0%, #00695c 100%);
        color: white; padding: 25px; border-radius: 12px;
        text-align: center; margin-bottom: 30px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    /* Section Headers */
    .section-header {
        border-left: 5px solid #00695c;
        background-color: #e0f2f1;
        color: #004d40;
        padding: 12px 20px;
        font-size: 1.25rem;
        font-weight: 700;
        border-radius: 0 8px 8px 0;
        margin-top: 25px; margin-bottom: 15px;
    }
    
    /* Input Cards */
    .input-card {
        background: #ffffff; border: 1px solid #cfd8dc;
        padding: 20px; border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Result Cards */
    .metric-card {
        background: #f5f5f5; border-left: 4px solid #00695c;
        padding: 15px; margin-bottom: 10px; border-radius: 4px;
    }

    /* Adjust Streamlit Default Padding */
    .block-container { padding-top: 2rem; padding-bottom: 5rem; }
    
    /* Input Field Bold */
    .stNumberInput input { font-weight: 600; color: #004d40; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. STRUCTURAL ANALYSIS ENGINE (MATRIX STIFFNESS METHOD)
# ==============================================================================
class MatrixBeamSolver:
    def __init__(self, spans, supports, loads, E=2e6, I=1e-3):
        self.spans = spans
        self.supports = supports
        self.loads = loads
        self.E = E
        self.I = I
        self.nodes = [0] + list(np.cumsum(spans))
        self.n_nodes = len(self.nodes)
        self.dof = 2 * self.n_nodes # 2 DOF per node (y, theta)

    def solve(self):
        # 1. Global Stiffness Matrix (K)
        K_global = np.zeros((self.dof, self.dof))
        
        for i, L in enumerate(self.spans):
            k = self.E * self.I / L**3
            # Element Matrix
            K_el = np.array([
                [12*k,      6*k*L,    -12*k,     6*k*L],
                [6*k*L,     4*k*L**2, -6*k*L,    2*k*L**2],
                [-12*k,    -6*k*L,     12*k,    -6*k*L],
                [6*k*L,     2*k*L**2, -6*k*L,    4*k*L**2]
            ])
            # Assemble
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            for r in range(4):
                for c in range(4):
                    K_global[idx[r], idx[c]] += K_el[r, c]

        # 2. Equivalent Nodal Forces (FEM)
        F_global = np.zeros(self.dof)
        
        for l in self.loads:
            i = l['span_idx']
            L = self.spans[i]
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            
            fem = np.zeros(4)
            if l['type'] == 'U':
                w = l['w']
                fem = np.array([w*L/2, w*L**2/12, w*L/2, -w*L**2/12])
            elif l['type'] == 'P':
                P = l['P']; a = l['x']; b = L - a
                fem = np.array([
                    (P*b**2*(3*a+b))/L**3,
                    (P*a*b**2)/L**2,
                    (P*a**2*(a+3*b))/L**3,
                    -(P*a**2*b)/L**2
                ])
            
            # Subtract FEM from Global Force Vector (F_node = F_ext - FEM)
            F_global[idx] -= fem

        # 3. Boundary Conditions
        constrained_dof = []
        for i, row in self.supports.iterrows():
            stype = row['type']
            if stype in ["Pin", "Roller"]:
                constrained_dof.append(2*i) # Fix Y
            elif stype == "Fixed":
                constrained_dof.append(2*i)   # Fix Y
                constrained_dof.append(2*i+1) # Fix Theta

        # 4. Solve Displacements
        free_dof = [x for x in range(self.dof) if x not in constrained_dof]
        
        if not free_dof:
            return pd.DataFrame(), pd.DataFrame() # Fully constrained error handle

        K_red = K_global[np.ix_(free_dof, free_dof)]
        F_red = F_global[free_dof]
        
        try:
            D_free = np.linalg.solve(K_red, F_red)
        except np.linalg.LinAlgError:
            raise ValueError("Structure is unstable!")

        D_total = np.zeros(self.dof)
        D_total[free_dof] = D_free

        # 5. Compute Internal Forces & Generate Plot Data
        all_x, all_v, all_m = [], [], []
        curr_x_offset = 0
        
        for i, L in enumerate(self.spans):
            # Element Displacements
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            u = D_total[idx]
            
            # Element Stiffness
            k = self.E * self.I / L**3
            K_el = np.array([
                [12*k, 6*k*L, -12*k, 6*k*L], [6*k*L, 4*k*L**2, -6*k*L, 2*k*L**2],
                [-12*k, -6*k*L, 12*k, -6*k*L], [6*k*L, 2*k*L**2, -6*k*L, 4*k*L**2]
            ])
            
            # Forces from Stiffness (F = K*u)
            f_stiff = np.dot(K_el, u)
            
            # Add FEM back to get total member end forces
            fem_total = np.zeros(4)
            span_loads = [l for l in self.loads if l['span_idx'] == i]
            
            for l in span_loads:
                if l['type'] == 'U':
                    w = l['w']
                    fem_total += np.array([w*L/2, w*L**2/12, w*L/2, -w*L**2/12])
                elif l['type'] == 'P':
                    P = l['P']; a = l['x']; b = L - a
                    fem_total += np.array([
                        (P*b**2*(3*a+b))/L**3, (P*a*b**2)/L**2,
                        (P*a**2*(a+3*b))/L**3, -(P*a**2*b)/L**2
                    ])
            
            f_final = f_stiff + fem_total
            
            # Integration for internal diagrams
            V_start = f_final[0]
            M_start = f_final[1] # CCW +
            
            # Discretize
            x_pts = np.linspace(0, L, 100)
            for x in x_pts:
                # Statics at distance x
                Vx = V_start
                Mx = -M_start + V_start*x
                
                for l in span_loads:
                    if l['type'] == 'U':
                        if x > 0:
                            w = l['w']
                            Vx -= w*x
                            Mx -= w*x**2/2
                    elif l['type'] == 'P':
                        if x > l['x']:
                            Vx -= l['P']
                            Mx -= l['P']*(x - l['x'])
                
                all_x.append(curr_x_offset + x)
                all_v.append(Vx)
                all_m.append(Mx)
            
            curr_x_offset += L

        res_df = pd.DataFrame({'x': all_x, 'shear': all_v, 'moment': all_m})
        
        # Calculate Reactions (approx from first/last points of spans or K*D)
        # Using K_global * D_total - F_external_nodes (Simplified for display)
        # R = K*D - F_applied_at_nodes
        R_vec = np.dot(K_global, D_total) 
        # Note: In this simple FEM, node loads are 0, loads are on members. 
        # So Reactions = Global Nodal Forces required.
        
        return res_df, R_vec

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================
def check_stability(df_sup):
    active = df_sup[df_sup['type'] != 'None']
    if len(active) == 0: return False, "❌ Structure is floating (No supports)"
    if len(active) == 1 and active.iloc[0]['type'] != 'Fixed': return False, "❌ Unstable (Need at least Fixed or 2 Supports)"
    return True, "✅ Stable"

def draw_engineering_support(fig, x, y, sup_type, size_scale):
    """
    วาดสัญลักษณ์ Support ให้ถูกต้องตามแบบวิศวกรรม (SVG Paths)
    """
    # Scale Constants
    h = size_scale * 1.2  # Height
    w = size_scale * 0.8  # Width base
    color_line = "#263238"
    color_fill = "#CFD8DC"
    
    if sup_type == "Pin":
        # Triangle
        path = f"M {x},{y} L {x-w},{y-h} L {x+w},{y-h} Z"
        fig.add_shape(type="path", path=path, fillcolor=color_fill, line=dict(color=color_line, width=2), row=1, col=1)
        # Base Line
        fig.add_shape(type="line", x0=x-w*1.5, y0=y-h, x1=x+w*1.5, y1=y-h, line=dict(color=color_line, width=3), row=1, col=1)
        # Hatching (Optional visual flair)
        for i in range(-2, 3):
             fig.add_shape(type="line", x0=x+i*w/2, y0=y-h, x1=x+i*w/2-w/4, y1=y-h-w/4, line=dict(color=color_line, width=1), row=1, col=1)

    elif sup_type == "Roller":
        # Circle
        r = w * 0.8
        fig.add_shape(type="circle", x0=x-r, y0=y-r, x1=x+r, y1=y+r, fillcolor=color_fill, line=dict(color=color_line, width=2), row=1, col=1)
        # Base Line (Lower)
        base_y = y - r
        fig.add_shape(type="line", x0=x-w*1.5, y0=base_y, x1=x+w*1.5, y1=base_y, line=dict(color=color_line, width=3), row=1, col=1)
        
    elif sup_type == "Fixed":
        # Vertical Line
        h_fix = h * 1.5
        fig.add_shape(type="line", x0=x, y0=y-h_fix/2, x1=x, y1=y+h_fix/2, line=dict(color=color_line, width=4), row=1, col=1)
        # Hatching
        for i in np.linspace(-h_fix/2, h_fix/2, 6):
             fig.add_shape(type="line", x0=x, y0=y+i, x1=x-w/1.5, y1=y+i+w/3, line=dict(color=color_line, width=1), row=1, col=1)

# ==============================================================================
# 4. PLOTTING ENGINE (CORRECTED & BEAUTIFIED)
# ==============================================================================
def create_diagrams(df, spans, supports, loads):
    cum_len = [0] + list(np.cumsum(spans))
    total_len = cum_len[-1]
    
    # 1. Determine Visualization Scale
    # หาค่า Load สูงสุดเพื่อนำมา Scale ความสูงของกราฟ (Normalize)
    load_vals = [l['w'] for l in loads if l['type']=='U'] + [l['P'] for l in loads if l['type']=='P']
    max_load = max(map(abs, load_vals)) if load_vals else 100
    
    # Define aesthetic scaling factors
    viz_h = max_load * 1.5 if max_load > 0 else 10
    sup_scale = total_len * 0.025 # ขนาด Support สัมพันธ์กับความยาวคาน
    sup_scale = max(sup_scale, 0.3) # ขั้นต่ำไม่ให้เล็กเกิน
    
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=("<b>1. Loading Diagram (FBD)</b>", "<b>2. Shear Force Diagram (SFD)</b>", "<b>3. Bending Moment Diagram (BMD)</b>"),
        row_heights=[0.3, 0.35, 0.35]
    )

    # --- Row 1: Loading Diagram ---
    # Beam Line
    fig.add_shape(type="line", x0=0, y0=0, x1=total_len, y1=0, line=dict(color="black", width=4), row=1, col=1)
    
    # Supports
    for i, x in enumerate(cum_len):
        if i < len(supports):
            draw_engineering_support(fig, x, 0, supports.iloc[i]['type'], sup_scale)

    # Loads
    for l in loads:
        if l['type'] == 'U':
            x1, x2 = cum_len[l['span_idx']], cum_len[l['span_idx']+1]
            h = (abs(l['w']) / max_load) * (viz_h * 0.5) # Height relative to max
            
            # UDL Block
            fig.add_trace(go.Scatter(
                x=[x1, x2, x2, x1], y=[0, 0, h, h], 
                fill='toself', fillcolor='rgba(255, 111, 0, 0.2)', 
                mode='none', showlegend=False, hoverinfo='skip'
            ), row=1, col=1)
            
            # Top Line
            fig.add_shape(type="line", x0=x1, y0=h, x1=x2, y1=h, line=dict(color="#FF6F00", width=2), row=1, col=1)
            
            # Arrows for UDL (Multiple small arrows)
            n_arrows = max(3, int((x2-x1)/total_len * 10))
            for xa in np.linspace(x1, x2, n_arrows+2)[1:-1]:
                fig.add_annotation(
                    x=xa, y=0, ax=xa, ay=h, xref="x1", yref="y1", axref="x1", ayref="y1",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor="#FF6F00",
                    row=1, col=1
                )
            # Label
            fig.add_annotation(x=(x1+x2)/2, y=h, text=f"<b>w = {l['w']:,.0f}</b>", showarrow=False, yshift=15, font=dict(color="#E65100"), row=1, col=1)

        elif l['type'] == 'P':
            px = cum_len[l['span_idx']] + l['x']
            h = (abs(l['P']) / max_load) * (viz_h * 0.6)
            
            # Point Load Arrow
            fig.add_annotation(
                x=px, y=0, ax=px, ay=h, xref="x1", yref="y1", axref="x1", ayref="y1",
                showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2.5, arrowcolor="#BF360C",
                text=f"<b>P = {l['P']:,.0f}</b>", row=1, col=1
            )

    fig.update_yaxes(visible=False, range=[-sup_scale*2, viz_h*1.2], row=1, col=1, fixedrange=True)

    # --- Row 2: SFD ---
    color_v = '#1976D2'
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color=color_v, width=2), fillcolor='rgba(25, 118, 210, 0.1)', name="Shear"), row=2, col=1)
    
    # --- Row 3: BMD ---
    color_m = '#D32F2F'
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color=color_m, width=2), fillcolor='rgba(211, 47, 47, 0.1)', name="Moment"), row=3, col=1)

    # --- Labeling Max/Min ---
    def add_labels(col_name, row, color):
        arr = df[col_name].to_numpy()
        if len(arr) == 0: return
        
        # Find local peaks could be complex, let's just do Global Max/Min for cleanliness
        # Or better: Max/Min per span? Let's stick to Global for now to avoid clutter
        g_max, g_min = np.max(arr), np.min(arr)
        
        # Add labels
        for val in [g_max, g_min]:
            if abs(val) > 0.1:
                idx = np.where(arr == val)[0][0]
                x_pos = df['x'].iloc[idx]
                pos = "top" if val >= 0 else "bottom"
                ys = 10 if pos=="top" else -10
                
                fig.add_annotation(
                    x=x_pos, y=val, text=f"<b>{val:,.2f}</b>",
                    showarrow=False,
                    bgcolor="rgba(255,255,255,0.85)", bordercolor=color, borderwidth=1,
                    font=dict(color=color, size=11),
                    yshift=ys, row=row, col=1
                )
        
        # Smart Range Padding
        rn = g_max - g_min
        pad = rn * 0.2 if rn > 0 else 1.0
        fig.update_yaxes(range=[g_min-pad, g_max+pad], row=row, col=1)

    add_labels('shear', 2, color_v)
    add_labels('moment', 3, color_m)

    # Global Layout
    fig.update_layout(
        height=1000, 
        width=max(1000, len(spans)*150), # Responsive Width: ถ้า Span เยอะ กราฟจะกว้างขึ้นแล้วมี Scrollbar
        template="plotly_white", 
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode="x unified"
    )
    
    # Add Grid Lines at Supports
    for cx in cum_len:
        fig.add_vline(x=cx, line_dash="dot", line_color="gray", opacity=0.5)

    return fig

# ==============================================================================
# 5. UI MAIN APPLICATION
# ==============================================================================
def main():
    st.markdown('<div class="header-box"><h1>🏗️ RC Beam Pro: Engineering Suite</h1></div>', unsafe_allow_html=True)

    # --- SIDEBAR: SETTINGS ---
    with st.sidebar:
        st.header("⚙️ Design Parameters")
        unit_sys = st.radio("Unit System", ["Metric (kg, m)", "SI (kN, m)"])
        design_method = st.selectbox("Design Method", ["SDM (Strength Design)", "WSD (Working Stress)"])
        
        st.markdown("---")
        st.markdown("### 🧱 Material & Section")
        with st.expander("Properties", expanded=True):
            fc = st.number_input("f'c (ksc/MPa)", value=240.0)
            fy = st.number_input("fy (Main)", value=4000.0)
            fys = st.number_input("fy (Stirrup)", value=2400.0)
            b = st.number_input("Width b (cm)", value=25.0)
            h = st.number_input("Depth h (cm)", value=50.0)
            cv = st.number_input("Covering (cm)", value=3.0)
            
        st.markdown("### ⚖️ Load Factors")
        if "SDM" in design_method:
            c1, c2 = st.columns(2)
            fdl = c1.number_input("DL Factor", value=1.4)
            fll = c2.number_input("LL Factor", value=1.7)
        else:
            fdl, fll = 1.0, 1.0
            st.info("Factors = 1.0 for WSD")

        params = {'fc':fc, 'fy':fy, 'fys':fys, 'b':b, 'h':h, 'cv':cv, 
                  'fdl':fdl, 'fll':fll, 'unit':unit_sys, 'method':design_method}

    # --- MAIN CONTENT ---
    
    # 1. Geometry Input
    col_geo, col_load = st.columns([1, 1.2])
    
    with col_geo:
        st.markdown('<div class="section-header">1️⃣ Geometry</div>', unsafe_allow_html=True)
        with st.container(border=True):
            n_span = st.number_input("Number of Spans", min_value=1, max_value=20, value=2)
            
            # Use columns for compact input
            st.markdown("##### Span Lengths (m)")
            
            # Dynamic Grid for Spans
            spans = []
            # Calculate rows needed
            cols_per_row = 3
            for r in range(0, n_span, cols_per_row):
                cols = st.columns(cols_per_row)
                for c in range(cols_per_row):
                    idx = r + c
                    if idx < n_span:
                        with cols[c]:
                            val = st.number_input(f"L{idx+1}", min_value=0.5, value=5.0, step=0.5, key=f"span_{idx}")
                            spans.append(val)
                            
            st.markdown("##### Supports")
            sup_types = []
            sup_opts = ["Pin", "Roller", "Fixed", "None"]
            
            # Dynamic Grid for Supports
            n_sup = n_span + 1
            for r in range(0, n_sup, cols_per_row):
                cols = st.columns(cols_per_row)
                for c in range(cols_per_row):
                    idx = r + c
                    if idx < n_sup:
                        with cols[c]:
                            # Intelligent Default
                            def_idx = 0 if idx==0 else (1 if idx < n_sup-1 else 1)
                            val = st.selectbox(f"Sup {idx+1}", sup_opts, index=def_idx, key=f"sup_{idx}")
                            sup_types.append(val)
            
            # Stability Check
            df_sup = pd.DataFrame({'type': sup_types})
            is_stable, msg = check_stability(df_sup)
            if not is_stable:
                st.error(msg)

    # 2. Load Input
    with col_load:
        st.markdown('<div class="section-header">2️⃣ Loads</div>', unsafe_allow_html=True)
        st.info(f"Using Factors: {fdl} DL + {fll} LL")
        
        loads = []
        tabs = st.tabs([f"Span {i+1}" for i in range(n_span)])
        
        for i, tab in enumerate(tabs):
            with tab:
                st.markdown(f"**Loads on Span {i+1}**")
                c1, c2 = st.columns(2)
                
                # UDL
                with c1:
                    with st.container(border=True):
                        st.markdown("🔹 **Uniform Load**")
                        wdl = st.number_input("DL", 0.0, step=100.0, key=f"wdl_{i}")
                        wll = st.number_input("LL", 0.0, step=100.0, key=f"wll_{i}")
                        wu = wdl*fdl + wll*fll
                        if wu != 0:
                            loads.append({'span_idx': i, 'type': 'U', 'w': wu})
                            st.caption(f"Total Wu = {wu:,.2f}")
                
                # Point Load
                with c2:
                    with st.container(border=True):
                        st.markdown("🔻 **Point Loads**")
                        qty = st.number_input("Qty", 0, 5, 0, key=f"qty_{i}")
                        for j in range(qty):
                            cc1, cc2 = st.columns([1, 1.5])
                            pd = cc1.number_input(f"P_DL #{j+1}", key=f"pd_{i}_{j}")
                            pl = cc1.number_input(f"P_LL #{j+1}", key=f"pl_{i}_{j}")
                            px = cc2.number_input(f"x (m) #{j+1}", 0.0, spans[i], spans[i]/2, key=f"px_{i}_{j}")
                            
                            pu = pd*fdl + pl*fll
                            if pu != 0:
                                loads.append({'span_idx': i, 'type': 'P', 'P': pu, 'x': px})

    # 3. Action Button
    st.markdown("---")
    btn_col1, btn_col2 = st.columns([1, 2])
    with btn_col1:
        calc_btn = st.button("🚀 Analyze Structure", type="primary", use_container_width=True, disabled=not is_stable)

    if calc_btn and is_stable:
        try:
            with st.spinner("Calculating Stiffness Matrix & Generating Diagrams..."):
                # Call Solver
                solver = MatrixBeamSolver(spans, df_sup, loads)
                df_res, reactions = solver.solve()
                
                # --- RESULTS ---
                st.markdown('<div class="section-header">3️⃣ Analysis Results</div>', unsafe_allow_html=True)
                
                # Scrollable Container for Plot
                st.markdown("Scroll horizontally if the graph is too wide.")
                with st.container(border=True):
                    fig = create_diagrams(df_res, spans, df_sup, loads)
                    st.plotly_chart(fig, use_container_width=True) # Let Plotly handle internal scroll/resize
                
                # Numeric Results
                c1, c2 = st.columns(2)
                with c1:
                    st.info("💡 **Design Summary (Max Values)**")
                    max_v = df_res['shear'].abs().max()
                    max_m = df_res['moment'].abs().max()
                    st.metric("Max Shear (Vu)", f"{max_v:,.2f}")
                    st.metric("Max Moment (Mu)", f"{max_m:,.2f}")
                
                with c2:
                    st.success("✅ **RC Design Check (SDM)**")
                    # Simplified SDM Check
                    d = h - cv
                    # Convert to appropriate units (Assuming kg/cm for calc)
                    # If inputs are kg, m -> M is kg-m. Need kg-cm for formula
                    Mu_kgcm = max_m * 100 if 'kg' in unit_sys else max_m * 1000 * 10 # kN-m -> N-mm approx
                    
                    # Target M = phi * Mn
                    # Rn = Mu / (phi b d^2)
                    phi = 0.9
                    
                    if 'kg' in unit_sys:
                        b_c, d_c = b, d
                        fc_c, fy_c = fc, fy
                    else:
                        b_c, d_c = b*10, d*10 # mm
                        fc_c, fy_c = fc, fy # MPa
                    
                    Rn = Mu_kgcm / (phi * b_c * d_c**2)
                    rho_req = (0.85 * fc_c / fy_c) * (1 - np.sqrt(max(0, 1 - 2*Rn/(0.85*fc_c))))
                    
                    As_req = rho_req * b_c * d_c
                    st.write(f"**Required As(+):** {As_req:,.2f} cm² (Approx)")
                    
                    if np.isnan(rho_req) or rho_req > 0.025: # Rough limit
                        st.error("Section Size Insufficient! (Over Reinforced)")
                    else:
                        st.write("Section Size: **OK**")

        except Exception as e:
            st.error(f"Analysis Failed: {e}")
            st.error("Please check inputs (e.g., zero length spans, excessive loads).")

if __name__ == "__main__":
    main()
