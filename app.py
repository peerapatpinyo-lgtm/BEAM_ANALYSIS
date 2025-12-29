import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# ==============================================================================
# 1. CSS & PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="RC Beam Pro: Ultimate Edition", layout="wide", page_icon="🏗️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Sarabun', sans-serif; font-size: 15px; }
    
    h1, h2, h3 { color: #0D47A1; font-weight: 700; }
    
    .stApp { background-color: #FAFAFA; }
    
    .header-box {
        background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
        color: white; padding: 25px; border-radius: 10px;
        text-align: center; margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .section-header {
        border-left: 6px solid #1565C0; padding-left: 15px;
        font-size: 1.4rem; font-weight: 700; margin-top: 35px; margin-bottom: 20px;
        background-color: #E3F2FD; padding: 12px; border-radius: 0 8px 8px 0;
        color: #0D47A1; display: flex; align-items: center;
    }
    
    .card {
        background: white; padding: 20px; border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05); border: 1px solid #E0E0E0;
        margin-bottom: 15px;
    }
    
    .stNumberInput input { font-weight: bold; color: #1565C0; }
    .stSelectbox div[data-baseweb="select"] > div { font-weight: 600; }
    
    /* Custom Table Style */
    .dataframe { width: 100% !important; font-size: 14px; }
    .dataframe th { background-color: #1565C0 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. STRUCTURAL ANALYSIS ENGINE (MATRIX STIFFNESS METHOD)
# ==============================================================================
class BeamAnalysisEngine:
    def __init__(self, spans, supports, loads, E=2e6, I=1e-3):
        self.spans = spans
        self.supports = supports
        self.loads = loads
        self.E = E
        self.I = I
        self.nodes = [0] + list(np.cumsum(spans))
        self.n_nodes = len(self.nodes)
        self.dof = 2 * self.n_nodes # 2 DOF per node (Vertical, Rotation)
        
    def solve(self):
        # 1. Global Stiffness Matrix
        K_global = np.zeros((self.dof, self.dof))
        
        for i, L in enumerate(self.spans):
            k = self.E * self.I / L**3
            # Local K matrix for beam element
            K_elem = np.array([
                [12*k,      6*k*L,    -12*k,     6*k*L],
                [6*k*L,     4*k*L**2, -6*k*L,    2*k*L**2],
                [-12*k,    -6*k*L,     12*k,    -6*k*L],
                [6*k*L,     2*k*L**2, -6*k*L,    4*k*L**2]
            ])
            
            # Map to Global DOF
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            for r in range(4):
                for c in range(4):
                    K_global[idx[r], idx[c]] += K_elem[r, c]

        # 2. Force Vector (FEM)
        F_global = np.zeros(self.dof)
        
        # Fixed End Moments calculation
        for l in self.loads:
            span_idx = l['span_idx']
            L = self.spans[span_idx]
            
            idx_1 = 2*span_idx
            idx_2 = 2*span_idx + 1
            idx_3 = 2*span_idx + 2
            idx_4 = 2*span_idx + 3
            
            if l['type'] == 'U': # UDL
                w = l['w']
                # Reaction
                r1 = w*L/2; r2 = w*L/2
                m1 = w*L**2/12; m2 = -w*L**2/12
                F_global[idx_1] -= r1
                F_global[idx_2] -= m1
                F_global[idx_3] -= r2
                F_global[idx_4] -= m2
                
            elif l['type'] == 'P': # Point Load
                P = l['P']
                a = l['x']
                b = L - a
                # Reaction
                r1 = (P*b**2*(3*a+b))/L**3
                r2 = (P*a**2*(a+3*b))/L**3
                m1 = (P*a*b**2)/L**2
                m2 = -(P*a**2*b)/L**2
                F_global[idx_1] -= r1
                F_global[idx_2] -= m1
                F_global[idx_3] -= r2
                F_global[idx_4] -= m2

        # 3. Apply Boundary Conditions
        constrained_dof = []
        for i, row in self.supports.iterrows():
            stype = row['type']
            node_idx = i
            if stype == "Pin" or stype == "Roller":
                constrained_dof.append(2*node_idx) # Fix vertical Y
            elif stype == "Fixed":
                constrained_dof.append(2*node_idx) # Fix Y
                constrained_dof.append(2*node_idx+1) # Fix Rotation
        
        # 4. Solve for Displacements
        free_dof = [i for i in range(self.dof) if i not in constrained_dof]
        
        K_reduced = K_global[np.ix_(free_dof, free_dof)]
        F_reduced = F_global[free_dof]
        
        D_free = np.linalg.solve(K_reduced, F_reduced)
        
        D_total = np.zeros(self.dof)
        D_total[free_dof] = D_free
        
        # 5. Post-Processing (Internal Forces)
        results = []
        n_points = 50 # Resolution per span
        
        for i, L in enumerate(self.spans):
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            u_elem = D_total[idx] # [y1, theta1, y2, theta2]
            
            x_local = np.linspace(0, L, n_points)
            
            for x in x_local:
                # Shape functions derivatives for Shear (V) and Moment (M)
                # M(x) = E*I * d2v/dx2
                # V(x) = E*I * d3v/dx3 (but better to use equilibrium)
                
                # Using simple statics from node forces is often more stable for plotting, 
                # but here we use shape functions for displacement and superposition for loads
                
                # --- Quick Statics Approach for Diagram ---
                # We need exact forces at ends to do statics
                pass # (Detailed integration implemented below in simple statics loop instead)
                
        # --- Robust Statics Method (Shear Integration) ---
        # Recalculate Member End Forces from K*u + FEM
        final_x = []
        final_v = []
        final_m = []
        
        # Compute Reactions
        R_total = np.dot(K_global, D_total) + F_global # Net nodal forces (Reactions)
        # Note: F_global was -FEM, so we add it back? No, R = K*d - F_equiv
        # F_node_equiv = -FEM.
        # R = K*d - F_node_equiv
        
        # Let's use simple segment integration which is robust
        # 1. Get Reactions at Supports
        # Re-assemble F_equiv (positive loads)
        # Solve reactions from equilibrium is tricky with indeterminate. 
        # Best is K*d = F_ext + R => R = K*d - F_ext
        
        R_calc = np.dot(K_global, D_total) # Nodal forces required to maintain deformation
        # Subtract applied nodal loads (none in this simple model, only member loads)
        # The member loads contribute to equivalent nodal forces.
        # Correct: R = K*D - F_equivalent_load
        
        reactions = R_calc - F_global
        
        # Generate Plot Data by integrating shear
        curr_x = 0
        current_shear = 0
        current_moment = 0
        
        # We need to process span by span
        all_x, all_v, all_m = [], [], []
        
        # Re-calculate element end forces for starting conditions of each span?
        # No, easier: Use Segment Calculation
        
        # Element stiffness forces
        for i, L in enumerate(self.spans):
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            d_local = D_total[idx]
            
            k = self.E * self.I / L**3
            K_el = np.array([
                [12*k,      6*k*L,    -12*k,     6*k*L],
                [6*k*L,     4*k*L**2, -6*k*L,    2*k*L**2],
                [-12*k,    -6*k*L,     12*k,    -6*k*L],
                [6*k*L,     2*k*L**2, -6*k*L,    4*k*L**2]
            ])
            f_member = np.dot(K_el, d_local)
            
            # Add FEM
            fem_vec = np.zeros(4)
            # Find loads on this span
            span_loads = [l for l in self.loads if l['span_idx'] == i]
            
            for l in span_loads:
                if l['type'] == 'U':
                    w = l['w']
                    fem_vec += np.array([w*L/2, w*L**2/12, w*L/2, -w*L**2/12])
                elif l['type'] == 'P':
                    P = l['P']; a = l['x']; b = L - a
                    r1 = (P*b**2*(3*a+b))/L**3
                    m1 = (P*a*b**2)/L**2
                    r2 = (P*a**2*(a+3*b))/L**3
                    m2 = -(P*a**2*b)/L**2
                    fem_vec += np.array([r1, m1, r2, m2])
            
            f_final = f_member + fem_vec
            
            # f_final = [V_left, M_left, V_right, M_right]
            V_start = f_final[0]
            M_start = f_final[1] # Counter-clockwise positive
            
            # Generate points
            x_span = np.linspace(0, L, 50)
            
            for x in x_span:
                v_x = V_start
                m_x = -M_start + V_start * x # Beam sign convention: M positive compression top?
                # Standard structural analysis: M positive causes tension at bottom.
                # Left Moment CCW is negative in beam convention usually.
                # Let's stick to: M(x) = M_start_force + integral(V)
                
                # Apply loads within x
                for l in span_loads:
                    if l['type'] == 'U':
                        if x > 0:
                            w = l['w']
                            v_x -= w * x
                            m_x -= w * x**2 / 2
                    elif l['type'] == 'P':
                        if x > l['x']:
                            v_x -= l['P']
                            m_x -= l['P'] * (x - l['x'])
                
                all_x.append(curr_x + x)
                all_v.append(v_x)
                all_m.append(m_x)
                
            curr_x += L
            
        return pd.DataFrame({'x': all_x, 'shear': all_v, 'moment': all_m}), reactions

# ==============================================================================
# 3. HELPER FUNCTIONS & UI LOGIC
# ==============================================================================

def check_stability(supports_df):
    if supports_df.empty: return False, "No supports."
    types = supports_df[supports_df['type'] != 'None']['type'].tolist()
    if not types: return False, "No active supports."
    if len(types) < 2 and "Fixed" not in types: return False, "Unstable."
    return True, "Stable"

def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Project Settings")
        unit_sys = st.radio("Unit System", ["Metric (kg, m)", "SI (kN, m)"])
        code = st.selectbox("Design Code", ["ACI 318-19", "EIT 1007-34"])
        method = st.radio("Method", ["SDM (Strength)", "WSD (Working)"], index=0)
        
        st.markdown("### 🧱 Material Properties")
        with st.expander("Concrete & Steel", expanded=True):
            fc = st.number_input("fc' (ksc/MPa)", 240.0)
            fy = st.number_input("fy (Main)", 4000.0)
            fys = st.number_input("fy (Stirrup)", 2400.0)
        
        with st.expander("Section Size", expanded=True):
            b = st.number_input("Width b (cm)", 25.0)
            h = st.number_input("Depth h (cm)", 50.0)
            cover = st.number_input("Covering (cm)", 3.0)
            
        st.markdown("### ⚖️ Safety Factors")
        if "SDM" in method:
            c1, c2 = st.columns(2)
            fdl = c1.number_input("DL Factor", 1.4)
            fll = c2.number_input("LL Factor", 1.7)
        else:
            fdl, fll = 1.0, 1.0
            
        return {'fc':fc, 'fy':fy, 'fys':fys, 'b':b, 'h':h, 'cv':cover, 
                'fdl':fdl, 'fll':fll, 'method':method, 'unit':unit_sys}

def render_geometry():
    st.markdown('<div class="section-header">1️⃣ Geometry & Supports</div>', unsafe_allow_html=True)
    n = st.number_input("Number of Spans", 1, 10, 2)
    
    # Grid Spans
    spans = []
    st.markdown("**Span Lengths (m)**")
    cols = st.columns(4)
    for i in range(n):
        with cols[i%4]:
            spans.append(st.number_input(f"L{i+1}", 1.0, 50.0, 5.0, key=f"s{i}"))
            
    # Grid Supports
    st.markdown("**Supports**")
    sups = []
    sup_opts = ["Pin", "Roller", "Fixed", "None"]
    cols = st.columns(5)
    for i in range(n+1):
        with cols[i%5]:
            def_idx = 0 if i==0 else (1 if i<n else 1)
            sups.append(st.selectbox(f"S{i+1}", sup_opts, index=def_idx, key=f"sup{i}"))
            
    df_sup = pd.DataFrame({'x': [0]+list(np.cumsum(spans)), 'type': sups})
    ok, msg = check_stability(df_sup)
    if not ok: st.error(msg)
    return n, spans, df_sup, ok

def render_loads(n, spans, p):
    st.markdown('<div class="section-header">2️⃣ Loading Conditions</div>', unsafe_allow_html=True)
    loads = []
    tabs = st.tabs([f"Span {i+1}" for i in range(n)])
    
    for i, tab in enumerate(tabs):
        with tab:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Uniform Load")
                wdl = st.number_input("DL", 0.0, key=f"wdl{i}")
                wll = st.number_input("LL", 0.0, key=f"wll{i}")
                wu = wdl*p['fdl'] + wll*p['fll']
                if wu!=0: loads.append({'span_idx':i, 'type':'U', 'w':wu})
            with c2:
                st.markdown("#### Point Load")
                qty = st.number_input("Qty", 0, 5, 0, key=f"q{i}")
                for j in range(qty):
                    cc1, cc2, cc3 = st.columns(3)
                    pd = cc1.number_input(f"P_DL {j+1}", key=f"pd{i}{j}")
                    pl = cc2.number_input(f"P_LL {j+1}", key=f"pl{i}{j}")
                    px = cc3.number_input("x", 0.0, spans[i], spans[i]/2, key=f"px{i}{j}")
                    pu = pd*p['fdl'] + pl*p['fll']
                    if pu!=0: loads.append({'span_idx':i, 'type':'P', 'P':pu, 'x':px})
    return loads

# ==============================================================================
# 4. PLOTTING ENGINE (Detailed & Corrected)
# ==============================================================================
def draw_results(df, spans, supports, loads):
    cum_len = [0] + list(np.cumsum(spans))
    L_total = cum_len[-1]
    
    # Scale Factors
    max_load = max([abs(l['w']) for l in loads if l['type']=='U'] + [abs(l['P']) for l in loads if l['type']=='P'] + [100])
    viz_h = max_load * 1.5
    sup_h = viz_h * 0.15
    sup_w = max(0.4, L_total * 0.03)
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("<b>Loading Diagram</b>", "<b>Shear Force (V)</b>", "<b>Bending Moment (M)</b>"),
                        row_heights=[0.3, 0.35, 0.35])
    
    # 1. Loading
    fig.add_hline(y=0, line_color="black", line_width=3, row=1, col=1)
    # Supports
    for i, x in enumerate(cum_len):
        stype = supports.iloc[i]['type']
        if stype == "Pin":
            fig.add_trace(go.Scatter(x=[x, x-sup_w/2, x+sup_w/2, x], y=[0, -sup_h, -sup_h, 0], fill="toself", fillcolor="#B0BEC5", line_color="#455A64", showlegend=False), row=1, col=1)
        elif stype == "Roller":
            fig.add_trace(go.Scatter(x=[x], y=[-sup_h/2], mode="markers", marker=dict(size=12, color="#B0BEC5", line=dict(color="#455A64", width=2)), showlegend=False), row=1, col=1)
        elif stype == "Fixed":
            fig.add_shape(type="line", x0=x, y0=-sup_h, x1=x, y1=sup_h, line=dict(color="#455A64", width=5), row=1, col=1)
            
    # Loads
    for l in loads:
        if l['type'] == 'U':
            x1, x2 = cum_len[l['span_idx']], cum_len[l['span_idx']+1]
            h = (abs(l['w'])/max_load) * (viz_h * 0.6)
            fig.add_trace(go.Scatter(x=[x1, x2, x2, x1], y=[0, 0, h, h], fill="toself", fillcolor="rgba(255, 152, 0, 0.2)", line_width=0, showlegend=False), row=1, col=1)
            fig.add_annotation(x=(x1+x2)/2, y=h, text=f"<b>Wu={l['w']:.0f}</b>", showarrow=False, yshift=10, font=dict(color="#E65100"), row=1, col=1)
        elif l['type'] == 'P':
            px = cum_len[l['span_idx']] + l['x']
            h = (abs(l['P'])/max_load) * (viz_h * 0.6)
            fig.add_annotation(x=px, y=0, ax=px, ay=h, arrowcolor="black", arrowhead=2, text=f"<b>Pu={l['P']:.0f}</b>", yshift=10, row=1, col=1)
            
    fig.update_yaxes(visible=False, range=[-sup_h*1.5, viz_h*1.2], row=1, col=1)
    
    # 2. SFD
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color='#D32F2F', width=2), fillcolor='rgba(211, 47, 47, 0.1)', name="Shear"), row=2, col=1)
    # 3. BMD
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color='#1565C0', width=2), fillcolor='rgba(21, 101, 192, 0.1)', name="Moment"), row=3, col=1)
    
    # Labels Max/Min
    for col, row, color in [('shear', 2, '#D32F2F'), ('moment', 3, '#1565C0')]:
        arr = df[col].to_numpy()
        mx, mn = np.max(arr), np.min(arr)
        imx, imn = np.argmax(arr), np.argmin(arr)
        for val, idx, pos in [(mx, imx, "top"), (mn, imn, "bottom")]:
            if abs(val) > 0.1:
                ys = 15 if pos=="top" else -15
                fig.add_annotation(x=df['x'].iloc[idx], y=val, text=f"<b>{val:,.2f}</b>", showarrow=False, bgcolor="rgba(255,255,255,0.8)", font=dict(color=color, size=11), yshift=ys, row=row, col=1)
        rng = mx - mn
        pad = rng*0.2 if rng>0 else 1.0
        fig.update_yaxes(range=[mn-pad, mx+pad], row=row, col=1)

    fig.update_layout(height=900, template="plotly_white", margin=dict(t=50, b=50, l=50, r=50), showlegend=False)
    fig.update_xaxes(showgrid=True, gridcolor='#ECEFF1', title="Distance (m)", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# 5. DESIGN MODULE (RC CALCULATION)
# ==============================================================================
def perform_design(df, params):
    st.markdown('<div class="section-header">4️⃣ Reinforced Concrete Design</div>', unsafe_allow_html=True)
    
    # Determine Max Moment & Shear
    m_max = df['moment'].max()
    m_min = df['moment'].min() # Negative moment (Supports)
    v_max = df['shear'].abs().max()
    
    fc = params['fc']
    fy = params['fy']
    b = params['b']
    h = params['h']
    d = h - params['cv']
    
    results_html = []
    
    # --- CALCULATION LOGIC ---
    if "SDM" in params['method']:
        # SDM Parameters
        phi_b = 0.90
        phi_v = 0.75 # ACI318-19 shear
        beta1 = 0.85 if fc <= 280 else max(0.65, 0.85 - 0.05*(fc-280)/70)
        
        # Design Moment Capacity Check
        def design_moment(Mu, type_str):
            Mu_kgm = abs(Mu) * (1000 if 'kN' in params['unit'] else 1) # Convert to kg-m approx or keep consistency
            # Let's stick to unit consistency: Input is user defined. 
            # If User input kg, m -> Moment is kg-m. fc is ksc.
            # If User input kN, m -> Moment is kN-m. fc is MPa.
            
            # Unit Conversion Factor to kg-cm (for ksc inputs)
            if 'Metric' in params['unit']:
                M_design = abs(Mu) * 100 # kg-m to kg-cm
                fc_calc = fc
                fy_calc = fy
                b_calc = b
                d_calc = d
            else: # SI
                M_design = abs(Mu) * 1e6 # kN-m to N-mm
                fc_calc = fc # MPa
                fy_calc = fy # MPa
                b_calc = b * 10 # cm to mm
                d_calc = d * 10
            
            # Required Rho
            Rn = M_design / (phi_b * b_calc * d_calc**2)
            rho = (0.85 * fc_calc / fy_calc) * (1 - np.sqrt(1 - 2*Rn/(0.85*fc_calc)))
            
            rho_min = max(1.4/fy_calc, 0.25*np.sqrt(fc_calc)/fy_calc) if 'Metric' not in params['unit'] else 14/fy_calc
            rho_bal = 0.85 * beta1 * (fc_calc/fy_calc) * (6000/(6000+fy_calc)) if 'Metric' not in params['unit'] else 0.85 * beta1 * (fc_calc/fy_calc) * (6120/(6120+fy_calc))
            rho_max = 0.75 * rho_bal # Simplified
            
            As_req = rho * b_calc * d_calc
            As_min = rho_min * b_calc * d_calc
            
            # Steel Selection (Mockup)
            # Find number of DB12, DB16, DB20, DB25
            bars = [12, 16, 20, 25]
            select_str = ""
            if np.isnan(rho) or rho > rho_max:
                status = "❌ Section too small (Over Reinforced)"
                col_status = "red"
            elif rho < rho_min:
                As_req = As_min
                status = "⚠️ Minimum Steel Governs"
                col_status = "orange"
            else:
                status = "✅ OK"
                col_status = "green"
                
            if "Over" not in status:
                best_bar = ""
                for db in bars:
                    area = 3.1416 * (db/10 if 'Metric' in params['unit'] else db)**2 / 4 # cm2 or mm2
                    if 'Metric' in params['unit']: area = 3.1416*(db/10)**2/4
                    else: area = 3.1416*(db)**2/4
                    
                    num = math.ceil(As_req / area)
                    select_str += f"{num}-DB{db} "
            
            return {
                "Type": type_str,
                "Mu": abs(Mu),
                "As_req": As_req,
                "Status": status,
                "Bars": select_str
            }

        res_pos = design_moment(m_max, "Positive Moment (+M)")
        res_neg = design_moment(m_min, "Negative Moment (-M)")
        
        # Display Design Cards
        c1, c2 = st.columns(2)
        for c, r in zip([c1, c2], [res_pos, res_neg]):
            with c:
                st.markdown(f"""
                <div class="card">
                    <h4>{r['Type']}</h4>
                    <div style="font-size:2em; font-weight:bold; color:#1565C0;">{r['Mu']:,.2f} <span style="font-size:0.5em">unit</span></div>
                    <hr>
                    <p><b>Required As:</b> {r['As_req']:,.2f} cm²</p>
                    <p style="color:{'red' if 'Over' in r['Status'] else 'green'}"><b>{r['Status']}</b></p>
                    <div style="background:#eee; padding:10px; border-radius:5px;">
                        <b>Suggest:</b> {r['Bars']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.info("WSD Method logic would go here (Similar structure). Switched to SDM for brevity in this display.")
        
    # Shear Design (Simplified)
    st.markdown("#### Shear Reinforcement (Stirrups)")
    Vu = v_max
    if 'Metric' in params['unit']:
        vc = 0.53 * np.sqrt(fc) * b * d # kg
        phi_vc = 0.85 * vc 
        vu_val = Vu # kg
    else:
        vc = 0.17 * np.sqrt(fc) * b*10 * d*10 # N
        phi_vc = 0.75 * (vc / 1000) # kN
        vu_val = Vu # kN

    req_stirrup = "None"
    if vu_val <= phi_vc/2:
        req_stirrup = "Theoretical not required"
    elif vu_val <= phi_vc:
        req_stirrup = "Minimum Stirrups (Av_min)"
    else:
        req_stirrup = "Design Stirrups Required (Vs)"
        
    st.info(f"Max Shear Vu = {vu_val:,.2f} | Capacity $\phi V_c$ = {phi_vc:,.2f} -> **{req_stirrup}**")

# ==============================================================================
# 6. MAIN EXECUTION
# ==============================================================================
def main():
    st.markdown('<div class="header-box"><h1>🏗️ RC Beam Pro: Analysis & Design Suite</h1></div>', unsafe_allow_html=True)
    
    # 1. Inputs
    params = render_sidebar()
    c_geo, c_load = st.columns([1, 1.3])
    with c_geo:
        n, spans, sup_df, stable = render_geometry()
    with c_load:
        loads = render_loads(n, spans, params)
        
    st.markdown("---")
    if st.button("🚀 RUN ANALYSIS & DESIGN", type="primary", use_container_width=True, disabled=not stable):
        try:
            with st.spinner("Solving Stiffness Matrix..."):
                # Initialize Engine
                engine = BeamAnalysisEngine(spans, sup_df, loads)
                df_res, reactions = engine.solve()
                
                # --- RESULTS ---
                st.markdown('<div class="section-header">3️⃣ Analysis Results</div>', unsafe_allow_html=True)
                
                # Summary Tables
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Support Reactions**")
                    # Map DOF reactions to readable format
                    reac_data = []
                    # Simplified extraction from engine logic
                    # Just showing raw vector for now to save space, normally would map to nodes
                    st.dataframe(pd.DataFrame(reactions[:(n+1)*2].reshape(-1,2), columns=["Fy", "Mz"]), use_container_width=True)
                with c2:
                    st.markdown("**Max Forces**")
                    st.write(f"Max Shear: **{df_res['shear'].abs().max():,.2f}**")
                    st.write(f"Max Moment: **{df_res['moment'].abs().max():,.2f}**")

                # Diagrams
                draw_results(df_res, spans, sup_df, loads)
                
                # Design
                perform_design(df_res, params)
                
        except Exception as e:
            st.error(f"Calculation Error: {e}")
            st.code(e)

if __name__ == "__main__":
    main()
