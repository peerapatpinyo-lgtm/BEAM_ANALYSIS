import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ==========================================
# 🧠 PART 1: BEAM ENGINE (ฝังมาในไฟล์เดียวเลย)
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
        self.K = np.zeros((self.dof, self.dof))
        self.F = np.zeros(self.dof)
        self.u = None
        self.R = None
        
    def solve(self):
        # 1. Stiffness Matrix
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

        # 2. Loads
        for load in self.loads:
            span_idx = load['span_idx']
            L = self.spans[span_idx]
            n1, n2 = span_idx, span_idx + 1
            idx = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
            fem = np.zeros(4)
            
            if load['type'] == 'Uniform':
                w = load['total_w']
                fem = np.array([-w*L/2, -w*L**2/12, -w*L/2, w*L**2/12])
            elif load['type'] == 'Point':
                P = load['total_w']
                a = load['pos']
                b = L - a
                fem = np.array([
                    -(P*b**2*(3*a+b))/L**3, -(P*a*b**2)/L**2,
                    -(P*a**2*(a+3*b))/L**3, (P*a**2*b)/L**2
                ])
            self.F[idx] += fem

        # 3. Supports
        fixed_dof = []
        for i, supp in enumerate(self.supports):
            if supp in ['Pin', 'Roller']: fixed_dof.append(2*i)
            elif supp == 'Fix': fixed_dof.extend([2*i, 2*i+1])
        
        free_dof = [x for x in range(self.dof) if x not in fixed_dof]
        if not free_dof: return None, "Fully constrained"
        
        try:
            u_f = np.linalg.solve(self.K[np.ix_(free_dof, free_dof)], self.F[free_dof])
        except np.linalg.LinAlgError:
            return None, "Unstable Structure"
            
        self.u = np.zeros(self.dof)
        self.u[free_dof] = u_f
        self.R = self.K @ self.u - self.F
        return self.u, None

    def get_internal_forces(self, num_points=100):
        x_tot, v_list, m_list = [], [], []
        node_ry = {i: self.R[2*i] for i in range(self.nodes)}
        node_rm = {i: self.R[2*i+1] for i in range(self.nodes)}
        cur_x = 0
        for i, L in enumerate(self.spans):
            x_loc = np.linspace(0, L, num_points)
            x_glob = cur_x + x_loc
            for x_curr in x_glob:
                V, M = 0.0, 0.0
                # Reactions
                for ni in range(i+1):
                    rx = sum(self.spans[:ni])
                    if rx <= x_curr + 1e-6:
                        V += node_ry[ni]
                        M += -node_rm[ni] + node_ry[ni]*(x_curr - rx)
                # Loads
                for load in self.loads:
                    lx = sum(self.spans[:load['span_idx']])
                    if load['type'] == 'Point':
                        px = lx + load['pos']
                        if px <= x_curr:
                            P = load['total_w']
                            V -= P
                            M -= P*(x_curr - px)
                    elif load['type'] == 'Uniform':
                        lx_e = lx + self.spans[load['span_idx']]
                        es, ee = lx, min(x_curr, lx_e)
                        if ee > es:
                            w = load['total_w']
                            V -= w*(ee-es)
                            M -= w*(ee-es)*(x_curr - (es+ee)/2)
                v_list.append(V)
                m_list.append(M)
            x_tot.extend(x_glob)
            v_list.extend(v_list)
            cur_x += L
        # Convert internal units (N, Nm) to kN, kNm
        return pd.DataFrame({'x': x_tot, 'shear': np.array(v_list)/1000, 'moment': np.array(m_list)/1000})

# ==========================================
# ⚙️ PART 2: UI & CONFIG
# ==========================================
st.set_page_config(page_title="Beam Pro V10", layout="wide")

st.sidebar.header("⚙️ Settings")
unit_opt = st.sidebar.radio("ระบบหน่วย (Unit System)", ["SI Units (kN, m)", "MKS Units (kg, m)"])

if "kN" in unit_opt:
    UNIT_F = "kN"; UNIT_D = "m"; UNIT_M = "kN-m"; UNIT_L = "kN/m"
    TO_N = 1000.0 # Input to Engine (N)
else:
    UNIT_F = "kg"; UNIT_D = "m"; UNIT_M = "kg-m"; UNIT_L = "kg/m"
    TO_N = 9.80665 # Input to Engine (N)

def add_peak_labels(fig, x, y, inverted=False):
    max_i, min_i = np.argmax(y), np.argmin(y)
    peaks = [(x[max_i], y[max_i]), (x[min_i], y[min_i])]
    for px, py in peaks:
        shift = 15 if (py >= 0 and not inverted) or (py < 0 and inverted) else -15
        fig.add_annotation(x=px, y=py, text=f"{py:.2f}", showarrow=False, yshift=shift, font=dict(color="black", size=11))

def draw_beam(spans, supports, loads):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, sum(spans)], y=[0, 0], mode='lines', line=dict(color='black', width=4)))
    cx, sx = 0, 0
    for i, L in enumerate(spans):
        fig.add_annotation(x=cx+L/2, y=-0.6, text=f"{L} m", showarrow=False, font=dict(color="blue"))
        fig.add_shape(type="line", x0=cx+L, y0=-0.3, x1=cx+L, y1=0.3, line=dict(color="gray", dash="dot"))
        cx += L
    for i, s in enumerate(supports):
        sym = "triangle-up" if s != "Fix" else "square"
        col = "green" if s != "Fix" else "red"
        fig.add_trace(go.Scatter(x=[sx], y=[-0.2], mode='markers', marker=dict(symbol=sym, size=12, color=col), showlegend=False))
        if i < len(spans): sx += spans[i]
    for ld in loads:
        sx = sum(spans[:ld['span_idx']])
        val = ld['display']
        if ld['type'] == 'Point':
            fig.add_annotation(x=sx+ld['pos'], y=0.1, ax=0, ay=-30, text=f"{val:.1f} {UNIT_F}", arrowcolor="red")
        elif ld['type'] == 'Uniform':
            ex = sx + spans[ld['span_idx']]
            fig.add_shape(type="rect", x0=sx, y0=0, x1=ex, y1=0.25, fillcolor="rgba(255,0,0,0.15)", line_width=0)
            fig.add_annotation(x=(sx+ex)/2, y=0.3, text=f"{val:.1f} {UNIT_L}", showarrow=False, font=dict(color="red", size=10))
    fig.update_layout(height=250, xaxis=dict(showgrid=False, visible=True), yaxis=dict(visible=False, range=[-1, 1]), margin=dict(t=30, b=20))
    return fig

# ==========================================
# 🖥️ MAIN APP
# ==========================================
st.title(f"🏗️ RC Beam Design (Complete V10) [{unit_opt.split()[0]}]")

# 1. INPUT
with st.expander("1. Geometry & Supports", expanded=True):
    c1, c2 = st.columns([1, 2])
    n_span = c1.number_input("Spans", 1, 6, 2)
    spans = [st.columns(n_span)[i].number_input(f"L{i+1}", 1.0, 20.0, 4.0) for i in range(n_span)]
    supports = [st.columns(n_span+1)[i].selectbox(f"S{i+1}", ['Pin', 'Roller', 'Fix'], index=0 if i==0 else 1) for i in range(n_span+1)]

st.subheader(f"2. Loads (Factored: 1.4DL + 1.7LL)")
loads_in = []
cols = st.columns(n_span)
for i in range(n_span):
    with cols[i]:
        st.info(f"Span {i+1}")
        udl = st.number_input(f"UDL ({UNIT_L})", 0.0, key=f"u_{i}")
        ull = st.number_input(f"ULL ({UNIT_L})", 0.0, key=f"ul_{i}")
        if (udl+ull)>0:
            loads_in.append({'span_idx': i, 'type': 'Uniform', 'total_w': (1.4*udl+1.7*ull)*TO_N, 'display': udl+ull})
        
        n_pt = st.number_input(f"Points", 0, 5, 0, key=f"np_{i}")
        for j in range(n_pt):
            pd = st.number_input(f"P{j+1}DL", 0.0, key=f"pd_{i}_{j}")
            pl = st.number_input(f"P{j+1}LL", 0.0, key=f"pl_{i}_{j}")
            pp = st.number_input(f"Pos", 0.0, spans[i], spans[i]/2, key=f"pp_{i}_{j}")
            if (pd+pl)>0:
                loads_in.append({'span_idx': i, 'type': 'Point', 'total_w': (1.4*pd+1.7*pl)*TO_N, 'pos': pp, 'display': pd+pl})

if st.button("🚀 Calculate Analysis", type="primary", use_container_width=True):
    solver = SimpleBeamSolver(spans, supports, loads_in)
    u, err = solver.solve()
    if err:
        st.error(err)
    else:
        df = solver.get_internal_forces(100)
        # Convert Result Units
        conv = 1.0 if "kN" in unit_opt else (1000.0/9.80665)
        df['shear_d'] = df['shear'] * conv
        df['moment_d'] = df['moment'] * conv
        
        st.session_state['res'] = df
        st.session_state['done'] = True
        st.session_state['viz'] = draw_beam(spans, supports, loads_in)

# 3. RESULTS
if st.session_state.get('done', False):
    df = st.session_state['res']
    st.plotly_chart(st.session_state['viz'], use_container_width=True)
    
    c1, c2 = st.columns(2)
    fig_v = go.Figure(go.Scatter(x=df['x'], y=df['shear_d'], fill='tozeroy', line=dict(color='#D32F2F')))
    add_peak_labels(fig_v, df['x'], df['shear_d'])
    fig_v.update_layout(title=f"Shear ({UNIT_F})", hovermode="x")
    c1.plotly_chart(fig_v, use_container_width=True)
    
    fig_m = go.Figure(go.Scatter(x=df['x'], y=df['moment_d'], fill='tozeroy', line=dict(color='#1976D2')))
    add_peak_labels(fig_m, df['x'], df['moment_d'], inverted=True)
    fig_m.update_layout(title=f"Moment ({UNIT_M})", yaxis=dict(autorange="reversed"))
    c2.plotly_chart(fig_m, use_container_width=True)

    # 4. DESIGN
    st.markdown("---")
    st.header("🛠️ Design (Interactive)")
    cd1, cd2 = st.columns([1, 1.5])
    
    with cd1:
        with st.form("des"):
            fc = st.number_input("f'c (ksc)", value=240.0)
            fy = st.number_input("fy (ksc)", value=4000.0)
            b = st.number_input("Width b (cm)", 15.0, 100.0, 25.0)
            h = st.number_input("Depth h (cm)", 20.0, 200.0, 50.0)
            cov = st.number_input("Cover (cm)", 2.0, 5.0, 3.0)
            bar_sz = st.selectbox("Main Bar", ['DB12','DB16','DB20','DB25'], index=1)
            st.form_submit_button("Recalculate")

    with cd2:
        # Convert inputs to SI (N, mm, MPa)
        fc_mpa = fc * 0.0980665
        fy_mpa = fy * 0.0980665
        b_mm, d_mm = b*10, (h-cov)*10
        
        # Max Moment
        Mu_disp = max(abs(df['moment_d'].max()), abs(df['moment_d'].min()))
        Mu_Nmm = Mu_disp * (1e6 if "kN" in unit_opt else 9.80665*1000)
        
        # Flexure Check
        phi_b = 0.9
        Rn = (Mu_Nmm/phi_b) / (b_mm * d_mm**2)
        m = fy_mpa / (0.85*fc_mpa)
        term = 1 - (2*m*Rn)/fy_mpa
        
        st.subheader("1. Flexural Design")
        if term < 0:
            st.error("❌ Section Fail (Too Small)")
        else:
            rho = (1/m)*(1 - np.sqrt(term))
            rho_min = max(np.sqrt(fc_mpa)/(4*fy_mpa), 1.4/fy_mpa)
            As = max(rho, rho_min) * b_mm * d_mm / 100
            
            bar_areas = {'DB12':1.13, 'DB16':2.01, 'DB20':3.14, 'DB25':4.91}
            nb = max(2, int(np.ceil(As/bar_areas[bar_sz])))
            st.success(f"✅ OK | Mu={Mu_disp:.2f}")
            st.latex(f"A_{{s,req}} = {As:.2f} cm^2 \\rightarrow Use\\ {nb}-{bar_sz}")

        # Shear Check (Added Back!)
        st.subheader("2. Shear Design (Stirrups)")
        Vu_disp = df['shear_d'].abs().max()
        Vu_N = Vu_disp * (1000.0 if "kN" in unit_opt else 9.80665)
        
        Vc = 0.17 * np.sqrt(fc_mpa) * b_mm * d_mm # N
        phi_v = 0.85
        phiVc = phi_v * Vc
        
        phiVc_disp = phiVc / (1000.0 if "kN" in unit_opt else 9.80665)
        
        st.write(f"Vu_max = {Vu_disp:.2f} {UNIT_F} | $\\phi V_c$ = {phiVc_disp:.2f} {UNIT_F}")
        
        if Vu_N <= phiVc / 2:
            st.info("✅ No Stirrups theoretically required.")
        elif Vu_N <= phiVc:
            st.warning("⚠️ Min Stirrups required.")
            st.latex("Use\\ RB6 @ 20 cm (Min)")
        else:
            Vs = (Vu_N - phiVc) / phi_v
            # Simple spacing calc for RB6 (2 legs) -> Av = 2*28mm2 = 56mm2
            Av = 2 * 28.27 # RB6
            s_req = (Av * fy_mpa * d_mm) / Vs
            st.error(f"❗ Stirrups Required")
            st.latex(f"V_s = {(Vs/1000):.1f} kN \\rightarrow Use\\ RB6 @ {min(s_req/10, d_mm/20):.0f} cm")
