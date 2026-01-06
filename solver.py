import numpy as np
import pandas as pd
from scipy.linalg import solve

class BeamSolver:
    def __init__(self, spans, supports_input, loads_input, E, b=0.3, h=0.5, I_custom=None):
        self.spans = [float(s) for s in spans]
        self.E = float(E)
        self.b = b
        self.h = h
        # 1. Moment of Inertia (I) Logic
        self.I = float(I_custom) if I_custom else (b * h**3) / 12
        self.cum_spans = [round(x, 4) for x in ([0.0] + list(np.cumsum(self.spans)))]
        self.loads_df = self._sanitize_loads(loads_input)
        self.supports_df = self._sanitize_supports(supports_input)

    def _sanitize_supports(self, supports_input):
        # แปลงข้อมูล Support ให้เป็น DataFrame ที่มีพิกัด x จริง
        sanitized = []
        for s in supports_input:
            idx = int(s['id'])
            if idx < len(self.cum_spans):
                sanitized.append({'x': self.cum_spans[idx], 'type': s['type'], 'id': idx})
        return pd.DataFrame(sanitized)

    def _sanitize_loads(self, loads_input):
        if not loads_input: return pd.DataFrame(columns=['span_index', 'type', 'mag', 'x', 'dist', 'case'])
        return pd.DataFrame(loads_input)

    def solve(self):
        try:
            # --- Node Generation (Internal Nodes for Loads) ---
            pts = self.cum_spans.copy()
            for _, l in self.loads_df.iterrows():
                gx = self.cum_spans[int(l['span_index'])] + float(l['x'])
                pts.append(gx)
                if l['type'] == 'U': pts.append(round(gx + float(l['dist']), 4))
            
            nodes = sorted(list(set([round(p, 4) for p in pts])))
            num_nodes = len(nodes)
            dof = 2 * num_nodes
            K, F = np.zeros((dof, dof)), np.zeros(dof)

            for i in range(num_nodes - 1):
                L = nodes[i+1] - nodes[i]
                if L > 1e-5:
                    # Element Stiffness Matrix
                    EI = self.E * self.I
                    k_el = (EI / L**3) * np.array([
                        [12, 6*L, -12, 6*L], [6*L, 4*L**2, -6*L, 2*L**2],
                        [-12, -6*L, 12, -6*L], [6*L, 2*L**2, -6*L, 4*L**2]
                    ])
                    idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
                    K[np.ix_(idx, idx)] += k_el

            # --- 2. Load Application (P, U, M) ---
            for _, l in self.loads_df.iterrows():
                gx = self.cum_spans[int(l['span_index'])] + float(l['x'])
                if l['type'] == 'P':
                    nid = np.argmin([abs(n - gx) for n in nodes])
                    F[2*nid] -= float(l['mag'])
                elif l['type'] == 'M': # Moment Load
                    nid = np.argmin([abs(n - gx) for n in nodes])
                    F[2*nid+1] += float(l['mag'])
                elif l['type'] == 'U':
                    s_g, e_g = gx, round(gx + float(l['dist']), 4)
                    for i in range(num_nodes - 1):
                        n1, n2 = nodes[i], nodes[i+1]
                        overlap = min(e_g, n2) - max(s_g, n1)
                        if overlap > 1e-5:
                            w, L_el = float(l['mag']), n2 - n1
                            F[2*i] -= (w * L_el / 2); F[2*i+1] -= (w * L_el**2 / 12)
                            F[2*(i+1)] -= (w * L_el / 2); F[2*(i+1)+1] += (w * L_el**2 / 12)

            # Boundary Conditions
            free_dof = np.full(dof, True)
            for _, sup in self.supports_df.iterrows():
                nid = np.argmin([abs(n - sup['x']) for n in nodes])
                if sup['type'] in ['Pin', 'Roller', 'Fixed']: free_dof[2*nid] = False
                if sup['type'] == 'Fixed': free_dof[2*nid+1] = False

            U = np.zeros(dof)
            if any(free_dof):
                U[free_dof] = solve(K[np.ix_(free_dof, free_dof)], F[free_dof])

            R_full = K @ U - F
            
            # --- Result Calculation ---
            results = []
            for x in np.linspace(0, nodes[-1], 500):
                V, M, defl = 0.0, 0.0, 0.0
                for i, n_p in enumerate(nodes):
                    if n_p <= x + 1e-5:
                        V += R_full[2*i]; M += R_full[2*i]*(x - n_p) + R_full[2*i+1]
                for _, l in self.loads_df.iterrows():
                    gx = self.cum_spans[int(l['span_index'])] + float(l['x'])
                    if l['type'] == 'P' and gx <= x + 1e-5:
                        V -= l['mag']; M -= l['mag']*(x - gx)
                    elif l['type'] == 'M' and gx <= x + 1e-5:
                        M -= l['mag']
                    elif l['type'] == 'U' and gx < x:
                        d = min(x, gx + l['dist']) - gx
                        V -= l['mag']*d; M -= l['mag']*d*(x - (gx + d/2))
                
                # Deflection calculation (Shape Functions)
                for i in range(num_nodes - 1):
                    if nodes[i] <= x <= nodes[i+1] + 1e-5:
                        L_el, s = nodes[i+1] - nodes[i], (x - nodes[i]) / (nodes[i+1] - nodes[i])
                        H = np.array([1-3*s**2+2*s**3, L_el*(s-2*s**2+s**3), 3*s**2-2*s**3, L_el*(s**3-s**2)])
                        defl = np.dot(H, U[2*i:2*i+4])
                        break
                results.append({'x': x, 'deflection': defl, 'shear': V, 'moment': M})

            df_res = pd.DataFrame(results)
            return df_res, R_full, self._create_summary(df_res)
        except Exception as e:
            return pd.DataFrame(), None, {"error": str(e)}

    def _create_summary(self, df):
        # 5. Deep Beam Check (L/d < 2)
        d_eff = self.h - 0.05
        is_deep = any((L / d_eff) < 2.0 for L in self.spans)
        return {
            'V_max': df['shear'].abs().max(),
            'M_pos': df['moment'].max(),
            'M_neg': df['moment'].min(),
            'D_max': df['deflection'].abs().max(),
            'is_deep': is_deep, 'd_eff': d_eff
        }
