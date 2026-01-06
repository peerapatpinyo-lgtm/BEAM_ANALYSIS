import numpy as np
import pandas as pd
from scipy.linalg import solve

class BeamSolver:
    def __init__(self, spans, supports_input, loads_input, E, b=0.3, h=0.5, I_custom=None):
        self.spans = [float(s) for s in spans]
        self.E = float(E)
        self.b = b
        self.h = h
        # 1. จัดการค่า I: ถ้ามีค่า Custom ให้ใช้ ถ้าไม่มีให้คำนวณจาก bh^3/12
        self.I = float(I_custom) if I_custom else (b * h**3) / 12
        self.cum_spans = [round(x, 4) for x in ([0.0] + list(np.cumsum(self.spans)))]
        self.loads_df = self._sanitize_loads(loads_input)
        self.supports_df = self._sanitize_supports(supports_input)
        self.last_summ = None

    def _sanitize_supports(self, supports_input):
        sanitized = []
        data = supports_input.to_dict('records') if hasattr(supports_input, 'to_dict') else supports_input
        for s in data:
            stype = str(s.get('type', s.get('Support Type', 'None')))
            if stype == "None": continue
            raw_id = s.get('id', s.get('Node ID'))
            idx = int(raw_id)
            if idx < len(self.cum_spans):
                sanitized.append({'x': self.cum_spans[idx], 'type': stype, 'id': idx})
        return pd.DataFrame(sanitized)

    def _sanitize_loads(self, loads_input):
        if not loads_input: return pd.DataFrame(columns=['span_index', 'type', 'mag', 'x', 'dist', 'case'])
        return pd.DataFrame(loads_input)

    def solve(self):
        try:
            # --- Node Generation ---
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
                    k_el = self._get_k(L)
                    idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
                    K[np.ix_(idx, idx)] += k_el

            # --- Load Application (Including Moment Load) ---
            for _, l in self.loads_df.iterrows():
                gx = self.cum_spans[int(l['span_index'])] + float(l['x'])
                if l['type'] == 'P':
                    nid = np.argmin([abs(n - gx) for n in nodes])
                    F[2*nid] -= float(l['mag'])
                elif l['type'] == 'M': # 2. Moment Load Logic
                    nid = np.argmin([abs(n - gx) for n in nodes])
                    F[2*nid+1] += float(l['mag'])
                elif l['type'] == 'U':
                    s_g, e_g = gx, round(gx + float(l['dist']), 4)
                    for i in range(num_nodes - 1):
                        n1, n2 = nodes[i], nodes[i+1]
                        L_el = n2 - n1
                        overlap = min(e_g, n2) - max(s_g, n1)
                        if overlap > 1e-5:
                            w = float(l['mag'])
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
            
            # --- Results Generation ---
            results = []
            for x in np.linspace(0, nodes[-1], 400):
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
                
                for i in range(num_nodes - 1):
                    if nodes[i] <= x <= nodes[i+1] + 1e-5:
                        L_el = nodes[i+1] - nodes[i]
                        s = (x - nodes[i]) / L_el
                        H = np.array([1-3*s**2+2*s**3, L_el*(s-2*s**2+s**3), 3*s**2-2*s**3, L_el*(s**3-s**2)])
                        defl = np.dot(H, U[2*i:2*i+4])
                        break
                results.append({'x': x, 'deflection': defl, 'shear': V, 'moment': M})

            df_res = pd.DataFrame(results)
            self.last_summ = self._create_summary(df_res)
            return df_res, R_full, self.last_summ
        except Exception as e:
            return pd.DataFrame(), None, {"error": str(e)}

    def _get_k(self, L):
        EI = self.E * self.I
        return (EI / L**3) * np.array([[12, 6*L, -12, 6*L], [6*L, 4*L**2, -6*L, 2*L**2], [-12, -6*L, 12, -6*L], [6*L, 2*L**2, -6*L, 4*L**2]])

    def _create_summary(self, df):
        # 4 & 5. Analysis Summary & Deep Beam Check (L/d < 2)
        v_max = df['shear'].abs().max()
        m_max_pos = df['moment'].max()
        m_max_neg = df['moment'].min()
        d_max = df['deflection'].abs().max()
        
        d_eff = self.h - 0.05 # ระยะ d โดยประมาณ
        is_deep = any((L / d_eff) < 2.0 for L in self.spans)

        return {
            'V_max': v_max, 'M_pos': m_max_pos, 'M_neg': m_max_neg, 'D_max': d_max,
            'is_deep': is_deep, 'd_eff': d_eff
        }
