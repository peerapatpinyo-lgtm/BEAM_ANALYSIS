import numpy as np
import pandas as pd
from scipy.linalg import solve

class BeamSolver:
    def __init__(self, spans, supports_input, loads_input, E, I, A=None, G=None, b=0.3, h=0.5):
        self.spans = [float(s) for s in spans]
        self.E = float(E)
        self.I = float(I)
        self.b = b
        self.h = h
        self.A = float(A) if A is not None else (b * h)
        self.G = float(G) if G is not None else 7.7e10
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
            try:
                idx = int(raw_id)
                if 0 <= idx < len(self.cum_spans):
                    sanitized.append({'x': self.cum_spans[idx], 'type': stype, 'id': idx})
            except: continue
        return pd.DataFrame(sanitized)

    def _sanitize_loads(self, loads_input):
        if not loads_input: return pd.DataFrame(columns=['span_index', 'type', 'mag', 'x', 'dist'])
        df = pd.DataFrame(loads_input)
        def get_global_x(row):
            s_idx = int(row.get('span_index', row.get('span_idx', 0)))
            lx = float(row.get('x', 0))
            return round(self.cum_spans[s_idx] + lx, 4)
        df['x'] = df.apply(get_global_x, axis=1)
        return df

    def solve(self):
        try:
            pts = self.cum_spans.copy()
            for _, l in self.loads_df.iterrows():
                pts.append(l['x'])
                if l['type'] == 'U': pts.append(round(l['x'] + l['dist'], 4))
            
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

            for _, l in self.loads_df.iterrows():
                if l['type'] == 'P':
                    nid = np.argmin([abs(n - l['x']) for n in nodes])
                    F[2*nid] -= l['mag']
                elif l['type'] == 'M':
                    nid = np.argmin([abs(n - l['x']) for n in nodes])
                    F[2*nid+1] += l['mag']
                elif l['type'] == 'U':
                    s_g, e_g = l['x'], round(l['x'] + l['dist'], 4)
                    for i in range(num_nodes - 1):
                        n1, n2 = nodes[i], nodes[i+1]
                        L_el = n2 - n1
                        overlap = min(e_g, n2) - max(s_g, n1)
                        if overlap > 1e-5:
                            w = l['mag']
                            F[2*i] -= (w * L_el / 2); F[2*i+1] -= (w * L_el**2 / 12)
                            F[2*(i+1)] -= (w * L_el / 2); F[2*(i+1)+1] += (w * L_el**2 / 12)

            free_dof = np.full(dof, True)
            for _, sup in self.supports_df.iterrows():
                nid = np.argmin([abs(n - sup['x']) for n in nodes])
                if sup['type'] in ['Pin', 'Roller', 'Fixed']: free_dof[2*nid] = False
                if sup['type'] == 'Fixed': free_dof[2*nid+1] = False

            U = np.zeros(dof)
            if any(free_dof):
                U[free_dof] = solve(K[np.ix_(free_dof, free_dof)], F[free_dof])

            R_full = K @ U - F
            r_mapped = np.zeros(2 * (len(self.spans) + 1))
            for i, target_x in enumerate(self.cum_spans):
                nid = np.argmin([abs(n - target_x) for n in nodes])
                r_mapped[2*i] = R_full[2*nid]
                r_mapped[2*i+1] = R_full[2*nid+1]

            results = []
            for x in np.linspace(0, nodes[-1], 300):
                V, M, defl = 0.0, 0.0, 0.0
                for i, n_p in enumerate(nodes):
                    if n_p <= x + 1e-5:
                        V += R_full[2*i]; M += R_full[2*i]*(x - n_p) + R_full[2*i+1]
                for _, l in self.loads_df.iterrows():
                    if l['type'] == 'P' and l['x'] <= x + 1e-5:
                        V -= l['mag']; M -= l['mag']*(x - l['x'])
                    elif l['type'] == 'M' and l['x'] <= x + 1e-5:
                        M -= l['mag']
                    elif l['type'] == 'U' and l['x'] < x:
                        d = min(x, l['x'] + l['dist']) - l['x']
                        V -= l['mag']*d; M -= l['mag']*d*(x - (l['x'] + d/2))
                
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
            return df_res, r_mapped, self.last_summ
        except:
            return pd.DataFrame(), np.zeros(2 * (len(self.spans) + 1)), {}

    def _get_k(self, L):
        EI = self.E * self.I
        return (EI / L**3) * np.array([[12, 6*L, -12, 6*L], [6*L, 4*L**2, -6*L, 2*L**2], [-12, -6*L, 12, -6*L], [6*L, 2*L**2, -6*L, 4*L**2]])

    def _create_summary(self, df):
        if df.empty: return {}
        return {
            'V_max': {'value': df['shear'].abs().max(), 'x': df.iloc[df['shear'].abs().idxmax()]['x']},
            'M_pos': {'value': df['moment'].max(), 'x': df.iloc[df['moment'].idxmax()]['x']},
            'M_neg': {'value': df['moment'].min(), 'x': df.iloc[df['moment'].idxmin()]['x']},
            'D_max': {'value': df['deflection'].abs().max(), 'x': df.iloc[df['deflection'].abs().idxmax()]['x']}
        }

    def design_rc_section(self, fc, fy):
        try:
            import rc_design
            res = rc_design.calculate_reinforcement(self.last_summ, self.b, self.h, fc, fy)
            return res
        except: return {'as_pos': 0, 'as_neg': 0}
