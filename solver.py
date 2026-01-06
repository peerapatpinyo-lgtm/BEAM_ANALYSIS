import numpy as np
import pandas as pd
from scipy.linalg import solve

class BeamSolver:
    def __init__(self, spans, supports_input, loads_input, E, b=0.3, h=0.5, I_custom=None):
        self.spans = [float(s) for s in spans]
        self.E = float(E)
        self.b, self.h = b, h
        self.I = float(I_custom) if I_custom else (b * h**3) / 12
        self.G = self.E / (2 * (1 + 0.2)) 
        self.As = (5/6) * (b * h)        
        self.cum_spans = [round(x, 4) for x in ([0.0] + list(np.cumsum(self.spans)))]
        self.loads_df = pd.DataFrame(loads_input)
        self.supports_df = pd.DataFrame(supports_input)

    def _get_phi(self, L):
        EI = self.E * self.I
        return (12 * EI) / (L**2 * self.G * self.As)

    def _get_k_timoshenko(self, L):
        EI, Phi = self.E * self.I, self._get_phi(L)
        coeff = EI / (L**3 * (1 + Phi))
        return coeff * np.array([
            [12, 6*L, -12, 6*L],
            [6*L, (4+Phi)*L**2, -6*L, (2-Phi)*L**2],
            [-12, -6*L, 12, -6*L],
            [6*L, (2-Phi)*L**2, -6*L, (4+Phi)*L**2]
        ])

    def solve(self):
        try:
            pts = self.cum_spans.copy()
            for _, l in self.loads_df.iterrows():
                gx = self.cum_spans[int(l['span_index'])] + float(l['x'])
                pts.append(gx)
                if l['type'] == 'U': pts.append(round(gx + float(l['dist']), 4))
            nodes = sorted(list(set([round(p, 4) for p in pts])))
            num_nodes, dof = len(nodes), 2 * len(nodes)
            K, F = np.zeros((dof, dof)), np.zeros(dof)
            t_load_fy, t_load_m0 = 0.0, 0.0

            for i in range(num_nodes - 1):
                L = nodes[i+1] - nodes[i]
                if L > 1e-5:
                    idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
                    K[np.ix_(idx, idx)] += self._get_k_timoshenko(L)

            for _, l in self.loads_df.iterrows():
                gx = self.cum_spans[int(l['span_index'])] + float(l['x'])
                mag = float(l['mag'])
                if l['type'] == 'P':
                    nid = np.argmin([abs(n - gx) for n in nodes]); F[2*nid] -= mag
                    t_load_fy += mag; t_load_m0 += mag * gx
                elif l['type'] == 'M':
                    nid = np.argmin([abs(n - gx) for n in nodes]); F[2*nid+1] += mag
                    t_load_m0 -= mag 
                elif l['type'] == 'U':
                    dist = float(l['dist'])
                    t_load_fy += mag * dist; t_load_m0 += (mag * dist) * (gx + dist/2)
                    for i in range(num_nodes - 1):
                        overlap = min(gx+dist, nodes[i+1]) - max(gx, nodes[i])
                        if overlap > 1e-5:
                            w, Le = mag, nodes[i+1] - nodes[i]
                            F[2*i] -= (w * Le / 2); F[2*i+1] -= (w * Le**2 / 12)
                            F[2*(i+1)] -= (w * Le / 2); F[2*(i+1)+1] += (w * Le**2 / 12)

            free_d = np.full(dof, True)
            for _, s in self.supports_df.iterrows():
                if s['type'] == "None": continue
                nid = np.argmin([abs(n - self.cum_spans[int(s['id'])]) for n in nodes])
                if s['type'] in ['Pin', 'Roller', 'Fixed']: free_d[2*nid] = False
                if s['type'] == 'Fixed': free_d[2*nid+1] = False

            U = np.zeros(dof)
            U[free_d] = solve(K[np.ix_(free_d, free_d)], F[free_d])
            R = K @ U - F

            reac_res = []
            t_reac_fy, t_reac_m0 = 0.0, 0.0
            for _, s in self.supports_df.iterrows():
                if s['type'] == "None": continue
                nid = np.argmin([abs(n - self.cum_spans[int(s['id'])]) for n in nodes])
                t_reac_fy += R[2*nid]
                t_reac_m0 += (R[2*nid] * self.cum_spans[int(s['id'])]) + R[2*nid+1]
                reac_res.append({'id': int(s['id']), 'type': s['type'], 'Ry (kN)': round(R[2*nid]/1000, 3), 'M (kNm)': round(R[2*nid+1]/1000, 3)})

            res = []
            for x in np.linspace(0, nodes[-1], 500):
                v_sh, m_bm, d_defl = 0.0, 0.0, 0.0
                for i, np_x in enumerate(nodes):
                    if np_x <= x + 1e-5:
                        v_sh += R[2*i]; m_bm += R[2*i]*(x - np_x) + R[2*i+1]
                for _, l in self.loads_df.iterrows():
                    gx = self.cum_spans[int(l['span_index'])] + float(l['x'])
                    if l['type'] == 'P' and gx <= x + 1e-5: v_sh -= l['mag']; m_bm -= l['mag']*(x - gx)
                    elif l['type'] == 'M' and gx <= x + 1e-5: m_bm -= l['mag']
                    elif l['type'] == 'U' and gx < x:
                        d = min(x, gx + l['dist']) - gx
                        v_sh -= l['mag']*d; m_bm -= l['mag']*d*(x - (gx + d/2))
                for i in range(num_nodes - 1):
                    if nodes[i] <= x <= nodes[i+1] + 1e-5:
                        Le, xi = nodes[i+1] - nodes[i], (x - nodes[i]) / (nodes[i+1] - nodes[i])
                        Phi = self._get_phi(Le)
                        N1 = (1/(1+Phi))*(1-3*xi**2+2*xi**3+Phi*(1-xi))
                        N2 = (Le/(1+Phi))*(xi-2*xi**2+xi**3+0.5*Phi*(xi-xi**2))
                        N3 = (1/(1+Phi))*(3*xi**2-2*xi**3+Phi*xi)
                        N4 = (Le/(1+Phi))*(-xi**2+xi**3-0.5*Phi*(xi-xi**2))
                        d_defl = N1*U[2*i] + N2*U[2*i+1] + N3*U[2*i+2] + N4*U[2*i+3]
                        break
                res.append({'x': x, 'shear': v_sh, 'moment': m_bm, 'deflection': d_defl})
            return pd.DataFrame(res), pd.DataFrame(reac_res), {'l_fy': t_load_fy, 'r_fy': t_reac_fy, 'l_m0': t_load_m0, 'r_m0': t_reac_m0}
        except Exception as e: return pd.DataFrame(), pd.DataFrame(), {"error": str(e)}
