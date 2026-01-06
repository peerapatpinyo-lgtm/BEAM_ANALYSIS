import numpy as np
import pandas as pd
from scipy.linalg import solve

class BeamSolver:
    def __init__(self, spans, supports_input, loads_input, E, b=0.3, h=0.5, I_custom=None):
        self.spans = [float(s) for s in spans]
        self.E = float(E)
        self.b, self.h = b, h
        self.I = float(I_custom) if I_custom else (b * h**3) / 12
        # Timoshenko Constants
        self.G = self.E / (2 * (1 + 0.2)) 
        self.As = (5/6) * (b * h)        
        self.cum_spans = [round(x, 4) for x in ([0.0] + list(np.cumsum(self.spans)))]
        self.loads_df = pd.DataFrame(loads_input)
        self.supports_df = pd.DataFrame(supports_input)

    def _get_k_timoshenko(self, L):
        EI = self.E * self.I
        Phi = (12 * EI) / (L**2 * self.G * self.As)
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
            num_nodes = len(nodes)
            dof = 2 * num_nodes
            K, F = np.zeros((dof, dof)), np.zeros(dof)

            total_load_fy = 0.0
            total_load_moment_at_0 = 0.0

            for i in range(num_nodes - 1):
                L = nodes[i+1] - nodes[i]
                if L > 1e-5:
                    K[np.ix_([2*i, 2*i+1, 2*(i+1), 2*(i+1)+1], [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1])] += self._get_k_timoshenko(L)

            for _, l in self.loads_df.iterrows():
                gx = self.cum_spans[int(l['span_index'])] + float(l['x'])
                mag = float(l['mag'])
                if l['type'] == 'P':
                    nid = np.argmin([abs(n - gx) for n in nodes])
                    F[2*nid] -= mag
                    total_load_fy += mag
                    total_load_moment_at_0 += mag * gx
                elif l['type'] == 'M':
                    nid = np.argmin([abs(n - gx) for n in nodes])
                    F[2*nid+1] += mag
                    total_load_moment_at_0 -= mag
                elif l['type'] == 'U':
                    dist = float(l['dist'])
                    total_load_fy += mag * dist
                    total_load_moment_at_0 += (mag * dist) * (gx + dist/2)
                    for i in range(num_nodes - 1):
                        overlap = min(gx+dist, nodes[i+1]) - max(gx, nodes[i])
                        if overlap > 1e-5:
                            w, L_el = mag, nodes[i+1] - nodes[i]
                            F[2*i] -= (w * L_el / 2); F[2*i+1] -= (w * L_el**2 / 12)
                            F[2*(i+1)] -= (w * L_el / 2); F[2*(i+1)+1] += (w * L_el**2 / 12)

            free_dof = np.full(dof, True)
            for _, sup in self.supports_df.iterrows():
                if sup['type'] == "None": continue
                nid = np.argmin([abs(n - self.cum_spans[int(sup['id'])]) for n in nodes])
                if sup['type'] in ['Pin', 'Roller', 'Fixed']: free_dof[2*nid] = False
                if sup['type'] == 'Fixed': free_dof[2*nid+1] = False

            U = np.zeros(dof)
            U[free_dof] = solve(K[np.ix_(free_dof, free_dof)], F[free_dof])
            R_full = K @ U - F

            reac_list = []
            total_reac_fy, total_reac_moment_at_0 = 0.0, 0.0
            for _, sup in self.supports_df.iterrows():
                if sup['type'] == "None": continue
                node_idx = int(sup['id'])
                nid = np.argmin([abs(n - self.cum_spans[node_idx]) for n in nodes])
                total_reac_fy += R_full[2*nid]
                total_reac_moment_at_0 += (R_full[2*nid] * self.cum_spans[node_idx]) + R_full[2*nid+1]
                reac_list.append({'Node': node_idx, 'Type': sup['type'], 'Ry (kN)': round(R_full[2*nid]/1000, 2), 'M (kNm)': round(R_full[2*nid+1]/1000, 2)})

            res_data = []
            for x in np.linspace(0, nodes[-1], 400):
                V, M, defl = 0.0, 0.0, 0.0
                for i, n_p in enumerate(nodes):
                    if n_p <= x + 1e-5:
                        V += R_full[2*i]; M += R_full[2*i]*(x - n_p) + R_full[2*i+1]
                for _, l in self.loads_df.iterrows():
                    gx = self.cum_spans[int(l['span_index'])] + float(l['x'])
                    if l['type'] == 'P' and gx <= x + 1e-5: V -= l['mag']; M -= l['mag']*(x - gx)
                    elif l['type'] == 'M' and gx <= x + 1e-5: M -= l['mag']
                    elif l['type'] == 'U' and gx < x:
                        d = min(x, gx + l['dist']) - gx
                        V -= l['mag']*d; M -= l['mag']*d*(x - (gx + d/2))
                for i in range(num_nodes - 1):
                    if nodes[i] <= x <= nodes[i+1] + 1e-5:
                        L_el, s = nodes[i+1] - nodes[i], (x - nodes[i]) / (nodes[i+1] - nodes[i])
                        H = np.array([1-3*s**2+2*s**3, L_el*(s-2*s**2+s**3), 3*s**2-2*s**3, L_el*(s**3-s**2)])
                        defl = np.dot(H, U[2*i:2*i+4])
                        break
                res_data.append({'x': x, 'shear': V, 'moment': M, 'deflection': defl})

            return pd.DataFrame(res_data), pd.DataFrame(reac_list), {
                'sum_fy_load': total_load_fy, 'sum_fy_reac': total_reac_fy,
                'sum_m0_load': total_load_moment_at_0, 'sum_m0_reac': total_reac_moment_at_0
            }
        except Exception as e: return pd.DataFrame(), pd.DataFrame(), {"error": str(e)}
