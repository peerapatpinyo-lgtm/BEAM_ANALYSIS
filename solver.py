import numpy as np
import pandas as pd
from scipy.linalg import solve

class BeamSolver:
    def __init__(self, spans, supports_input, loads_input, E, b=0.3, h=0.5, I_custom=None):
        self.spans = [float(s) for s in spans]
        self.E = float(E)
        self.b, self.h = b, h
        # 1. จัดการค่า I
        self.I = float(I_custom) if I_custom else (b * h**3) / 12
        # Timoshenko Constants: G = E / (2 * (1 + nu)), nu = 0.2, As = Shear Area Factor (5/6 for Rect)
        self.G = self.E / (2 * (1 + 0.2)) 
        self.As = (5/6) * (b * h)        
        self.cum_spans = [round(x, 4) for x in ([0.0] + list(np.cumsum(self.spans)))]
        self.loads_df = pd.DataFrame(loads_input)
        self.supports_df = pd.DataFrame(supports_input)

    def _get_k_timoshenko(self, L):
        EI = self.E * self.I
        # Phi (Shear Deformation Parameter)
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

            # Equation Check Accumulators
            total_load_fy, total_load_m0 = 0.0, 0.0

            # Stiffness Matrix Assembly
            for i in range(num_nodes - 1):
                L = nodes[i+1] - nodes[i]
                if L > 1e-5:
                    idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
                    K[np.ix_(idx, idx)] += self._get_k_timoshenko(L)

            # --- 2. Load Application & Equation Check Logic ---
            for _, l in self.loads_df.iterrows():
                gx = self.cum_spans[int(l['span_index'])] + float(l['x'])
                mag = float(l['mag'])
                if l['type'] == 'P':
                    nid = np.argmin([abs(n - gx) for n in nodes])
                    F[2*nid] -= mag
                    total_load_fy += mag
                    total_load_m0 += mag * gx
                elif l['type'] == 'M':
                    nid = np.argmin([abs(n - gx) for n in nodes])
                    F[2*nid+1] += mag
                    total_load_m0 -= mag 
                elif l['type'] == 'U':
                    dist = float(l['dist'])
                    s_g, e_g = gx, round(gx + dist, 4)
                    total_load_fy += mag * dist
                    total_load_m0 += (mag * dist) * (gx + dist/2)
                    for i in range(num_nodes - 1):
                        overlap = min(e_g, nodes[i+1]) - max(s_g, nodes[i])
                        if overlap > 1e-5:
                            w, L_el = mag, nodes[i+1] - nodes[i]
                            F[2*i] -= (w * L_el / 2); F[2*i+1] -= (w * L_el**2 / 12)
                            F[2*(i+1)] -= (w * L_el / 2); F[2*(i+1)+1] += (w * L_el**2 / 12)

            # Boundary Conditions
            free_dof = np.full(dof, True)
            for _, sup in self.supports_df.iterrows():
                if sup['type'] == "None": continue
                nid = np.argmin([abs(n - self.cum_spans[int(sup['id'])]) for n in nodes])
                if sup['type'] in ['Pin', 'Roller', 'Fixed']: free_dof[2*nid] = False
                if sup['type'] == 'Fixed': free_dof[2*nid+1] = False

            U = np.zeros(dof)
            U[free_dof] = solve(K[np.ix_(free_dof, free_dof)], F[free_dof])
            R_full = K @ U - F

            # --- 3. Reaction & Statics Verification ---
            reac_list = []
            total_reac_fy, total_reac_m0 = 0.0, 0.0
            for _, sup in self.supports_df.iterrows():
                if sup['type'] == "None": continue
                node_idx = int(sup['id'])
                nid = np.argmin([abs(n - self.cum_spans[node_idx]) for n in nodes])
                ry, rm = R_full[2*nid], R_full[2*nid+1]
                total_reac_fy += ry
                total_reac_m0 += (ry * self.cum_spans[node_idx]) + rm
                reac_list.append({'Node': node_idx, 'Type': sup['type'], 'Ry (kN)': round(ry/1000, 2), 'M (kNm)': round(rm/1000, 2)})

            # Results Sampling
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
                'load_fy': total_load_fy, 'reac_fy': total_reac_fy,
                'load_m0': total_load_m0, 'reac_m0': total_reac_m0
            }
        except Exception as e:
            return pd.DataFrame(), pd.DataFrame(), {"error": str(e)}
