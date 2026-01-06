import numpy as np
import pandas as pd
from scipy.linalg import solve

class BeamSolver:
    def __init__(self, spans, supports_input, loads_input, E, b=0.3, h=0.5, I_custom=None):
        self.spans = [float(s) for s in spans]
        self.E = float(E)
        self.b = b
        self.h = h
        self.I = float(I_custom) if I_custom else (b * h**3) / 12
        self.cum_spans = [round(x, 4) for x in ([0.0] + list(np.cumsum(self.spans)))]
        self.loads_df = pd.DataFrame(loads_input)
        self.supports_df = pd.DataFrame(supports_input)

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

            # Assemble Global Stiffness Matrix
            for i in range(num_nodes - 1):
                L = nodes[i+1] - nodes[i]
                if L > 1e-5:
                    EI = self.E * self.I
                    k_el = (EI / L**3) * np.array([
                        [12, 6*L, -12, 6*L], [6*L, 4*L**2, -6*L, 2*L**2],
                        [-12, -6*L, 12, -6*L], [6*L, 2*L**2, -6*L, 4*L**2]
                    ])
                    idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
                    K[np.ix_(idx, idx)] += k_el

            # Applied Loads (P, U, M) & Total Load for Eq Check
            total_applied_force = 0.0
            for _, l in self.loads_df.iterrows():
                gx = self.cum_spans[int(l['span_index'])] + float(l['x'])
                if l['type'] == 'P':
                    nid = np.argmin([abs(n - gx) for n in nodes])
                    F[2*nid] -= float(l['mag']); total_applied_force += float(l['mag'])
                elif l['type'] == 'M':
                    nid = np.argmin([abs(n - gx) for n in nodes])
                    F[2*nid+1] += float(l['mag'])
                elif l['type'] == 'U':
                    s_g, e_g = gx, round(gx + float(l['dist']), 4)
                    for i in range(num_nodes - 1):
                        overlap = min(e_g, nodes[i+1]) - max(s_g, nodes[i])
                        if overlap > 1e-5:
                            w, L_el = float(l['mag']), nodes[i+1] - nodes[i]
                            F[2*i] -= (w * L_el / 2); F[2*i+1] -= (w * L_el**2 / 12)
                            F[2*(i+1)] -= (w * L_el / 2); F[2*(i+1)+1] += (w * L_el**2 / 12)
                            total_applied_force += (w * L_el)

            # Apply Boundary Conditions
            free_dof = np.full(dof, True)
            for _, sup in self.supports_df.iterrows():
                if sup['type'] == "None": continue
                nid = np.argmin([abs(n - self.cum_spans[int(sup['id'])]) for n in nodes])
                if sup['type'] in ['Pin', 'Roller', 'Fixed']: free_dof[2*nid] = False
                if sup['type'] == 'Fixed': free_dof[2*nid+1] = False

            U = np.zeros(dof)
            U[free_dof] = solve(K[np.ix_(free_dof, free_dof)], F[free_dof])
            R_full = K @ U - F

            # Reactions Table Data
            reac_list = []
            total_rx_force = 0.0
            for _, sup in self.supports_df.iterrows():
                if sup['type'] == "None": continue
                nid = np.argmin([abs(n - self.cum_spans[int(sup['id'])]) for n in nodes])
                reac_list.append({
                    'Node': int(sup['id']),
                    'Type': sup['type'],
                    'Vertical Reaction (kN)': round(R_full[2*nid]/1000, 2),
                    'Moment Reaction (kNm)': round(R_full[2*nid+1]/1000, 2)
                })
                total_rx_force += R_full[2*nid]

            # Results sampling for Diagrams
            res = []
            for x in np.linspace(0, nodes[-1], 400):
                V, M = 0.0, 0.0
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
                res.append({'x': x, 'shear': V, 'moment': M, 'deflection': 0.0})

            return pd.DataFrame(res), pd.DataFrame(reac_list), {
                'total_load': total_applied_force,
                'total_reac': total_rx_force,
                'error': abs(total_applied_force - total_rx_force)
            }
        except Exception as e:
            return pd.DataFrame(), pd.DataFrame(), {"error": str(e)}
