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
            pts = self.cum_spans.copy()
            for _, l in self.loads_df.iterrows():
                gx = self.cum_spans[int(l['span_index'])] + float(l['x'])
                pts.append(gx)
                if l['type'] == 'U': pts.append(round(gx + float(l['dist']), 4))
            nodes = sorted(list(set([round(p, 4) for p in pts])))
            num_nodes = len(nodes)
            dof = 2 * num_nodes
            K, F = np.zeros((dof, dof)), np.zeros(dof)

            # --- Equation Check Setup: คำนวณ Load จริงเพื่อเช็คสมดุล ---
            total_load_fy = 0.0
            total_load_moment_at_0 = 0.0

            for i in range(num_nodes - 1):
                L = nodes[i+1] - nodes[i]
                if L > 1e-5:
                    EI = self.E * self.I
                    k_el = (EI / L**3) * np.array([[12, 6*L, -12, 6*L], [6*L, 4*L**2, -6*L, 2*L**2], [-12, -6*L, 12, -6*L], [6*L, 2*L**2, -6*L, 4*L**2]])
                    K[np.ix_([2*i, 2*i+1, 2*(i+1), 2*(i+1)+1], [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1])] += k_el

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
                    total_load_moment_at_0 -= mag # Moment direct sum
                elif l['type'] == 'U':
                    dist = float(l['dist'])
                    s_g, e_g = gx, round(gx + dist, 4)
                    total_load_fy += mag * dist
                    total_load_moment_at_0 += (mag * dist) * (gx + dist/2)
                    for i in range(num_nodes - 1):
                        overlap = min(e_g, nodes[i+1]) - max(s_g, nodes[i])
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

            # --- Reaction & Equation Check Calculation ---
            reac_list = []
            total_reac_fy = 0.0
            total_reac_moment_at_0 = 0.0
            for _, sup in self.supports_df.iterrows():
                if sup['type'] == "None": continue
                nid = np.argmin([abs(n - self.cum_spans[int(sup['id'])]) for n in nodes])
                ry = R_full[2*nid]
                rm = R_full[2*nid+1]
                reac_list.append({'Node': int(sup['id']), 'Type': sup['type'], 'Ry (kN)': round(ry/1000, 2), 'M (kNm)': round(rm/1000, 2)})
                total_reac_fy += ry
                total_reac_moment_at_0 += (ry * self.cum_spans[int(sup['id'])]) + rm

            res_df = [] # (ส่วนการคำนวณกราฟเหมือนเดิม...)
            for x in np.linspace(0, nodes[-1], 400):
                # ... [Code logic สำหรับ Shear/Moment Diagram] ...
                pass 

            # ส่งค่า Equation Check กลับไปแสดงผล
            eq_check = {
                'sum_fy_load': total_load_fy,
                'sum_fy_reac': total_reac_fy,
                'sum_m0_load': total_load_moment_at_0,
                'sum_m0_reac': total_reac_moment_at_0,
                'is_balanced': abs(total_load_fy - total_reac_fy) < 1e-3
            }
            return pd.DataFrame(res_df), pd.DataFrame(reac_list), eq_check
        except Exception as e: return pd.DataFrame(), pd.DataFrame(), {"error": str(e)}
