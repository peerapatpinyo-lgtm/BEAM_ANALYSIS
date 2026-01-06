import numpy as np
import pandas as pd
from scipy.linalg import solve

class Section:
    @staticmethod
    def rectangular(b, h):
        return (b * h**3) / 12, b * h
    
    @staticmethod
    def t_beam(bf, hf, bw, h):
        # BF=Flange Width, HF=Flange Thick, BW=Web Width, H=Total Height
        area = (bf * hf) + (bw * (h - hf))
        y_bar = ((bf * hf * (h - hf/2)) + (bw * (h - hf) * (h - hf)/2)) / area
        i_na = (bf * hf**3)/12 + (bf * hf * (h - hf/2 - y_bar)**2) + \
               (bw * (h - hf)**3)/12 + (bw * (h - hf) * (y_bar - (h-hf)/2)**2)
        return i_na, area

class BeamSolver:
    def __init__(self, spans, supports_input, loads_input, E, b, h, section_type="Rectangular", bf=None, hf=None):
        self.spans = [float(s) for s in spans]
        self.E = float(E)
        self.b, self.h = b, h
        
        # เลือกวิธีการคำนวณ I (ฟีเจอร์ที่เคยมี)
        if section_type == "T-Beam":
            self.I, self.A = Section.t_beam(bf, hf, b, h)
        else:
            self.I, self.A = Section.rectangular(b, h)
            
        self.cum_spans = [round(x, 4) for x in ([0.0] + list(np.cumsum(self.spans)))]
        self.loads_df = self._sanitize_loads(loads_input)
        self.supports_df = self._sanitize_supports(supports_input)

    # --- (ส่วน _sanitize_supports และ _sanitize_loads เหมือนเดิมกับ Full Code ก่อนหน้า) ---
    def _sanitize_supports(self, supports_input):
        sanitized = []
        data = supports_input.to_dict('records') if hasattr(supports_input, 'to_dict') else supports_input
        for s in data:
            stype = str(s.get('type', s.get('Support Type', 'None')))
            if stype == "None": continue
            try:
                idx = int(s.get('id', s.get('Node ID', 1))) - (1 if 'Node ID' in s else 0)
                if 0 <= idx < len(self.cum_spans):
                    sanitized.append({'x': self.cum_spans[idx], 'type': stype})
            except: continue
        return pd.DataFrame(sanitized)

    def _sanitize_loads(self, loads_input):
        if not loads_input: return pd.DataFrame(columns=['span_index', 'type', 'mag', 'x', 'dist'])
        df = pd.DataFrame(loads_input)
        def get_global_x(row):
            s_idx = int(row.get('span_index', 0))
            lx = float(row.get('x', 0))
            return round(self.cum_spans[s_idx] + lx, 4)
        df['global_x'] = df.apply(get_global_x, axis=1)
        return df

    def solve(self):
        # 1. สร้าง Node (Support + Load points)
        pts = self.cum_spans.copy()
        for _, l in self.loads_df.iterrows():
            pts.append(l['global_x'])
            if l['type'] == 'U': pts.append(round(l['global_x'] + l['dist'], 4))
        
        nodes = []
        for p in sorted(pts):
            if not any(abs(p - n) < 1e-4 for n in nodes): nodes.append(p)
        
        num_nodes = len(nodes)
        dof = 2 * num_nodes
        K, F = np.zeros((dof, dof)), np.zeros(dof)

        # 2. Stiffness Matrix Assembly (Prismatic Beam)
        for i in range(num_nodes - 1):
            L = nodes[i+1] - nodes[i]
            if L > 1e-5:
                k_el = self._get_k(L, self.I) # ส่งค่า I เข้าไปคำนวณ
                idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
                K[np.ix_(idx, idx)] += k_el

        # 3. Load Vector Assembly
        for _, l in self.loads_df.iterrows():
            nid = np.argmin([abs(n - l['global_x']) for n in nodes])
            if l['type'] == 'P': F[2*nid] -= l['mag']
            elif l['type'] == 'M': F[2*nid+1] += l['mag']
            elif l['type'] == 'U':
                s_g, e_g = l['global_x'], round(l['global_x'] + l['dist'], 4)
                for i in range(num_nodes - 1):
                    n1, n2 = nodes[i], nodes[i+1]
                    overlap = min(e_g, n2) - max(s_g, n1)
                    if overlap > 1e-5:
                        mid = (max(s_g, n1) + min(e_g, n2)) / 2
                        force = l['mag'] * overlap
                        F[2*i] -= force * (n2 - mid) / (n2 - n1)
                        F[2*(i+1)] -= force * (mid - n1) / (n2 - n1)

        # 4. Boundary Conditions
        free_dof = np.full(dof, True)
        for _, sup in self.supports_df.iterrows():
            nid = np.argmin([abs(n - sup['x']) for n in nodes])
            if sup['type'] in ['Pin', 'Roller', 'Fixed']: free_dof[2*nid] = False
            if sup['type'] == 'Fixed': free_dof[2*nid+1] = False

        # 5. Solve
        U = np.zeros(dof)
        if not np.all(free_dof):
            U[free_dof] = solve(K[np.ix_(free_dof, free_dof)], F[free_dof])

        R_full = K @ U - F
        r_mapped = np.zeros(2 * (len(self.spans) + 1))
        for i, tx in enumerate(self.cum_spans):
            nid = np.argmin([abs(n - tx) for n in nodes])
            r_mapped[2*i], r_mapped[2*i+1] = R_full[2*nid], R_full[2*nid+1]

        # 6. Internal Forces & Deflection (Internal calculation)
        # (ส่วนนี้เหมือนกับโค้ดล่าสุด เพื่อรักษาความถูกต้องของกราฟ)
        results = []
        plot_x = np.unique(np.sort(np.concatenate([np.linspace(0, nodes[-1], 350), nodes])))
        for x in plot_x:
            V, M, defl = 0.0, 0.0, 0.0
            for i, np_pt in enumerate(nodes):
                if np_pt <= x + 1e-5:
                    V += R_full[2*i]
                    M += R_full[2*i]*(x - np_pt) + R_full[2*i+1]
            for _, l in self.loads_df.iterrows():
                if l['type'] == 'P' and l['global_x'] <= x + 1e-5:
                    V -= l['mag']; M -= l['mag']*(x - l['global_x'])
                elif l['type'] == 'M' and l['global_x'] <= x + 1e-5:
                    M -= l['mag']
                elif l['type'] == 'U' and l['global_x'] < x:
                    d = min(x, l['global_x'] + l['dist']) - l['global_x']
                    if d > 0: V -= l['mag']*d; M -= l['mag']*d*(x - (l['global_x'] + d/2))
            for i in range(num_nodes - 1):
                if nodes[i] <= x <= nodes[i+1] + 1e-5:
                    s = (x - nodes[i]) / (nodes[i+1] - nodes[i])
                    H = np.array([1-3*s**2+2*s**3, (x-nodes[i])*(1-s)**2, 3*s**2-2*s**3, (x-nodes[i])*(s**2-s)])
                    defl = np.dot(H, U[2*i:2*i+4])
                    break
            results.append({'x': x, 'deflection': defl, 'shear': V, 'moment': M})

        df_res = pd.DataFrame(results)
        return df_res, r_mapped, self._create_summary(df_res)

    def _get_k(self, L, I_val):
        EI = self.E * I_val
        return (EI / L**3) * np.array([[12, 6*L, -12, 6*L], [6*L, 4*L**2, -6*L, 2*L**2], [-12, -6*L, 12, -6*L], [6*L, 2*L**2, -6*L, 4*L**2]])

    def _create_summary(self, df):
        return {
            'V_max': {'value': df['shear'].abs().max(), 'x': df.iloc[df['shear'].abs().idxmax()]['x']},
            'M_pos': {'value': df['moment'].max(), 'x': df.iloc[df['moment'].idxmax()]['x']},
            'M_neg': {'value': df['moment'].min(), 'x': df.iloc[df['moment'].idxmin()]['x']},
            'D_max': {'value': df['deflection'].abs().max(), 'x': df.iloc[df['deflection'].abs().idxmax()]['x']}
        }

    def pro_design(self, fc_mpa, fy_mpa, bar_dia_mm):
        # (Logic การจัดเหล็ก และ As_min เหมือนโค้ดก่อนหน้า)
        # ...
        return d_res # ผลลัพธ์ As และจำนวนเส้นเหล็ก
