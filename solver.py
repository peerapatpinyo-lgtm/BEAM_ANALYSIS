import numpy as np
import pandas as pd
from scipy.linalg import solve

class BeamSolver:
    def __init__(self, spans, supports_input, loads_input, E, I, A=None, G=None):
        self.spans = [float(s) for s in spans]
        self.E = float(E)
        self.I = float(I)
        # 1. พิกัดสะสมของปลาย Span (ตำแหน่ง Support ที่ควรจะเป็น)
        self.cum_spans = [round(x, 4) for x in ([0.0] + list(np.cumsum(self.spans)))]
        
        # 2. แปลง Load และ Support เป็น Global Coordinate
        self.loads_df = self._sanitize_loads(loads_input)
        self.supports_df = self._sanitize_supports(supports_input)

    def _sanitize_supports(self, supports_input):
        sanitized = []
        data = supports_input.to_dict('records') if hasattr(supports_input, 'to_dict') else supports_input
        for s in data:
            stype = str(s.get('type', s.get('Support Type', 'None')))
            if stype == "None": continue
            
            raw_id = s.get('id', s.get('Node ID'))
            try:
                idx = int(raw_id)
                if 'Node ID' in s: idx -= 1 # แปลง 1-based เป็น 0-based
                if 0 <= idx < len(self.cum_spans):
                    sanitized.append({'x': self.cum_spans[idx], 'type': stype})
            except: continue
        return pd.DataFrame(sanitized)

    def _sanitize_loads(self, loads_input):
        if not loads_input: return pd.DataFrame(columns=['span_idx', 'type', 'mag', 'x', 'dist'])
        df = pd.DataFrame(loads_input)
        def get_global_x(row):
            s_idx = int(row.get('span_index', row.get('span_idx', 0)))
            lx = float(row.get('x', 0))
            return round(self.cum_spans[s_idx] + lx, 4)
        df['x'] = df.apply(get_global_x, axis=1)
        return df

    def solve(self):
        # --- [STEP 1: สร้างโหนดทั้งหมด] ---
        pts = self.cum_spans.copy()
        for _, l in self.loads_df.iterrows():
            pts.append(l['x'])
            if l['type'] == 'U': pts.append(round(l['x'] + l['dist'], 4))
        
        nodes = []
        for p in sorted(pts):
            if not any(abs(p - n) < 1e-4 for n in nodes):
                nodes.append(p)
        
        num_nodes = len(nodes)
        dof = 2 * num_nodes
        K = np.zeros((dof, dof))
        F = np.zeros(dof)

        # --- [STEP 2: Stiffness Matrix] ---
        for i in range(num_nodes - 1):
            L = nodes[i+1] - nodes[i]
            if L > 1e-5:
                k_el = self._get_k(L)
                idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
                K[np.ix_(idx, idx)] += k_el

        # --- [STEP 3: ใส่ Load - แก้ไขตรงนี้เพื่อ Point Load] ---
        for _, l in self.loads_df.iterrows():
            if l['type'] in ['P', 'M']:
                # หาโหนดที่ตำแหน่ง x ตรงกับโหลดเป๊ะๆ
                nid = np.argmin([abs(n - l['x']) for n in nodes])
                if l['type'] == 'P': F[2*nid] -= l['mag']
                else: F[2*nid+1] += l['mag']
            elif l['type'] == 'U':
                # สำหรับ UDL ต้องเช็คทุก Element ที่โหลดพาดผ่าน
                s_g, e_g = l['x'], round(l['x'] + l['dist'], 4)
                for i in range(num_nodes - 1):
                    n1, n2 = nodes[i], nodes[i+1]
                    overlap = min(e_g, n2) - max(s_g, n1)
                    if overlap > 1e-5:
                        F[2*i] -= l['mag'] * overlap * 0.5
                        F[2*(i+1)] -= l['mag'] * overlap * 0.5

        # --- [STEP 4: Boundary Conditions - ใช้พิกัด X เท่านั้น] ---
        free_dof = np.full(dof, True)
        for _, sup in self.supports_df.iterrows():
            # ค้นหาโหนดจากพิกัด X เพื่อให้ล็อคถูกที่แม้จะมีโหนด Point Load มาคั่น
            nid = np.argmin([abs(n - sup['x']) for n in nodes])
            if abs(nodes[nid] - sup['x']) < 1e-4:
                if sup['type'] in ['Pin', 'Roller', 'Fixed']:
                    free_dof[2*nid] = False
                if sup['type'] == 'Fixed':
                    free_dof[2*nid+1] = False

        # --- [STEP 5: Solve & Reactions] ---
        U = np.zeros(dof)
        if not np.all(free_dof):
            U[free_dof] = solve(K[np.ix_(free_dof, free_dof)], F[free_dof])

        R = K @ U - F # แรงปฏิกิริยาที่โหนดทั้งหมด

        # --- [STEP 6: Statics Check & Results] ---
        results = []
        plot_x = np.unique(np.sort(np.concatenate([np.linspace(0, nodes[-1], 300), nodes])))
        for x in plot_x:
            V, M, defl = 0.0, 0.0, 0.0
            # ตัด Section จากซ้าย (Summation of forces)
            for i, n_p in enumerate(nodes):
                if n_p <= x + 1e-5:
                    V += R[2*i]
                    M += R[2*i]*(x - n_p) + R[2*i+1]
            for _, l in self.loads_df.iterrows():
                if l['type'] == 'P' and l['x'] <= x + 1e-5:
                    V -= l['mag']; M -= l['mag']*(x - l['x'])
                elif l['type'] == 'M' and l['x'] <= x + 1e-5:
                    M -= l['mag']
                elif l['type'] == 'U' and l['x'] < x:
                    d = min(x, l['x'] + l['dist']) - l['x']
                    if d > 0:
                        V -= l['mag']*d; M -= l['mag']*d*(x - (l['x'] + d/2))
            
            # Find element for Deflection
            for i in range(num_nodes - 1):
                if nodes[i] <= x <= nodes[i+1] + 1e-5:
                    s = (x - nodes[i]) / (nodes[i+1] - nodes[i])
                    H = np.array([1-3*s**2+2*s**3, (x-nodes[i])*(1-s)**2, 3*s**2-2*s**3, (x-nodes[i])*(s**2-s)])
                    defl = np.dot(H, U[2*i:2*i+4])
                    break
            results.append({'x': x, 'deflection': defl, 'shear': V, 'moment': M})

        return pd.DataFrame(results), R, self._create_summary(pd.DataFrame(results))

    def _get_k(self, L):
        EI = self.E * self.I
        return (EI / L**3) * np.array([
            [12, 6*L, -12, 6*L], [6*L, 4*L**2, -6*L, 2*L**2],
            [-12, -6*L, 12, -6*L], [6*L, 2*L**2, -6*L, 4*L**2]
        ])

    def _create_summary(self, df):
        if df.empty: return {}
        return {
            'V_max': {'value': df['shear'].abs().max(), 'x': df.iloc[df['shear'].abs().idxmax()]['x']},
            'M_pos': {'value': df['moment'].max(), 'x': df.iloc[df['moment'].idxmax()]['x']},
            'M_neg': {'value': df['moment'].min(), 'x': df.iloc[df['moment'].idxmin()]['x']},
            'D_max': {'value': df['deflection'].abs().max(), 'x': df.iloc[df['deflection'].abs().idxmax()]['x']}
        }
