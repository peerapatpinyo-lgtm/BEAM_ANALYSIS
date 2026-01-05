import numpy as np
import pandas as pd
from scipy.linalg import solve

class BeamSolver:
    def __init__(self, spans, supports_input, loads_input, E, I, A=None, G=None):
        self.spans = [float(s) for s in spans]
        self.E = float(E)
        self.I = float(I)
        self.A = float(A) if A is not None else 0.01
        self.G = float(G) if G is not None else 7.7e10
        
        # 1. พิกัด Global x (ใช้ทศนิยม 4 ตำแหน่ง)
        self.cum_spans = [round(x, 4) for x in ([0.0] + list(np.cumsum(self.spans)))]
        
        # 2. จัดการข้อมูลเบื้องต้น
        self.supports_df = self._sanitize_supports(supports_input)
        self.loads_df = self._sanitize_loads(loads_input)

    def _sanitize_supports(self, supports_input):
        sanitized = []
        data = supports_input.to_dict('records') if isinstance(supports_input, pd.DataFrame) else supports_input
        for s in data:
            raw_id = s.get('id', s.get('Node ID'))
            if raw_id is None: continue
            idx = int(raw_id)
            if 'Node ID' in s: idx -= 1 # ปรับจาก 1-based เป็น 0-based
            stype = s.get('type', s.get('Support Type', 'None'))
            if 0 <= idx < len(self.cum_spans):
                sanitized.append({'x': self.cum_spans[idx], 'type': str(stype)})
        return pd.DataFrame(sanitized)

    def _sanitize_loads(self, loads_input):
        if not loads_input: return pd.DataFrame(columns=['span_idx', 'type', 'mag', 'x', 'dist', 'case'])
        df = pd.DataFrame(loads_input)
        def get_global_x(row):
            s_idx = int(row.get('span_index', row.get('span_idx', 0)))
            local_x = float(row.get('x', 0))
            return round(self.cum_spans[s_idx] + local_x, 4)
        df['x'] = df.apply(get_global_x, axis=1)
        return df

    def solve(self):
        # --- กุญแจสำคัญ: สร้าง Node List ที่สะอาด ---
        pts = self.cum_spans.copy()
        for _, l in self.loads_df.iterrows():
            pts.append(l['x'])
            if l['type'] == 'U': pts.append(round(l['x'] + l['dist'], 4))
        
        # ลบพิกัดที่ซ้ำกันออกโดยใช้ tolerance
        nodes = []
        for p in sorted(pts):
            if not any(np.isclose(p, n, atol=1e-5) for n in nodes):
                nodes.append(p)
        
        num_nodes = len(nodes)
        dof = 2 * num_nodes
        K = np.zeros((dof, dof))
        F = np.zeros(dof)

        # 1. Stiffness Matrix K
        for i in range(num_nodes - 1):
            L = nodes[i+1] - nodes[i]
            if L > 1e-6:
                k_el = self._get_k(L)
                idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
                K[np.ix_(idx, idx)] += k_el

        # 2. Force Vector F (Point Load & Moments)
        for _, l in self.loads_df.iterrows():
            if l['type'] == 'P' or l['type'] == 'M':
                # หาโหนดที่ใกล้ที่สุดแบบเป๊ะๆ
                nid = np.argmin([abs(n - l['x']) for n in nodes])
                if l['type'] == 'P': F[2*nid] -= l['mag']
                else: F[2*nid+1] += l['mag']
            elif l['type'] == 'U':
                # Uniform load distribution
                s_glob, e_glob = l['x'], round(l['x'] + l['dist'], 4)
                for i in range(num_nodes - 1):
                    n1, n2 = nodes[i], nodes[i+1]
                    overlap = min(e_glob, n2) - max(s_glob, n1)
                    if overlap > 1e-6:
                        # กระจายแรงลงโหนด (Simple nodal resultant)
                        F[2*i] -= l['mag'] * overlap * 0.5
                        F[2*(i+1)] -= l['mag'] * overlap * 0.5

        # 3. Boundary Conditions
        free_dof = np.full(dof, True)
        for _, sup in self.supports_df.iterrows():
            nid = np.argmin([abs(n - sup['x']) for n in nodes])
            if sup['type'] in ['Pin', 'Roller', 'Fixed']: free_dof[2*nid] = False
            if sup['type'] == 'Fixed': free_dof[2*nid+1] = False

        # 4. Solve
        U = np.zeros(dof)
        K_sub = K[np.ix_(free_dof, free_dof)]
        if K_sub.size > 0:
            U[free_dof] = solve(K_sub, F[free_dof])

        # 5. Reactions (หัวใจของการเช็ค Equilibrium)
        R = K @ U - F

        # 6. Post-processing (คำนวณกราฟ)
        results = []
        plot_x = np.unique(np.concatenate([np.linspace(0, nodes[-1], 400), nodes]))
        for x in plot_x:
            V, M, defl = 0.0, 0.0, 0.0
            # Shear & Moment จากสมดุลแรง (Statics)
            for i, n_pos in enumerate(nodes):
                if n_pos <= x + 1e-5:
                    V += R[2*i]
                    M += R[2*i]*(x - n_pos) + R[2*i+1]
            for _, l in self.loads_df.iterrows():
                lx, lmag = l['x'], l['mag']
                if l['type'] == 'P' and lx <= x + 1e-5:
                    V -= lmag
                    M -= lmag*(x - lx)
                elif l['type'] == 'M' and lx <= x + 1e-5:
                    M -= lmag
                elif l['type'] == 'U' and lx < x:
                    dist = min(x, lx + l['dist']) - lx
                    if dist > 0:
                        V -= lmag * dist
                        M -= lmag * dist * (x - (lx + dist/2))
            
            # Deflection
            for i in range(num_nodes - 1):
                x1, x2 = nodes[i], nodes[i+1]
                if x1 <= x <= x2 + 1e-6:
                    L_el = x2 - x1
                    s = (x - x1) / L_el
                    idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
                    H = np.array([1-3*s**2+2*s**3, (x-x1)*(1-s)**2, 3*s**2-2*s**3, (x-x1)*(s**2-s)])
                    defl = np.dot(H, U[idx])
                    break
            results.append({'x': x, 'deflection': defl, 'shear': V, 'moment': M})

        df_res = pd.DataFrame(results)
        return df_res, R, self._create_summary(df_res)

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
