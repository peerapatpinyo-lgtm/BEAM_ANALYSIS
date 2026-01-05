import numpy as np
import pandas as pd
from scipy.linalg import solve

class BeamSolver:
    def __init__(self, spans, supports_df, loads_input, E, I, A=None, G=None):
        self.spans = [float(s) for s in spans]
        self.E = float(E)
        self.I = float(I)
        self.A = float(A) if A is not None else 0.01
        self.G = float(G) if G is not None else 7.7e10
        
        # ปัดเศษทศนิยมเพื่อความแม่นยำในการเปรียบเทียบตำแหน่ง
        self.cum_spans = [round(x, 4) for x in ([0.0] + list(np.cumsum(self.spans)))]
        
        self.loads_df = self._sanitize_loads(loads_input)
        self.supports_df = self._sanitize_supports(supports_df)

    def _sanitize_loads(self, loads_input):
        if not loads_input:
            return pd.DataFrame(columns=['span_idx', 'type', 'mag', 'x', 'dist', 'case'])
        df = pd.DataFrame(loads_input)
        def get_global_x(row):
            s_idx = int(row.get('span_index', row.get('span_idx', 0)))
            local_x = float(row.get('x', 0))
            return round(self.cum_spans[s_idx] + local_x, 4)
        df['x'] = df.apply(get_global_x, axis=1)
        return df

    def _sanitize_supports(self, supports_input):
        if isinstance(supports_input, pd.DataFrame):
            data = supports_input.to_dict('records')
        else:
            data = supports_input
        sanitized = []
        for s in data:
            node_id = int(s.get('id', s.get('Node ID', 0)))
            if node_id < len(self.cum_spans):
                sanitized.append({
                    'x': self.cum_spans[node_id], # ใช้ค่าจาก cum_spans โดยตรงเพื่อให้เป๊ะ
                    'type': s.get('type', s.get('Support Type', 'None'))
                })
        return pd.DataFrame(sanitized)

    def _find_nearest_node(self, nodes, val):
        arr = np.array(nodes)
        diff = np.abs(arr - val)
        idx = diff.argmin()
        if diff[idx] < 1e-3: # เพิ่ม Tolerance ให้กว้างขึ้นเล็กน้อยเพื่อความปลอดภัย
            return idx
        return -1

    def solve(self):
        # 1. รวบรวมตำแหน่ง Nodes ทั้งหมด
        points = set([round(x, 4) for x in self.cum_spans])
        for _, l in self.loads_df.iterrows():
            points.add(round(l['x'], 4))
            if l['type'] == 'U': 
                points.add(round(l['x'] + l['dist'], 4))
        
        nodes = sorted(list(points))
        num_nodes = len(nodes)
        dof = 2 * num_nodes
        
        # 2. Stiffness Matrix K
        K = np.zeros((dof, dof))
        elements = []
        for i in range(num_nodes - 1):
            L = nodes[i+1] - nodes[i]
            if L > 1e-6:
                elements.append({'n1': i, 'n2': i+1, 'L': L})
                k_el = self._get_k(L)
                idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
                K[np.ix_(idx, idx)] += k_el

        # 3. Load Vector F (Equivalent Nodal Forces)
        F = np.zeros(dof)
        for _, load in self.loads_df.iterrows():
            lx, lmag = load['x'], load['mag']
            if load['type'] == 'P':
                nid = self._find_nearest_node(nodes, lx)
                if nid != -1: F[2*nid] -= lmag
            elif load['type'] == 'U':
                start, end = lx, lx + load['dist']
                for elem in elements:
                    ex1, ex2 = nodes[elem['n1']], nodes[elem['n2']]
                    overlap = min(end, ex2) - max(start, ex1)
                    if overlap > 1e-6:
                        F[2*elem['n1']] -= lmag * overlap * 0.5
                        F[2*elem['n2']] -= lmag * overlap * 0.5

        # 4. Boundary Conditions (จุดที่เคยพลาด)
        free_dof = np.full(dof, True)
        for _, sup in self.supports_df.iterrows():
            nid = self._find_nearest_node(nodes, sup['x'])
            if nid != -1 and sup['type'] != "None":
                if sup['type'] in ['Pin', 'Roller', 'Fixed']:
                    free_dof[2*nid] = False # ล็อคแกน Y
                if sup['type'] == 'Fixed':
                    free_dof[2*nid+1] = False # ล็อค Moment

        # 5. Solve U
        U = np.zeros(dof)
        if not np.all(free_dof):
            K_reduced = K[np.ix_(free_dof, free_dof)]
            F_reduced = F[free_dof]
            U[free_dof] = solve(K_reduced, F_reduced)

        # 6. Reactions
        R = K @ U - F

        # 7. Internal Forces (Statics Integration)
        results = []
        plot_x = np.unique(np.concatenate([np.linspace(0, nodes[-1], 300), nodes]))
        for x in plot_x:
            V, M, defl = 0.0, 0.0, 0.0
            # Calculation for Deflection ...
            for elem in elements:
                x1, x2 = nodes[elem['n1']], nodes[elem['n2']]
                if x1 <= x <= x2 + 1e-6:
                    s = (x - x1) / elem['L']
                    idx = [2*elem['n1'], 2*elem['n1']+1, 2*elem['n2'], 2*elem['n2']+1]
                    H = np.array([1-3*s**2+2*s**3, (x-x1)*(1-s)**2, 3*s**2-2*s**3, (x-x1)*(s**2-s)])
                    defl = np.dot(H, U[idx])
                    break
            
            # Sum forces from left to x
            for i, nx in enumerate(nodes):
                if nx <= x + 1e-4:
                    V += R[2*i]
                    M += R[2*i]*(x-nx) + R[2*i+1]
            for _, l in self.loads_df.iterrows():
                lx, lmag = l['x'], l['mag']
                if l['type'] == 'P' and lx <= x + 1e-4:
                    V -= lmag
                    M -= lmag*(x-lx)
                elif l['type'] == 'U' and lx < x:
                    dist = min(x, lx + l['dist']) - lx
                    if dist > 0:
                        V -= lmag * dist
                        M -= lmag * dist * (x - (lx + dist/2))
            
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
        return {
            'V_max': {'value': df['shear'].abs().max(), 'x': df.iloc[df['shear'].abs().idxmax()]['x']},
            'M_pos': {'value': df['moment'].max(), 'x': df.iloc[df['moment'].idxmax()]['x']},
            'M_neg': {'value': df['moment'].min(), 'x': df.iloc[df['moment'].idxmin()]['x']},
            'D_max': {'value': df['deflection'].abs().max(), 'x': df.iloc[df['deflection'].abs().idxmax()]['x']}
        }
