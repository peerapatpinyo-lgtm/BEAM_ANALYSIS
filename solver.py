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
        
        # ปัดเศษตำแหน่งเพื่อป้องกันปัญหา Precision ตอนสร้าง Node
        self.cum_spans = [round(x, 6) for x in ([0.0] + list(np.cumsum(self.spans)))]
        
        # เพิ่มฟังก์ชันที่ขาดหายไปจากการรัน
        self.loads_df = self._sanitize_loads(loads_input)
        self.supports_df = self._sanitize_supports(supports_df)

    def _sanitize_loads(self, loads_input):
        """แปลงรายการ Load ให้อยู่ในรูป DataFrame และ Global Coordinates"""
        if not loads_input:
            return pd.DataFrame(columns=['span_idx', 'type', 'mag', 'x', 'dist', 'case'])
        
        df = pd.DataFrame(loads_input)
        # คำนวณ x ให้เป็นตำแหน่ง Global ของคานทั้งหมด
        def get_global_x(row):
            s_idx = int(row.get('span_index', row.get('span_idx', 0)))
            local_x = float(row.get('x', 0))
            if s_idx < len(self.cum_spans):
                return round(self.cum_spans[s_idx] + local_x, 6)
            return local_x
            
        df['x'] = df.apply(get_global_x, axis=1)
        return df

    def _sanitize_supports(self, supports_input):
        """แปลงข้อมูล Support และคำนวณตำแหน่ง Global x"""
        if isinstance(supports_input, pd.DataFrame):
            data = supports_input.to_dict('records')
        else:
            data = supports_input
            
        sanitized = []
        for s in data:
            node_id = int(s.get('id', s.get('Node ID', 0)))
            if node_id < len(self.cum_spans):
                sanitized.append({
                    'x': self.cum_spans[node_id],
                    'type': s.get('type', s.get('Support Type', 'None'))
                })
        return pd.DataFrame(sanitized)

    def _find_nearest_node(self, nodes, val):
        arr = np.array(nodes)
        diff = np.abs(arr - val)
        idx = diff.argmin()
        if diff[idx] < 1e-5: # Tolerance สำหรับ Snap Load เข้า Node
            return idx
        return -1

    def solve(self):
        # 1. รวบรวมตำแหน่งที่ต้องสร้าง Nodes (Supports, Loads, Span Ends)
        points = set([round(x, 6) for x in self.cum_spans])
        for _, l in self.loads_df.iterrows():
            points.add(round(l['x'], 6))
            if l['type'] == 'U': 
                points.add(round(l['x'] + l['dist'], 6))
        for _, s in self.supports_df.iterrows():
            points.add(round(s['x'], 6))
            
        nodes = sorted(list(points))
        num_nodes = len(nodes)
        dof = 2 * num_nodes
        
        # 2. Build Global Stiffness Matrix K
        K = np.zeros((dof, dof))
        elements = []
        for i in range(num_nodes - 1):
            L = nodes[i+1] - nodes[i]
            if L > 1e-8:
                elements.append({'n1': i, 'n2': i+1, 'L': L})
                k_el = self._get_k(L)
                idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
                for r in range(4):
                    for c in range(4):
                        K[idx[r], idx[c]] += k_el[r, c]

        # 3. Build Global Force Vector F
        F = np.zeros(dof)
        for _, load in self.loads_df.iterrows():
            nid = self._find_nearest_node(nodes, load['x'])
            if load['type'] == 'P' and nid != -1:
                F[2*nid] -= load['mag']
            elif load['type'] == 'M' and nid != -1:
                F[2*nid+1] += load['mag']
            elif load['type'] == 'U':
                start, dist, mag = load['x'], load['dist'], load['mag']
                end = start + dist
                for elem in elements:
                    ex1, ex2 = nodes[elem['n1']], nodes[elem['n2']]
                    overlap_s = max(start, ex1)
                    overlap_e = min(end, ex2)
                    len_load = overlap_e - overlap_s
                    if len_load > 1e-8:
                        # Equivalent Nodal Forces using mid-point (Gauss-like)
                        mid = (overlap_s + overlap_e)/2
                        # Simpson's or Exact could be used, here is a simplified nodal distribution
                        s_mid = (mid - ex1) / elem['L']
                        F[2*elem['n1']] -= mag * len_load * 0.5 # Simplified
                        F[2*elem['n2']] -= mag * len_load * 0.5

        # 4. Apply Boundary Conditions
        free_dof = np.full(dof, True)
        for _, sup in self.supports_df.iterrows():
            nid = self._find_nearest_node(nodes, sup['x'])
            if nid != -1:
                stype = sup.get('type', 'None')
                if stype in ['Pin', 'Roller', 'Fixed']: 
                    free_dof[2*nid] = False
                if stype == 'Fixed': 
                    free_dof[2*nid+1] = False

        # 5. Solve for Displacements U
        U = np.zeros(dof)
        if np.any(~free_dof):
            try:
                K_sub = K[np.ix_(free_dof, free_dof)]
                F_sub = F[free_dof]
                U[free_dof] = solve(K_sub, F_sub)
            except:
                return pd.DataFrame(), np.zeros(dof), {}

        # 6. Reactions R = K*U - F
        R = K @ U - F

        # 7. Generate Results (Internal Forces)
        base_x = np.linspace(0, nodes[-1], 400)
        # ใส่จุดวิกฤตลงไปในกราฟเพื่อให้เส้นกราฟคมชัด
        critical_x = []
        for n in nodes: 
            critical_x.extend([n-1e-7, n, n+1e-7])
        all_x = np.unique(np.sort(np.concatenate([base_x, critical_x])))
        all_x = all_x[(all_x >= 0) & (all_x <= nodes[-1])]
        
        results = []
        for x in all_x:
            V, M, defl = 0.0, 0.0, 0.0
            # ดึง Displacement จาก element ที่ x สังกัดอยู่
            for elem in elements:
                x1, x2 = nodes[elem['n1']], nodes[elem['n2']]
                if x1 <= x <= x2 + 1e-7:
                    s = (x - x1) / elem['L']
                    idx = [2*elem['n1'], 2*elem['n1']+1, 2*elem['n2'], 2*elem['n2']+1]
                    H = np.array([1-3*s**2+2*s**3, (x-x1)*(1-s)**2, 3*s**2-2*s**3, (x-x1)*(s**2-s)])
                    defl = np.dot(H, U[idx])
                    break
            
            # --- แก้ไข Equilibrium: รวม Reaction และ Load จากซ้ายไปขวา ---
            for i, nx in enumerate(nodes):
                if nx <= x + 1e-6: # ใช้ tolerance เดียวกับข้อ 3
                    V += R[2*i]
                    M += R[2*i]*(x-nx) + R[2*i+1]
            
            for _, l in self.loads_df.iterrows():
                lx, lmag = l['x'], l['mag']
                if l['type'] == 'P' and lx <= x + 1e-6:
                    V -= lmag
                    M -= lmag * (x - lx)
                elif l['type'] == 'M' and lx <= x + 1e-6:
                    M -= lmag
                elif l['type'] == 'U' and lx < x:
                    dist_covered = min(x, lx + l['dist']) - lx
                    if dist_covered > 0:
                        f_total = lmag * dist_covered
                        V -= f_total
                        M -= f_total * (x - (lx + dist_covered/2))

            results.append({'x': x, 'deflection': defl, 'shear': V, 'moment': M})
            
        df_res = pd.DataFrame(results)
        return df_res, R, self._create_summary(df_res)

    def _get_k(self, L):
        EI = self.E * self.I
        return (EI / L**3) * np.array([
            [12, 6*L, -12, 6*L],
            [6*L, 4*L**2, -6*L, 2*L**2],
            [-12, -6*L, 12, -6*L],
            [6*L, 2*L**2, -6*L, 4*L**2]
        ])

    def _create_summary(self, df):
        if df.empty: return {}
        return {
            'V_max': {'value': df['shear'].abs().max(), 'x': df.iloc[df['shear'].abs().idxmax()]['x']},
            'M_pos': {'value': df['moment'].max(), 'x': df.iloc[df['moment'].idxmax()]['x']},
            'M_neg': {'value': df['moment'].min(), 'x': df.iloc[df['moment'].idxmin()]['x']},
            'D_max': {'value': df['deflection'].abs().max(), 'x': df.iloc[df['deflection'].abs().idxmax()]['x']}
        }
