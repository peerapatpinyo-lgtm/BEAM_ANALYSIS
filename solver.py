import numpy as np
import pandas as pd
from scipy.linalg import solve

class BeamSolver:
    def __init__(self, spans, supports_df, loads_input, E, I, A=None, G=None):
        self.spans = [float(s) for s in spans]
        self.E = float(E)
        self.I = float(I)
        self.cum_spans = [0.0] + list(np.cumsum(self.spans))
        
        self.loads_df = self._sanitize_loads(loads_input)
        self.supports_df = self._sanitize_supports(supports_df)

    def _sanitize_loads(self, data):
        if isinstance(data, list): df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame): df = data.copy()
        else: df = pd.DataFrame()

        if df.empty: return pd.DataFrame(columns=['x', 'mag', 'type', 'dist'])

        df.columns = [str(c).lower().strip() for c in df.columns]
        mapper = {
            'location': 'x', 'pos': 'x', 'loc': 'x',
            'magnitude': 'mag', 'force': 'mag', 'val': 'mag', 'p': 'mag',
            'kind': 'type', 'load_type': 'type',
            'length': 'dist', 'span': 'dist',
            'span_idx': 'span_index', 'span_id': 'span_index', 'span_index': 'span_index'
        }
        df.rename(columns=mapper, inplace=True)
        
        defaults = {'x': 0.0, 'mag': 0.0, 'dist': 0.0, 'type': 'P', 'span_index': -1}
        for col, val in defaults.items():
            if col not in df.columns: df[col] = val

        def clean_t(t):
            t = str(t).upper()
            if 'U' in t: return 'U'
            if 'M' in t: return 'M'
            return 'P'
        df['type'] = df['type'].apply(clean_t)
        
        for c in ['x', 'mag', 'dist', 'span_index']:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)

        # Convert Local to Global X
        def adjust_x(row):
            idx = int(row['span_index'])
            local_x = float(row['x'])
            if 0 <= idx < len(self.cum_spans) - 1:
                return self.cum_spans[idx] + local_x
            return local_x 

        df['x'] = df.apply(adjust_x, axis=1)
        return df

    def _sanitize_supports(self, data):
        if isinstance(data, list): df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame): df = data.copy()
        else: df = pd.DataFrame()
        
        if df.empty: return pd.DataFrame(columns=['x', 'type'])

        df.columns = [str(c).lower().strip() for c in df.columns]
        mapper = {'location': 'x', 'pos': 'x', 'loc': 'x', 'id': 'node_id', 'node id': 'node_id'}
        df.rename(columns=mapper, inplace=True)
        
        if 'x' not in df.columns: df['x'] = np.nan
        
        def resolve_sup_x(row):
            if pd.notna(row['x']): return float(row['x'])
            if 'node_id' in row and pd.notna(row['node_id']):
                try:
                    nid = int(row['node_id'])
                    if 0 <= nid < len(self.cum_spans):
                        return self.cum_spans[nid]
                except: pass
            return np.nan

        df['x'] = df.apply(resolve_sup_x, axis=1)
        df.dropna(subset=['x'], inplace=True)
        return df

    def solve(self):
        # 1. Critical Points for Discretization
        points = set(self.cum_spans)
        for _, l in self.loads_df.iterrows():
            points.add(round(l['x'], 5))
            if l['type'] == 'U': points.add(round(l['x'] + l['dist'], 5))
        
        # Add support points explicitly
        for _, s in self.supports_df.iterrows():
            points.add(round(s['x'], 5))
            
        nodes = sorted(list(points))
        num_nodes = len(nodes)
        dof = 2 * num_nodes
        
        # 2. Stiffness K
        K = np.zeros((dof, dof))
        elements = []
        for i in range(num_nodes - 1):
            x1, x2 = nodes[i], nodes[i+1]
            L = x2 - x1
            elements.append({'n1': i, 'n2': i+1, 'L': L})
            if L > 0:
                k_el = self._get_k(L)
                idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
                for r in range(4):
                    for c in range(4):
                        K[idx[r], idx[c]] += k_el[r, c]

        # 3. Force F
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
                    if ex2 <= start + 1e-6 or ex1 >= end - 1e-6: continue
                    ov_s = max(start, ex1)
                    ov_e = min(end, ex2)
                    len_load = ov_e - ov_s
                    mid = (ov_s + ov_e)/2
                    for gp in [-0.57735, 0.57735]:
                        xi = mid + (len_load/2)*gp
                        s = (xi - ex1) / elem['L']
                        N_vec = np.array([1-3*s**2+2*s**3, (xi-ex1)*(1-s)**2, 3*s**2-2*s**3, (xi-ex1)*(s**2-s)])
                        F[[2*elem['n1'], 2*elem['n1']+1, 2*elem['n2'], 2*elem['n2']+1]] -= N_vec * mag * (len_load/2)

        # 4. Supports
        free_dof = np.full(dof, True)
        for _, sup in self.supports_df.iterrows():
            nid = self._find_nearest_node(nodes, sup['x'])
            if nid != -1:
                stype = sup.get('type', 'Pin')
                if stype in ['Pin', 'Roller', 'Fixed']: free_dof[2*nid] = False
                if stype == 'Fixed': free_dof[2*nid+1] = False

        # 5. Solve
        U = np.zeros(dof)
        if np.sum(free_dof) < dof:
            try:
                U[free_dof] = solve(K[np.ix_(free_dof, free_dof)], F[free_dof])
            except: return pd.DataFrame(), [], {}

        R = K @ U - F

        # 6. Post-Processing with "Micro-stepping" for Exact SFD Shape
        # สร้างจุด Evaluation ที่ละเอียด + จุด Discontinuity
        base_x = np.linspace(0, nodes[-1], 400)
        critical_x = []
        for n in nodes:
            # เพิ่มจุดก่อนและหลัง Node นิดเดียว เพื่อให้กราฟ Shear ตัดฉับพลัน (Vertical Line)
            critical_x.extend([n - 1e-6, n, n + 1e-6])
        
        # รวมจุดและเรียงลำดับ
        all_x = np.concatenate([base_x, critical_x])
        all_x = np.unique(np.sort(all_x))
        all_x = all_x[(all_x >= 0) & (all_x <= nodes[-1])] # ตัดส่วนเกิน
        
        results = []
        for x in all_x:
            x = float(x)
            # Deflection
            defl = 0.0
            for elem in elements:
                x1, x2 = nodes[elem['n1']], nodes[elem['n2']]
                if x1 <= x <= x2 + 1e-6:
                    s = (x - x1) / elem['L']
                    idx = [2*elem['n1'], 2*elem['n1']+1, 2*elem['n2'], 2*elem['n2']+1]
                    H = np.array([1-3*s**2+2*s**3, (x-x1)*(1-s)**2, 3*s**2-2*s**3, (x-x1)*(s**2-s)])
                    defl = np.dot(H, U[idx])
                    break
            
            # Statics Integration for V/M
            V, M = 0.0, 0.0
            # Reactions
            for i, nx in enumerate(nodes):
                if nx <= x + 1e-5:
                    V += R[2*i]
                    M += R[2*i]*(x-nx) + R[2*i+1]
            # Loads
            for _, l in self.loads_df.iterrows():
                lx, mag = l['x'], l['mag']
                # Point Load: คิดเมื่อ x เลยจุด load มาแล้ว (x >= lx)
                if l['type'] == 'P':
                    if lx <= x + 1e-5: 
                        V -= mag
                        M -= mag * (x - lx)
                elif l['type'] == 'M':
                    if lx <= x + 1e-5:
                        M -= mag # CW convention
                elif l['type'] == 'U':
                    start, end = lx, lx + l['dist']
                    if start < x:
                        cov = min(x, end) - start
                        force = mag * cov
                        arm = x - (start + cov/2)
                        V -= force
                        M -= force * arm
                        
            results.append({'x': x, 'deflection': defl, 'shear': V, 'moment': M})
            
        df_res = pd.DataFrame(results)
        summary = {}
        if not df_res.empty:
            summary['V_max'] = {'value': df_res['shear'].abs().max(), 'x': df_res.loc[df_res['shear'].abs().idxmax(), 'x']}
            summary['M_pos'] = {'value': df_res['moment'].max(), 'x': df_res.loc[df_res['moment'].idxmax(), 'x']}
            summary['M_neg'] = {'value': df_res['moment'].min(), 'x': df_res.loc[df_res['moment'].idxmin(), 'x']}
            summary['D_max'] = {'value': df_res['deflection'].abs().max(), 'x': df_res.loc[df_res['deflection'].abs().idxmax(), 'x']}

        return df_res, R, summary

    def _get_k(self, L):
        k = np.zeros((4,4))
        if L==0: return k
        c = self.E * self.I / L**3
        k = c * np.array([[12, 6*L, -12, 6*L], [6*L, 4*L**2, -6*L, 2*L**2], [-12, -6*L, 12, -6*L], [6*L, 2*L**2, -6*L, 4*L**2]])
        return k

    def _find_nearest_node(self, nodes, val):
        idx = (np.abs(np.array(nodes) - val)).argmin()
        return idx if abs(nodes[idx] - val) < 1e-4 else -1
