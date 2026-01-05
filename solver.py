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
        
        # 1. กำหนดตำแหน่ง Global x ของแต่ละ Span
        self.cum_spans = [round(x, 4) for x in ([0.0] + list(np.cumsum(self.spans)))]
        
        # 2. Sanitize ข้อมูล (ต้องเรียกตามลำดับนี้)
        self.supports_df = self._sanitize_supports(supports_input)
        self.loads_df = self._sanitize_loads(loads_input)

    def _sanitize_supports(self, supports_input):
        """แก้ไขให้ดึง ID ตรงๆ จาก session_state['supports'] ของ app.py"""
        sanitized = []
        # แปลง input ให้เป็น list ของ dict เสมอ
        if isinstance(supports_input, pd.DataFrame):
            data = supports_input.to_dict('records')
        else:
            data = supports_input

        for s in data:
            # ดึง ID: ลองทั้ง 'id' (0-based) และ 'Node ID' (1-based)
            raw_id = s.get('id', s.get('Node ID'))
            if raw_id is None: continue
            
            idx = int(raw_id)
            # ถ้าคีย์มาจาก 'Node ID' ในตาราง Editor (ซึ่งเป็น 1, 2, 3...) ให้ลบ 1
            if 'Node ID' in s: idx -= 1
            
            # ดึง Type
            stype = s.get('type', s.get('Support Type', 'None'))
            
            if 0 <= idx < len(self.cum_spans):
                sanitized.append({
                    'x': self.cum_spans[idx],
                    'type': str(stype)
                })
        return pd.DataFrame(sanitized)

    def _sanitize_loads(self, loads_input):
        if not loads_input:
            return pd.DataFrame(columns=['span_idx', 'type', 'mag', 'x', 'dist', 'case'])
        
        df = pd.DataFrame(loads_input)
        def get_global_x(row):
            # ดึง index ของ span ที่โหลดลง
            s_idx = int(row.get('span_index', row.get('span_idx', 0)))
            local_x = float(row.get('x', 0))
            if s_idx < len(self.cum_spans):
                return round(self.cum_spans[s_idx] + local_x, 4)
            return local_x
            
        df['x'] = df.apply(get_global_x, axis=1)
        return df

    def _find_nearest_node(self, nodes, val):
        arr = np.array(nodes)
        diff = np.abs(arr - val)
        idx = diff.argmin()
        if diff[idx] < 1e-3: 
            return idx
        return -1

    def solve(self):
        # 1. สร้าง Nodes (จุดต่อ, จุดรองรับ, จุดลงแรง)
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

        # 3. Force Vector F
        F = np.zeros(dof)
        for _, load in self.loads_df.iterrows():
            nid = self._find_nearest_node(nodes, load['x'])
            if load['type'] == 'P' and nid != -1:
                F[2*nid] -= load['mag']
            elif load['type'] == 'M' and nid != -1:
                F[2*nid+1] += load['mag']
            elif load['type'] == 'U':
                start, end = load['x'], load['x'] + load['dist']
                for elem in elements:
                    ex1, ex2 = nodes[elem['n1']], nodes[elem['n2']]
                    overlap = min(end, ex2) - max(start, ex1)
                    if overlap > 1e-6:
                        # กระจายแรงลง Node สองข้าง (Exact equivalent for UDL)
                        F[2*elem['n1']] -= load['mag'] * overlap * 0.5
                        F[2*elem['n2']] -= load['mag'] * overlap * 0.5

        # 4. Boundary Conditions (จุดที่ต้อง Re-check เป็นพิเศษ)
        free_dof = np.full(dof, True)
        for _, sup in self.supports_df.iterrows():
            nid = self._find_nearest_node(nodes, sup['x'])
            stype = str(sup['type'])
            if nid != -1 and stype != "None":
                # ล็อคการเคลื่อนที่ในแนวแกน Y (Vertical)
                if stype in ['Pin', 'Roller', 'Fixed']:
                    free_dof[2*nid] = False
                # ล็อคการหมุน (Moment)
                if stype == 'Fixed':
                    free_dof[2*nid+1] = False

        # 5. แก้สมการหา Displacement U
        U = np.zeros(dof)
        if not np.all(free_dof):
            try:
                K_sub = K[np.ix_(free_dof, free_dof)]
                F_sub = F[free_dof]
                U[free_dof] = solve(K_sub, F_sub)
            except: pass

        # 6. หา Reactions R = K*U - F
        R = K @ U - F

        # 7. คำนวณแรงภายในเพื่อวาดกราฟ
        results = []
        plot_x = np.unique(np.concatenate([np.linspace(0, nodes[-1], 350), nodes]))
        for x in plot_x:
            V, M, defl = 0.0, 0.0, 0.0
            # Displacement (Shape Functions)
            for elem in elements:
                x1, x2 = nodes[elem['n1']], nodes[elem['n2']]
                if x1 <= x <= x2 + 1e-6:
                    s = (x - x1) / elem['L']
                    idx = [2*elem['n1'], 2*elem['n1']+1, 2*elem['n2'], 2*elem['n2']+1]
                    H = np.array([1-3*s**2+2*s**3, (x-x1)*(1-s)**2, 3*s**2-2*s**3, (x-x1)*(s**2-s)])
                    defl = np.dot(H, U[idx])
                    break
            
            # ตัด Section รวมแรงจากซ้ายมาที่ x
            for i, nx in enumerate(nodes):
                if nx <= x + 1e-4:
                    V += R[2*i]
                    M += R[2*i]*(x-nx) + R[2*i+1]
            for _, l in self.loads_df.iterrows():
                lx, lmag = l['x'], l['mag']
                if l['type'] == 'P' and lx <= x + 1e-4:
                    V -= lmag
                    M -= lmag*(x-lx)
                elif l['type'] == 'M' and lx <= x + 1e-4:
                    M -= lmag
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
