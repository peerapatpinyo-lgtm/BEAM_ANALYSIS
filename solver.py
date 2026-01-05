import numpy as np
import pandas as pd
from scipy.linalg import solve

class BeamSolver:
    def __init__(self, spans, supports_df, loads_input, E, I, A=None, G=None):
        self.spans = spans
        self.E = float(E)
        self.I = float(I)
        
        # --- 1. Sanitize Inputs Immediately ---
        # ต้องล้างข้อมูลทั้งคู่ให้เป็น Format มาตรฐานที่มีคอลัมน์ 'x' เป็น float
        self.loads_df = self._sanitize_loads(loads_input)
        self.supports_df = self._sanitize_supports(supports_df)

    def _sanitize_loads(self, data):
        # แปลง Input เป็น DataFrame
        if isinstance(data, list): df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame): df = data.copy()
        else: df = pd.DataFrame()

        if df.empty: return pd.DataFrame(columns=['x', 'mag', 'type', 'dist'])

        # แก้ชื่อ Column ให้ตรงกันหมด
        df.columns = [str(c).lower().strip() for c in df.columns]
        mapper = {
            'location': 'x', 'pos': 'x', 'loc': 'x',
            'magnitude': 'mag', 'force': 'mag', 'val': 'mag', 'p': 'mag',
            'kind': 'type', 'load_type': 'type',
            'length': 'dist', 'span': 'dist'
        }
        df.rename(columns=mapper, inplace=True)
        
        # เติมค่าที่ขาด
        defaults = {'x': 0.0, 'mag': 0.0, 'dist': 0.0, 'type': 'P'}
        for col, val in defaults.items():
            if col not in df.columns: df[col] = val

        # Clean Type
        def clean_t(t):
            t = str(t).upper()
            if 'U' in t: return 'U'
            if 'M' in t: return 'M'
            return 'P'
        df['type'] = df['type'].apply(clean_t)
        
        # Force Float (สำคัญมาก! ป้องกัน bug กราฟแบน)
        for c in ['x', 'mag', 'dist']:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
            
        return df

    def _sanitize_supports(self, data):
        # ล้างข้อมูล Support ให้มี 'x' แน่นอน
        if isinstance(data, list): df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame): df = data.copy()
        else: df = pd.DataFrame()
        
        if df.empty: return pd.DataFrame(columns=['x', 'type'])

        df.columns = [str(c).lower().strip() for c in df.columns]
        mapper = {'location': 'x', 'pos': 'x', 'loc': 'x', 'id': 'node_id'}
        df.rename(columns=mapper, inplace=True)
        
        # ถ้าไม่มี x แต่มี node_id เราจะไป map ทีหลัง แต่ต้องเตรียม column ไว้
        if 'x' not in df.columns: df['x'] = np.nan
        
        # Force Float for x
        df['x'] = pd.to_numeric(df['x'], errors='coerce')
        
        return df

    def solve(self):
        # --- 2. สร้าง Nodes (Discretization) ---
        # รวบรวมจุดสำคัญทั้งหมด: ปลายคาน, จุดโหลด, จุดซัพพอร์ต
        points = {0.0}
        curr = 0.0
        # Add span points
        for s in self.spans:
            curr += float(s)
            points.add(round(curr, 5))
            
        # Add load points
        for _, l in self.loads_df.iterrows():
            points.add(round(l['x'], 5))
            if l['type'] == 'U': points.add(round(l['x'] + l['dist'], 5))
            
        # Add support points (ถ้ามีระบุ x)
        for _, s in self.supports_df.iterrows():
            if pd.notna(s['x']): points.add(round(s['x'], 5))

        # สร้าง Node List ที่เรียงลำดับแล้ว
        nodes = sorted(list(points))
        num_nodes = len(nodes)
        dof = 2 * num_nodes
        
        # Mapping Node ID สำหรับ Support ที่ระบุมาเป็น ID
        node_lookup = {i: nodes[i] for i in range(num_nodes)}

        # --- 3. สร้าง Matrix K และ Vector F ---
        K = np.zeros((dof, dof))
        F = np.zeros(dof)
        elements = []

        for i in range(num_nodes - 1):
            x1, x2 = nodes[i], nodes[i+1]
            L = x2 - x1
            elements.append({'n1': i, 'n2': i+1, 'L': L})
            
            # Element Stiffness
            k_el = self._get_k(L)
            idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
            for r in range(4):
                for c in range(4):
                    K[idx[r], idx[c]] += k_el[r, c]

        # Apply Loads
        for _, load in self.loads_df.iterrows():
            # หา Node ที่ใกล้ที่สุด
            nid = self._find_nearest_node(nodes, load['x'])
            
            if load['type'] == 'P':
                F[2*nid] -= load['mag']
            elif load['type'] == 'M':
                F[2*nid+1] += load['mag']
            elif load['type'] == 'U':
                # Fixed End Forces (FEM) for UDL
                start, dist, mag = load['x'], load['dist'], load['mag']
                end = start + dist
                for elem in elements:
                    ex1, ex2 = nodes[elem['n1']], nodes[elem['n2']]
                    if ex2 <= start + 1e-6 or ex1 >= end - 1e-6: continue
                    
                    # คำนวณส่วนที่ UDL ทับ Element นี้
                    ov_s = max(start, ex1)
                    ov_e = min(end, ex2)
                    len_load = ov_e - ov_s
                    
                    # Simple Gauss Quadrature
                    mid = (ov_s + ov_e)/2
                    for gp in [-0.57735, 0.57735]:
                        xi = mid + (len_load/2)*gp
                        s = (xi - ex1) / elem['L']
                        # Shape functions for force distribution
                        N = np.array([1-3*s**2+2*s**3, xi*(1-s)**2, 3*s**2-2*s**3, xi*(s**2-s)]) # ผิดสูตรนิดหน่อยสำหรับ Moment แต่ใช้แก้ขัดได้
                        # ใช้สูตร Reaction ตรงๆ ดีกว่าสำหรับ UDL เพื่อความชัวร์ใน Element Force Vector
                        # แต่เพื่อความเร็ว ใช้ Nodal Equivalent Load แบบ Integration
                        N_trans = np.array([
                            1 - 3*s**2 + 2*s**3,       # v1
                            (xi - ex1)*(1-s)**2,       # theta1
                            3*s**2 - 2*s**3,           # v2
                            (xi - ex1)*(s**2-s)        # theta2
                        ])
                        F[[2*elem['n1'], 2*elem['n1']+1, 2*elem['n2'], 2*elem['n2']+1]] -= N_trans * mag * (len_load/2)

        # --- 4. Boundary Conditions ---
        free_dof = np.full(dof, True)
        
        for _, sup in self.supports_df.iterrows():
            target_node = -1
            
            # Case 1: ระบุด้วย x
            if pd.notna(sup['x']):
                target_node = self._find_nearest_node(nodes, sup['x'])
            # Case 2: ระบุด้วย id (ต้องระวัง id เปลี่ยน)
            elif 'node_id' in sup and pd.notna(sup['node_id']):
                # พยายามเดาว่า User หมายถึง Node ไหน
                # ถ้า User บอก Node 1 (และมี 3 Span) มันอาจจะหมายถึง x ที่ 5.0
                # ตรงนี้เราข้ามไปก่อน ให้ยึด x เป็นหลักถ้าทำได้
                try:
                    tid = int(sup['node_id'])
                    if tid < num_nodes: target_node = tid
                except: pass
                
            if target_node != -1:
                stype = sup.get('type', 'Pin')
                if stype in ['Pin', 'Roller', 'Fixed']:
                    free_dof[2*target_node] = False # Fix Y
                if stype == 'Fixed':
                    free_dof[2*target_node+1] = False # Fix Rotation

        # --- 5. Solve ---
        U = np.zeros(dof)
        if np.sum(free_dof) < dof:
            try:
                U[free_dof] = solve(K[np.ix_(free_dof, free_dof)], F[free_dof])
            except: return pd.DataFrame(), [], {}
            
        # Calculate Reactions (R = K*U - F_applied)
        # R คือแรงที่ Node กระทำต่อคาน (Reaction)
        R = K @ U - F 

        # --- 6. Generate Results (Graphing) ---
        # จุดตายอยู่ตรงนี้! ผมเขียนใหม่ให้ถึกที่สุด
        x_eval = np.linspace(0, nodes[-1], 200)
        results = []
        
        for x in x_eval:
            x = float(x)
            
            # A. Deflection
            defl = 0.0
            for elem in elements:
                x1, x2 = nodes[elem['n1']], nodes[elem['n2']]
                if x1 <= x <= x2 + 1e-6:
                    s = (x - x1) / elem['L']
                    idx = [2*elem['n1'], 2*elem['n1']+1, 2*elem['n2'], 2*elem['n2']+1]
                    u_vec = U[idx]
                    # Shape function v(x)
                    H = np.array([1-3*s**2+2*s**3, (x-x1)*(1-s)**2, 3*s**2-2*s**3, (x-x1)*(s**2-s)])
                    defl = np.dot(H, u_vec)
                    break
            
            # B. Shear & Moment (Summation Method from Left)
            V = 0.0
            M = 0.0
            
            # 1. Add Reactions from Nodes on the left
            for i, nx in enumerate(nodes):
                if nx <= x + 1e-4: # ถ้า Node อยู่ซ้ายกว่า x
                    Ry = R[2*i]
                    Mz = R[2*i+1]
                    V += Ry
                    M += Ry * (x - nx) - Mz # Sign convention correction
            
            # 2. Subtract Applied Loads on the left
            for _, l in self.loads_df.iterrows():
                lx, mag = l['x'], l['mag']
                if l['type'] == 'P' and lx <= x + 1e-4:
                    V -= mag
                    M -= mag * (x - lx)
                elif l['type'] == 'M' and lx <= x + 1e-4:
                    M -= mag
                elif l['type'] == 'U':
                    start, end = lx, lx + l['dist']
                    if start < x:
                        cov_end = min(x, end)
                        cov_len = cov_end - start
                        force = mag * cov_len
                        arm = x - (start + cov_len/2)
                        V -= force
                        M -= force * arm
                        
            results.append({'x': x, 'deflection': defl, 'shear': V, 'moment': M})
            
        return pd.DataFrame(results), R, {}

    def _get_k(self, L):
        if L == 0: return np.zeros((4,4))
        k = (self.E * self.I / L**3) * np.array([
            [12, 6*L, -12, 6*L],
            [6*L, 4*L**2, -6*L, 2*L**2],
            [-12, -6*L, 12, -6*L],
            [6*L, 2*L**2, -6*L, 4*L**2]
        ])
        return k

    def _find_nearest_node(self, nodes, val):
        val = float(val)
        idx = (np.abs(np.array(nodes) - val)).argmin()
        if abs(nodes[idx] - val) < 1e-4:
            return idx
        return -1
