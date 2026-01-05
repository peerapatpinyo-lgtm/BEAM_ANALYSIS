import numpy as np
import pandas as pd
from scipy.linalg import solve

class BeamSolver:
    def __init__(self, spans, supports_df, loads_input, E, I, A=None, G=None):
        self.spans = spans
        self.supports_df = supports_df
        self.E = E
        self.I = I
        
        # --- 1. Auto-Fix Loads Data ---
        self.loads_df = self._sanitize_loads(loads_input)
        
        # --- DEBUG: Print to Terminal to check data ---
        print("\n--- DEBUG: LOADS RECEIVED BY SOLVER ---")
        print(self.loads_df)
        print("---------------------------------------\n")

    def _sanitize_loads(self, loads_input):
        # 1. Convert to DataFrame
        if isinstance(loads_input, list):
            df = pd.DataFrame(loads_input)
        elif isinstance(loads_input, pd.DataFrame):
            df = loads_input.copy()
        else:
            return pd.DataFrame(columns=['x', 'mag', 'type', 'dist'])

        if df.empty:
            return pd.DataFrame(columns=['x', 'mag', 'type', 'dist'])

        # 2. Normalize Column Names (แก้ปัญหาชื่อไม่ตรง)
        # แปลงชื่อคอลัมน์ทั้งหมดเป็นตัวเล็กก่อน
        df.columns = [str(c).lower().strip() for c in df.columns]
        
        # สร้าง Map สำหรับแปลงชื่อตัวแปรที่คนมักใช้ผิด
        col_mapper = {
            'location': 'x', 'pos': 'x', 'position': 'x', 'loc': 'x',
            'magnitude': 'mag', 'load': 'mag', 'value': 'mag', 'force': 'mag', 'p': 'mag', 'w': 'mag',
            'load_type': 'type', 'kind': 'type',
            'distance': 'dist', 'span': 'dist', 'length': 'dist'
        }
        df.rename(columns=col_mapper, inplace=True)

        # 3. Normalize Load Types (แก้ปัญหาคำว่า Point Load vs P)
        # ถ้าไม่มี column type ให้เดาว่าเป็น P ไว้ก่อน
        if 'type' not in df.columns:
            df['type'] = 'P'
        
        def clean_type(val):
            s = str(val).upper().strip()
            if 'POINT' in s or s == 'P': return 'P'
            if 'UNIFORM' in s or 'UDL' in s or s == 'U': return 'U'
            if 'MOMENT' in s or s == 'M': return 'M'
            return 'P' # Default fallback
            
        df['type'] = df['type'].apply(clean_type)

        # 4. Fill Missing Columns with Defaults
        if 'x' not in df.columns: df['x'] = 0.0
        if 'mag' not in df.columns: df['mag'] = 0.0
        if 'dist' not in df.columns: df['dist'] = 0.0

        # 5. Convert to Numeric (Force Float)
        for col in ['x', 'mag', 'dist']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        return df

    def solve(self):
        # --- 2. Model Discretization ---
        nodes, elements = self._discretize_model()
        num_nodes = len(nodes)
        dof = 2 * num_nodes
        
        K = np.zeros((dof, dof))
        F = np.zeros(dof)
        
        # --- 3. Stiffness Matrix ---
        for elem in elements:
            n1, n2 = elem['n1'], elem['n2']
            x1, x2 = nodes[n1], nodes[n2]
            L = x2 - x1
            k_local = self._get_element_stiffness(L)
            
            idx = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
            for r in range(4):
                for c in range(4):
                    K[idx[r], idx[c]] += k_local[r, c]
                    
        # --- 4. Force Vector Assembly ---
        # A. Nodal Loads
        for _, load in self.loads_df.iterrows():
            # Find closest node
            node_idx = -1
            min_dist = 1e9
            for i, x in enumerate(nodes):
                dist = abs(x - load['x'])
                if dist < 1e-4:
                    node_idx = i
                    break
            
            if node_idx != -1:
                # Direct Nodal Load
                if load['type'] == 'P':
                    F[2 * node_idx] -= load['mag']
                elif load['type'] == 'M':
                    F[2 * node_idx + 1] += load['mag']

        # B. Member Loads (Equivalent Nodal Forces)
        for _, load in self.loads_df.iterrows():
            if load['type'] == 'U':
                start, end = load['x'], load['x'] + load['dist']
                mag = load['mag']
                
                for elem in elements:
                    x1, x2 = nodes[elem['n1']], nodes[elem['n2']]
                    L = x2 - x1
                    if L <= 1e-9: continue
                    
                    overlap_start = max(start, x1)
                    overlap_end = min(end, x2)
                    
                    if overlap_end > overlap_start + 1e-6:
                        # Gauss Quadrature for FE
                        a, b = overlap_start - x1, overlap_end - x1
                        load_len = b - a
                        mid = (a + b) / 2
                        
                        fe = np.zeros(4)
                        for gp in [-0.57735, 0.57735]: # 2-point Gauss
                            x_loc = mid + (load_len/2)*gp
                            s = x_loc / L
                            # Shape Functions
                            N = np.array([1-3*s**2+2*s**3, x_loc*(1-s)**2, 3*s**2-2*s**3, x_loc*(s**2-s)])
                            fe += N * (-mag) * (load_len/2) # Weight=1.0
                            
                        idx = [2*elem['n1'], 2*elem['n1']+1, 2*elem['n2'], 2*elem['n2']+1]
                        F[idx] += fe

        # --- 5. Boundary Conditions ---
        free_dof = np.full(dof, True)
        if isinstance(self.supports_df, list):
             sup_data = self.supports_df
        else:
             sup_data = self.supports_df.to_dict('records')

        for sup in sup_data:
            try:
                nid = int(sup['id'])
                if nid < num_nodes:
                    stype = sup['type']
                    if stype in ['Pin', 'Roller', 'Fixed']: free_dof[2*nid] = False
                    if stype == 'Fixed': free_dof[2*nid+1] = False
            except: pass
        
        # --- 6. Solve ---
        U = np.zeros(dof)
        if np.sum(free_dof) > 0:
            try:
                K_red = K[np.ix_(free_dof, free_dof)]
                F_red = F[free_dof]
                U[free_dof] = solve(K_red, F_red)
            except:
                print("Error: Singular Matrix")
                return pd.DataFrame(), [], {}

        R = K @ U - F
        
        # --- 7. Generate Graph Points (Hybrid Method) ---
        x_eval = np.linspace(0, nodes[-1], 200)
        results = []
        
        for x in x_eval:
            # Deflection
            defl = 0
            for elem in elements:
                x1, x2 = nodes[elem['n1']], nodes[elem['n2']]
                if x1 <= x <= x2 + 1e-6:
                    s = (x - x1) / (x2 - x1)
                    u_local = U[[2*elem['n1'], 2*elem['n1']+1, 2*elem['n2'], 2*elem['n2']+1]]
                    N = np.array([1-3*s**2+2*s**3, (x-x1)*(1-s)**2, 3*s**2-2*s**3, (x-x1)*(s**2-s)])
                    defl = np.dot(N, u_local)
                    break
            
            # Shear & Moment (Statics Integration)
            V, M = 0.0, 0.0
            
            # Reactions
            for i, nx in enumerate(nodes):
                if nx <= x + 1e-5:
                    V += R[2*i]
                    M += R[2*i]*(x-nx) + R[2*i+1]
            
            # Loads
            for _, load in self.loads_df.iterrows():
                lx, mag = load['x'], load['mag']
                ltype = load['type']
                
                if ltype == 'P' and lx <= x + 1e-5:
                    V -= mag
                    M -= mag * (x - lx)
                elif ltype == 'M' and lx <= x + 1e-5:
                    M -= mag # Assuming CW applied moment
                elif ltype == 'U':
                    start, end = lx, lx + load['dist']
                    if start < x:
                        act_end = min(x, end)
                        cov = act_end - start
                        force = mag * cov
                        arm = x - (start + cov/2)
                        V -= force
                        M -= force * arm
            
            results.append({'x': x, 'deflection': defl, 'shear': V, 'moment': M})
            
        return pd.DataFrame(results), R, {}

    def _discretize_model(self):
        pts = {0.0}
        curr = 0
        for s in self.spans:
            curr += s
            pts.add(round(curr, 5))
        for _, l in self.loads_df.iterrows():
            pts.add(round(l['x'], 5))
            if l['type'] == 'U': pts.add(round(l['x']+l['dist'], 5))
        
        sx = sorted(list(pts))
        return sx, [{'n1':i, 'n2':i+1} for i in range(len(sx)-1)]

    def _get_element_stiffness(self, L):
        k = np.zeros((4,4))
        if L==0: return k
        c = self.E * self.I / L**3
        k = c * np.array([
            [12, 6*L, -12, 6*L],
            [6*L, 4*L**2, -6*L, 2*L**2],
            [-12, -6*L, 12, -6*L],
            [6*L, 2*L**2, -6*L, 4*L**2]
        ])
        return k
