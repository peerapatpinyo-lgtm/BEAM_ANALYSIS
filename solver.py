import numpy as np
import pandas as pd
from scipy.linalg import solve

class BeamSolver:
    def __init__(self, spans, supports_df, loads_input, E, I, A=None, G=None):
        self.spans = spans
        self.supports_df = supports_df
        self.E = E
        self.I = I
        # Robust Load Sanitization
        self.loads_df = self._sanitize_loads(loads_input)

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

        # 2. Rename Columns (Map common names to standard)
        df.columns = [str(c).lower().strip() for c in df.columns]
        mapper = {
            'location': 'x', 'pos': 'x', 'loc': 'x',
            'magnitude': 'mag', 'force': 'mag', 'val': 'mag', 'p': 'mag',
            'kind': 'type', 'load_type': 'type',
            'length': 'dist', 'span': 'dist'
        }
        df.rename(columns=mapper, inplace=True)

        # 3. Clean Types & Values
        if 'type' not in df.columns: df['type'] = 'P'
        if 'dist' not in df.columns: df['dist'] = 0.0
        
        # Standardize Load Type Chars
        def clean_t(t):
            t = str(t).upper()
            if 'U' in t: return 'U'
            if 'M' in t: return 'M'
            return 'P'
        df['type'] = df['type'].apply(clean_t)
        
        # Ensure Numeric
        for c in ['x', 'mag', 'dist']:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
            
        return df

    def solve(self):
        # --- 1. Discretize Model (Generate Nodes) ---
        nodes, elements = self._discretize_model()
        num_nodes = len(nodes)
        dof = 2 * num_nodes
        
        K = np.zeros((dof, dof))
        F = np.zeros(dof)
        
        # --- 2. Stiffness Matrix ---
        for elem in elements:
            L = nodes[elem['n2']] - nodes[elem['n1']]
            k_local = self._get_element_stiffness(L)
            idx = [2*elem['n1'], 2*elem['n1']+1, 2*elem['n2'], 2*elem['n2']+1]
            for r in range(4):
                for c in range(4):
                    K[idx[r], idx[c]] += k_local[r, c]
                    
        # --- 3. Loads (Mapping to Nodes) ---
        for _, load in self.loads_df.iterrows():
            # Find closest node index for load
            nid = self._find_node_idx(nodes, load['x'])
            
            if load['type'] == 'P' and nid is not None:
                F[2*nid] -= load['mag']
            elif load['type'] == 'M' and nid is not None:
                F[2*nid+1] += load['mag']
            elif load['type'] == 'U':
                # Equivalent Nodal Forces for UDL
                start, dist, mag = load['x'], load['dist'], load['mag']
                end = start + dist
                for elem in elements:
                    x1, x2 = nodes[elem['n1']], nodes[elem['n2']]
                    L = x2 - x1
                    if L < 1e-9: continue
                    
                    # Calculate overlap
                    ov_st = max(start, x1)
                    ov_en = min(end, x2)
                    if ov_en > ov_st + 1e-6:
                        # Integration
                        a, b = ov_st - x1, ov_en - x1
                        length = b - a
                        mid = (a + b)/2
                        fe = np.zeros(4)
                        for gp in [-0.57735, 0.57735]:
                            loc = mid + (length/2)*gp
                            s = loc / L
                            N = np.array([1-3*s**2+2*s**3, loc*(1-s)**2, 3*s**2-2*s**3, loc*(s**2-s)])
                            fe += N * (-mag) * (length/2)
                        
                        idx = [2*elem['n1'], 2*elem['n1']+1, 2*elem['n2'], 2*elem['n2']+1]
                        F[idx] += fe

        # --- 4. Boundary Conditions (FIXED THIS PART) ---
        free_dof = np.full(dof, True)
        
        # Convert supports to list of dicts
        if isinstance(self.supports_df, pd.DataFrame):
            sup_data = self.supports_df.to_dict('records')
        else:
            sup_data = self.supports_df

        for sup in sup_data:
            # Check if 'x' or 'location' exists, otherwise try 'id'
            sx = sup.get('x', sup.get('location', None))
            
            # Map Support X to Closest Node
            target_node = -1
            if sx is not None:
                target_node = self._find_node_idx(nodes, sx)
            elif 'id' in sup:
                # Fallback to ID if no X provided (less safe)
                target_node = int(sup['id']) if int(sup['id']) < num_nodes else -1

            if target_node != -1:
                stype = sup.get('type', 'Pin')
                # Apply Constraints
                if stype in ['Pin', 'Roller', 'Fixed']:
                    free_dof[2*target_node] = False # Fix Y
                if stype == 'Fixed':
                    free_dof[2*target_node+1] = False # Fix Rotation

        # --- 5. Solve ---
        U = np.zeros(dof)
        if np.sum(free_dof) < dof: # Only solve if constrained
            try:
                K_free = K[np.ix_(free_dof, free_dof)]
                F_free = F[free_dof]
                U[free_dof] = solve(K_free, F_free)
            except:
                print("Error: Matrix Singular (Unstable)")
                return pd.DataFrame(), [], {}
        
        R = K @ U - F
        
        # --- 6. Results Generation (Hybrid) ---
        x_eval = np.linspace(0, nodes[-1], 300)
        results = []
        
        for x in x_eval:
            # Deflection (Shape Function)
            defl = 0
            for elem in elements:
                x1, x2 = nodes[elem['n1']], nodes[elem['n2']]
                if x1 <= x <= x2 + 1e-6:
                    s = (x - x1)/(x2 - x1)
                    idx = [2*elem['n1'], 2*elem['n1']+1, 2*elem['n2'], 2*elem['n2']+1]
                    N = np.array([1-3*s**2+2*s**3, (x-x1)*(1-s)**2, 3*s**2-2*s**3, (x-x1)*(s**2-s)])
                    defl = np.dot(N, U[idx])
                    break
            
            # Shear/Moment (Sum of Forces)
            V, M = 0.0, 0.0
            
            # Reactions contribution
            for i, nx in enumerate(nodes):
                if nx <= x + 1e-5:
                    V += R[2*i]
                    M += R[2*i]*(x-nx) + R[2*i+1]
            
            # Loads contribution
            for _, l in self.loads_df.iterrows():
                lx, mag = l['x'], l['mag']
                if l['type'] == 'P' and lx <= x + 1e-5:
                    V -= mag
                    M -= mag*(x-lx)
                elif l['type'] == 'M' and lx <= x + 1e-5:
                    M += mag # Adjust sign based on convention
                elif l['type'] == 'U':
                    start, end = lx, lx + l['dist']
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
        # Gather all critical points
        pts = {0.0}
        curr = 0
        for s in self.spans:
            curr += s
            pts.add(round(curr, 5))
        for _, l in self.loads_df.iterrows():
            pts.add(round(l['x'], 5))
            if l['type'] == 'U': pts.add(round(l['x']+l['dist'], 5))
        
        # Convert supports to points too
        if isinstance(self.supports_df, pd.DataFrame):
            for _, s in self.supports_df.iterrows():
                if 'x' in s: pts.add(round(s['x'], 5))
                elif 'location' in s: pts.add(round(s['location'], 5))
        
        sx = sorted(list(pts))
        return sx, [{'n1':i, 'n2':i+1} for i in range(len(sx)-1)]
        
    def _find_node_idx(self, nodes, x_val):
        # Helper to find closest node index
        for i, val in enumerate(nodes):
            if abs(val - x_val) < 1e-4:
                return i
        return None

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
