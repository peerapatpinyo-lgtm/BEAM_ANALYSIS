import numpy as np
import pandas as pd
from scipy.linalg import solve

class BeamSolver:
    def __init__(self, spans, supports_df, loads_input, E, I, A=None, G=None):
        """
        :param loads_input: Can be a list of dicts or a DataFrame.
                            Must contain 'x' (global coordinate), 'mag', 'type'.
                            Optional: 'dist' (for UDL).
        """
        self.spans = spans
        self.supports_df = supports_df
        self.E = E
        self.I = I
        
        # --- 1. Sanitize & Prepare Loads Data ---
        self.loads_df = self._sanitize_loads(loads_input)

    def _sanitize_loads(self, loads_input):
        # Convert List to DataFrame if needed
        if isinstance(loads_input, list):
            df = pd.DataFrame(loads_input)
        elif isinstance(loads_input, pd.DataFrame):
            df = loads_input.copy()
        else:
            return pd.DataFrame(columns=['x', 'mag', 'type', 'dist'])

        if df.empty:
            return pd.DataFrame(columns=['x', 'mag', 'type', 'dist'])

        # Normalize column names to lowercase
        df.columns = [str(c).lower().strip() for c in df.columns]

        # Ensure essential columns exist
        required_cols = ['x', 'mag', 'type']
        for col in required_cols:
            if col not in df.columns:
                # If 'x' is missing but 'span_idx' exists, you might need extra logic here 
                # (Assuming input is already converted to Global X by main.py)
                df[col] = 0 

        if 'dist' not in df.columns:
            df['dist'] = 0.0

        # Force numeric types (Fill NaN with 0)
        cols_numeric = ['x', 'mag', 'dist']
        for col in cols_numeric:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        # Clean Types
        df['type'] = df['type'].astype(str).str.strip().str.upper() # Ensure P, U, M are uppercase

        return df

    def solve(self):
        # --- 2. Model Discretization (FEM Nodes) ---
        nodes, elements = self._discretize_model()
        num_nodes = len(nodes)
        dof = 2 * num_nodes
        
        K = np.zeros((dof, dof))
        F = np.zeros(dof)
        
        # --- 3. Stiffness Matrix Assembly ---
        for elem in elements:
            node_i = elem['n1']
            node_j = elem['n2']
            x1 = nodes[node_i]
            x2 = nodes[node_j]
            L = x2 - x1
            
            # Element Stiffness
            k_local = self._get_element_stiffness(L)
            
            # Map to Global K
            idx = [2*node_i, 2*node_i+1, 2*node_j, 2*node_j+1]
            for r in range(4):
                for c in range(4):
                    K[idx[r], idx[c]] += k_local[r, c]
                    
        # --- 4. Force Vector Assembly ---
        # A. Nodal Loads (Directly at nodes)
        for _, load in self.loads_df.iterrows():
            # Find closest node match
            node_idx = -1
            for i, x in enumerate(nodes):
                if np.isclose(x, load['x'], atol=1e-4):
                    node_idx = i
                    break
            
            if node_idx != -1:
                if load['type'] == 'P':
                    F[2 * node_idx] -= load['mag'] 
                elif load['type'] == 'M':
                    F[2 * node_idx + 1] += load['mag'] # Global Moment Convention

        # B. Member Loads (Equivalent Nodal Forces)
        for _, load in self.loads_df.iterrows():
            if load['type'] == 'U':
                start = load['x']
                dist = load['dist']
                if dist <= 1e-6: continue # Skip zero length
                
                end = start + dist
                mag = load['mag']
                
                for elem in elements:
                    x1, x2 = nodes[elem['n1']], nodes[elem['n2']]
                    L_elem = x2 - x1
                    if L_elem <= 1e-9: continue

                    # Check Overlap
                    overlap_start = max(start, x1)
                    overlap_end = min(end, x2)
                    
                    if overlap_end > overlap_start + 1e-6:
                        # Fixed End Forces Calculation for Partial UDL
                        a = overlap_start - x1
                        b = overlap_end - x1
                        w = -mag # Downward load is negative in FEM vector formulation
                        
                        # Gauss Quadrature (2-point)
                        load_len = b - a
                        mid = (a + b) / 2
                        gauss_pts = [-0.57735, 0.57735]
                        gauss_w = [1.0, 1.0]
                        
                        fe = np.zeros(4)
                        for gp, gw in zip(gauss_pts, gauss_w):
                            x_in_elem = mid + (load_len/2)*gp
                            s = (x_in_elem - x1) / L_elem
                            
                            # Shape Functions
                            n_vec = np.array([
                                1 - 3*s**2 + 2*s**3,       # v1
                                (x_in_elem - x1)*(1-s)**2, # theta1
                                3*s**2 - 2*s**3,           # v2
                                (x_in_elem - x1)*(s**2-s)  # theta2
                            ])
                            fe += n_vec * w * gw * (load_len / 2)

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
                node_i = int(sup['id'])
                if node_i >= num_nodes: continue
                
                stype = sup['type']
                if stype in ['Pin', 'Roller', 'Fixed']:
                    free_dof[2*node_i] = False # Fix Y
                if stype == 'Fixed':
                    free_dof[2*node_i+1] = False # Fix Rotation
            except: pass
        
        # --- 6. Solve System ---
        U = np.zeros(dof)
        
        # Check stability
        if np.sum(free_dof) < dof: # Only solve if constrained
            try:
                K_reduced = K[np.ix_(free_dof, free_dof)]
                F_reduced = F[free_dof]
                # Solve
                U_reduced = solve(K_reduced, F_reduced)
                U[free_dof] = U_reduced
            except np.linalg.LinAlgError:
                print("Error: Structure is unstable (Singular Matrix)")
                return pd.DataFrame(), [], None
        
        # Calculate Reactions: R = K*U - F_applied
        R_vector = K @ U - F 
        
        # --- 7. Post-Processing (Hybrid Method) ---
        total_len = nodes[-1]
        x_eval = np.linspace(0, total_len, 500)
        
        results = []
        
        for x in x_eval:
            # --- A. Deflection (FEM Interpolation) ---
            deflection = 0.0
            elem_idx = -1
            for idx, elem in enumerate(elements):
                if nodes[elem['n1']] <= x <= nodes[elem['n2']] + 1e-6:
                    elem_idx = idx
                    break
            
            if elem_idx != -1:
                n1, n2 = elements[elem_idx]['n1'], elements[elem_idx]['n2']
                x1, x2 = nodes[n1], nodes[n2]
                L_el = x2 - x1
                if L_el > 0:
                    s = (x - x1) / L_el
                    u_local = U[[2*n1, 2*n1+1, 2*n2, 2*n2+1]]
                    N = np.array([1 - 3*s**2 + 2*s**3, (x - x1) * (1 - s)**2, 3*s**2 - 2*s**3, (x - x1) * (s**2 - s)])
                    deflection = np.dot(N, u_local)

            # --- B. Shear & Moment (Statics Integration) ---
            V_x = 0.0
            M_x = 0.0
            
            # 1. Effect of Reactions (Left of x)
            for i, node_x in enumerate(nodes):
                if node_x <= x + 1e-5:
                    Ry = R_vector[2*i]
                    Mz = R_vector[2*i+1] # Reaction Moment
                    V_x += Ry
                    M_x += Ry * (x - node_x) + Mz 
            
            # 2. Effect of Applied Loads (Left of x)
            for _, load in self.loads_df.iterrows():
                lx = load['x']
                mag = load['mag']
                ltype = load['type']
                
                if ltype == 'P':
                    if lx <= x + 1e-5:
                        V_x -= mag
                        M_x -= mag * (x - lx)
                        
                elif ltype == 'M':
                     if lx <= x + 1e-5:
                         # Applied Moment (Clockwise is usually negative in this sign convention if using Reaction logic)
                         # Adjust based on your specific sign convention requirement
                         M_x -= mag 
                         
                elif ltype == 'U':
                    start = lx
                    dist = load['dist']
                    end = start + dist
                    
                    if start < x:
                        active_end = min(x, end)
                        dist_cover = active_end - start
                        if dist_cover > 0:
                            force = mag * dist_cover
                            centroid = start + dist_cover/2
                            moment_arm = x - centroid
                            
                            V_x -= force
                            M_x -= force * moment_arm
            
            results.append({'x': x, 'deflection': deflection, 'shear': V_x, 'moment': M_x})

        return pd.DataFrame(results), R_vector, {}

    def _discretize_model(self):
        # Create nodes at Supports and Load Points
        x_points = {0.0}
        current_x = 0.0
        # Add Span ends
        for s in self.spans:
            current_x += s
            x_points.add(round(current_x, 5))
            
        # Add Load points (Start and End of UDLs)
        for _, load in self.loads_df.iterrows():
            x_points.add(round(load['x'], 5))
            if load['type'] == 'U':
                x_points.add(round(load['x'] + load['dist'], 5))
                
        sorted_x = sorted(list(x_points))
        elements = [{'n1': i, 'n2': i+1} for i in range(len(sorted_x)-1)]
        return sorted_x, elements

    def _get_element_stiffness(self, L):
        E, I = self.E, self.I
        k = np.zeros((4,4))
        if L == 0: return k
        factor = E * I / (L**3)
        k[0,0] = 12;  k[0,1] = 6*L;    k[0,2] = -12;  k[0,3] = 6*L
        k[1,0] = 6*L; k[1,1] = 4*L**2; k[1,2] = -6*L; k[1,3] = 2*L**2
        k[2,0] = -12; k[2,1] = -6*L;   k[2,2] = 12;   k[2,3] = -6*L
        k[3,0] = 6*L; k[3,1] = 2*L**2; k[3,2] = -6*L; k[3,3] = 4*L**2
        return k * factor
