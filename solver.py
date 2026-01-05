import numpy as np
import pandas as pd
from scipy.linalg import solve

class BeamSolver:
    def __init__(self, spans, supports_df, loads_df, E, I, A=None, G=None):
        self.spans = spans
        self.supports_df = supports_df
        self.loads_df = loads_df
        self.E = E
        self.I = I
        
    def solve(self):
        # --- 1. Model Discretization (FEM Nodes) ---
        nodes, elements = self._discretize_model()
        num_nodes = len(nodes)
        dof = 2 * num_nodes
        
        K = np.zeros((dof, dof))
        F = np.zeros(dof)
        
        # --- 2. Stiffness Matrix Assembly ---
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
                    
        # --- 3. Force Vector Assembly ---
        # A. Nodal Loads (Directly at nodes)
        for _, load in self.loads_df.iterrows():
            # Find closest node match
            node_idx = -1
            for i, x in enumerate(nodes):
                if np.isclose(x, load['x'], atol=1e-5):
                    node_idx = i
                    break
            
            if node_idx != -1:
                # Apply strictly to global F vector if it aligns with a node
                if load['type'] == 'P':
                    F[2 * node_idx] -= load['mag'] 
                elif load['type'] == 'M':
                    F[2 * node_idx + 1] -= load['mag'] # Note: Mz convention might need check

        # B. Member Loads (Equivalent Nodal Forces)
        # This handles loads BETWEEN nodes for the FEM calculation
        for _, load in self.loads_df.iterrows():
            if load['type'] == 'U':
                start = load['x']
                end = start + load['dist']
                mag = load['mag']
                
                for elem in elements:
                    x1, x2 = nodes[elem['n1']], nodes[elem['n2']]
                    L_elem = x2 - x1
                    
                    # Check Overlap
                    overlap_start = max(start, x1)
                    overlap_end = min(end, x2)
                    
                    if overlap_end > overlap_start:
                        # Fixed End Forces Calculation for Partial UDL
                        # Simplified: Numerical Integration for Consistent Nodal Loads
                        a = overlap_start - x1
                        b = overlap_end - x1
                        w = -mag # Downward
                        
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

        # --- 4. Boundary Conditions ---
        free_dof = np.full(dof, True)
        for _, sup in self.supports_df.iterrows():
            node_i = sup['id']
            stype = sup['type']
            if stype in ['Pin', 'Roller', 'Fixed']:
                free_dof[2*node_i] = False # Fix Y
            if stype == 'Fixed':
                free_dof[2*node_i+1] = False # Fix Rotation
        
        # --- 5. Solve System ---
        U = np.zeros(dof)
        if np.any(free_dof):
            try:
                K_reduced = K[np.ix_(free_dof, free_dof)]
                F_reduced = F[free_dof]
                U_reduced = solve(K_reduced, F_reduced)
                U[free_dof] = U_reduced
            except:
                return pd.DataFrame(), [], None
        
        # Calculate Reactions: R = K*U - F_applied
        # Note: We must use the 'F' that includes Equivalent Nodal Forces
        R_vector = K @ U - F 
        
        # --- 6. Post-Processing (The "Exact" Method) ---
        # Use Statics/Method of Sections for V and M to get perfect curves
        # Use FEM Shape Functions for Deflection
        
        total_len = nodes[-1]
        x_eval = np.linspace(0, total_len, 500) # High resolution for smooth curves
        
        results = []
        
        for x in x_eval:
            # --- A. Deflection (from FEM interpolation) ---
            # Find which element x belongs to
            elem_idx = -1
            for idx, elem in enumerate(elements):
                if nodes[elem['n1']] <= x <= nodes[elem['n2']] + 1e-9:
                    elem_idx = idx
                    break
            
            if elem_idx != -1:
                n1, n2 = elements[elem_idx]['n1'], elements[elem_idx]['n2']
                x1, x2 = nodes[n1], nodes[n2]
                L_el = x2 - x1
                s = (x - x1) / L_el if L_el > 0 else 0
                
                # Nodal Displacements
                u_local = U[[2*n1, 2*n1+1, 2*n2, 2*n2+1]]
                
                # Shape Functions for v(x)
                N = np.array([
                    1 - 3*s**2 + 2*s**3,
                    (x - x1) * (1 - s)**2,
                    3*s**2 - 2*s**3,
                    (x - x1) * (s**2 - s)
                ])
                deflection = np.dot(N, u_local)
            else:
                deflection = 0.0

            # --- B. Shear & Moment (from Global Statics / Method of Sections) ---
            # V(x) = Sum of Vertical Forces to the left (including reactions)
            # M(x) = Sum of Moments caused by forces to the left
            
            V_x = 0.0
            M_x = 0.0
            
            # 1. Effect of Supports (Reactions) to the left
            for i, node_x in enumerate(nodes):
                if node_x <= x + 1e-5: # Include if on left or at current point
                    # Vertical Reaction
                    Ry = R_vector[2*i]
                    # Moment Reaction (External Moment at support)
                    Mz = R_vector[2*i+1]
                    
                    V_x += Ry
                    M_x += Ry * (x - node_x) - Mz # Sign convention: Sagging Positive
            
            # 2. Effect of Applied Loads to the left
            for _, load in self.loads_df.iterrows():
                lx = load['x']
                
                if load['type'] == 'P':
                    if lx <= x: # Point load is to the left
                        V_x -= load['mag'] # Downward load reduces Shear (Left Up Positive)
                        M_x -= load['mag'] * (x - lx)
                        
                elif load['type'] == 'M':
                    if lx <= x:
                         # Applied Moment. 
                         # Convention: Clockwise External Moment jumps Moment Graph UP?
                         # Let's stick to: Internal M = Sum Moments Left. 
                         # Clockwise Load = Negative Moment contribution?
                         # Standard: M (internal) jumps UP for CW applied moment.
                         M_x += load['mag'] 
                         
                elif load['type'] == 'U':
                    # UDL handling
                    start = lx
                    end = lx + load['dist']
                    
                    if start < x:
                        # Determine active length of UDL to the left of x
                        active_end = min(x, end)
                        dist_cover = active_end - start
                        
                        if dist_cover > 0:
                            force = load['mag'] * dist_cover
                            centroid = start + dist_cover/2
                            moment_arm = x - centroid
                            
                            V_x -= force
                            M_x -= force * moment_arm
            
            results.append({
                'x': x,
                'deflection': deflection,
                'shear': V_x,
                'moment': M_x
            })

        return pd.DataFrame(results), R_vector, {}

    def _discretize_model(self):
        # Create nodes at Supports and Load Points to ensure accuracy
        x_points = {0.0}
        current_x = 0.0
        for s in self.spans:
            current_x += s
            x_points.add(round(current_x, 5))
            
        for _, load in self.loads_df.iterrows():
            x_points.add(round(load['x'], 5))
            if load['type'] == 'U':
                x_points.add(round(load['x'] + load['dist'], 5))
                
        sorted_x = sorted(list(x_points))
        # Map indices
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
