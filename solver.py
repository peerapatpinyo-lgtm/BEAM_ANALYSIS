import numpy as np
import pandas as pd

class BeamSolver:
    def __init__(self, spans, supports, loads):
        self.spans = spans
        self.supports = supports
        self.loads = loads
        self.n_nodes = len(spans) + 1
        self.total_length = sum(spans)
        self.nodes = [0] + list(np.cumsum(spans))
        
    def solve(self):
        n_dof = 2 * self.n_nodes
        K_global = np.zeros((n_dof, n_dof))
        F_global = np.zeros(n_dof)
        
        # 1. Build Stiffness Matrix & FEM Loads
        for i, L in enumerate(self.spans):
            k = self._element_stiffness(L)
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            
            # Assembly K
            for r in range(4):
                for c in range(4):
                    K_global[idx[r], idx[c]] += k[r, c]
            
            # FEM Fixed End Actions
            fem = np.zeros(4)
            span_loads = [l for l in self.loads if int(l['span_idx']) == i]
            
            for load in span_loads:
                if load['type'] == 'P':
                    a, b = load['x'], L - load['x']
                    P = load['P']
                    # Reaction
                    fem[0] += P * b**2 * (3*a + b) / L**3
                    fem[2] += P * a**2 * (a + 3*b) / L**3
                    # Moment
                    fem[1] += P * a * b**2 / L**2
                    fem[3] -= P * a**2 * b / L**2
                elif load['type'] == 'U':
                    w = load['w']
                    fem[0] += w * L / 2
                    fem[2] += w * L / 2
                    fem[1] += w * L**2 / 12
                    fem[3] -= w * L**2 / 12
            
            F_global[idx] += fem

        # 2. Apply Boundary Conditions
        free_dofs = list(range(n_dof))
        fixed_dofs = []
        
        for _, sup in self.supports.iterrows():
            node_idx = int(sup['id'])
            # Y-constraint
            fixed_dofs.append(2*node_idx) 
            if sup['type'] == 'Fixed':
                # Rotation constraint
                fixed_dofs.append(2*node_idx + 1)
        
        # Unique and Sort
        fixed_dofs = sorted(list(set(fixed_dofs)))
        for d in sorted(fixed_dofs, reverse=True):
            if d in free_dofs: free_dofs.remove(d)
            
        # 3. Solve Displacements
        K_ff = K_global[np.ix_(free_dofs, free_dofs)]
        F_f = F_global[free_dofs]
        
        # Invert K_ff (Add small regularization for stability)
        try:
            U_f = np.linalg.solve(K_ff, F_f)
        except np.linalg.LinAlgError:
            U_f = np.zeros(len(free_dofs)) # Structure unstable

        U_global = np.zeros(n_dof)
        U_global[free_dofs] = U_f
        
        # 4. Calculate Internal Forces
        results = []
        step = 0.05 # Resolution
        
        for i, L in enumerate(self.spans):
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            u_ele = U_global[idx]
            
            x_vals = np.arange(0, L + step/2, step)
            
            # Element Stiffness
            k = self._element_stiffness(L)
            f_ele = k @ u_ele
            
            # Superposition with local loads
            span_loads = [l for l in self.loads if int(l['span_idx']) == i]
            
            for x in x_vals:
                # Forces from nodal displacements
                # V = dM/dx, M = EI * d2y/dx2 (Simplified via shape functions)
                # Standard beam theory approach:
                # V(x) = V_left + sum(Loads_left)
                # M(x) = M_left + V_left*x + sum(Moment_from_Loads)
                
                # Nodal forces (Sign convention: Up+, CCW+)
                V_start = f_ele[0]
                M_start = f_ele[1]
                
                V_x = V_start
                M_x = -M_start + V_start * x # Beam sign convention: Sagging Positive
                
                for load in span_loads:
                    if load['type'] == 'P':
                        if x >= load['x']:
                            V_x -= load['P']
                            M_x -= load['P'] * (x - load['x'])
                    elif load['type'] == 'U':
                        # Dist load up to x
                        w = load['w']
                        # Included length
                        wx = w * x 
                        V_x -= wx
                        M_x -= w * x**2 / 2
                
                results.append({
                    'x': self.nodes[i] + x,
                    'shear': V_x,
                    'moment': M_x
                })
                
        # Calculate Reactions
        Reactions = K_global @ U_global - F_global
        
        return pd.DataFrame(results), Reactions

    def _element_stiffness(self, L, E=1, I=1):
        k = (E * I / L**3) * np.array([
            [12, 6*L, -12, 6*L],
            [6*L, 4*L**2, -6*L, 2*L**2],
            [-12, -6*L, 12, -6*L],
            [6*L, 2*L**2, -6*L, 4*L**2]
        ])
        return k
