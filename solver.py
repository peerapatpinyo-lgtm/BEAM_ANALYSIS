import numpy as np
import pandas as pd

class BeamSolver:
    def __init__(self, spans, supports, loads, E=2e6, I=1e-4):
        self.spans = spans
        # supports format: DataFrame with 'id' (node index) and 'type'
        self.supports = supports 
        self.loads = loads
        self.E = E
        self.I = I
        self.nodes = [0] + list(np.cumsum(spans))
        self.n_nodes = len(self.nodes)
        
    def solve(self):
        n_dof = 2 * self.n_nodes # 2 DOF per node (Vertical Y, Rotation Theta)
        K_global = np.zeros((n_dof, n_dof))
        F_global = np.zeros(n_dof)
        
        # 1. Build Global Stiffness Matrix & Load Vector
        for i, L in enumerate(self.spans):
            # Element Stiffness Matrix (4x4)
            k_el = (self.E * self.I / L**3) * np.array([
                [12, 6*L, -12, 6*L],
                [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L],
                [6*L, 2*L**2, -6*L, 4*L**2]
            ])
            
            # Assembly indices
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            for r in range(4):
                for c in range(4):
                    K_global[idx[r], idx[c]] += k_el[r, c]
            
            # Equivalent Nodal Loads (Fixed End Forces)
            fem = np.zeros(4)
            # Filter loads in this span
            span_loads = [l for l in self.loads if int(l['span_idx']) == i]
            
            for load in span_loads:
                if load['type'] == 'P':
                    a = load['x']
                    b = L - a
                    P = load['P']
                    # Fixed End Actions
                    fem[0] += P * b**2 * (3*a + b) / L**3  # Fy1
                    fem[1] += P * a * b**2 / L**2          # M1
                    fem[2] += P * a**2 * (a + 3*b) / L**3  # Fy2
                    fem[3] -= P * a**2 * b / L**2          # M2 (Negative convention)
                elif load['type'] == 'U':
                    w = load['w']
                    fem[0] += w * L / 2
                    fem[1] += w * L**2 / 12
                    fem[2] += w * L / 2
                    fem[3] -= w * L**2 / 12
            
            # Add FEM to Global Force (Subtract because F = K*u + FEM -> K*u = F_ext - FEM)
            F_global[idx] += fem 

        # 2. Apply Boundary Conditions
        free_dofs = list(range(n_dof))
        
        for _, row in self.supports.iterrows():
            node_idx = int(row['id'])
            # Fix Vertical (Y)
            if 2*node_idx in free_dofs: free_dofs.remove(2*node_idx)
            
            # Fix Rotation if 'Fixed'
            if row['type'] == 'Fixed':
                if 2*node_idx+1 in free_dofs: free_dofs.remove(2*node_idx+1)

        # 3. Solve for Displacements
        U_global = np.zeros(n_dof)
        if free_dofs:
            K_ff = K_global[np.ix_(free_dofs, free_dofs)]
            F_f = -F_global[free_dofs] # Loads are applied opposite to FEM
            try:
                u_solved = np.linalg.solve(K_ff, F_f)
                U_global[free_dofs] = u_solved
            except np.linalg.LinAlgError:
                return pd.DataFrame(), np.zeros(n_dof)

        # 4. Compute Internal Forces (Shear/Moment) using Shape Functions
        # This guarantees clean curves and continuity
        results = []
        
        for i, L in enumerate(self.spans):
            # Nodal displacements for this element
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            u_el = U_global[idx]
            
            # Create evaluation points (dense mesh for smooth curves)
            # Add specific points for Point Loads to get sharp changes
            x_eval = np.linspace(0, L, 50)
            
            span_loads = [l for l in self.loads if int(l['span_idx']) == i]
            for l in span_loads:
                if l['type'] == 'P':
                    # Add point just before and after load
                    x_eval = np.append(x_eval, [l['x'] - 1e-5, l['x'], l['x'] + 1e-5])
            
            x_eval = np.sort(np.unique(x_eval))
            
            for x in x_eval:
                # Shape Function Derivatives for V and M
                # v(x) = N * u
                # M(x) = EI * v''(x)
                # V(x) = EI * v'''(x) -> constant for beam, but we need to account for loads
                
                # ...Actually, easier way for plotting:
                # Calculate internal forces using basic equilibrium at distance x
                # Starting from Left Node Forces:
                
                # Element End Forces from Stiffness: f = k*u + fem
                k_el = (self.E * self.I / L**3) * np.array([
                    [12, 6*L, -12, 6*L],
                    [6*L, 4*L**2, -6*L, 2*L**2],
                    [-12, -6*L, 12, -6*L],
                    [6*L, 2*L**2, -6*L, 4*L**2]
                ])
                
                # Recalculate FEM specific to this element for end forces
                fem_local = np.zeros(4)
                for load in span_loads:
                    if load['type'] == 'P':
                        a = load['x']; b = L - a; P = load['P']
                        fem_local[0] += P*b**2*(3*a+b)/L**3
                        fem_local[1] += P*a*b**2/L**2
                        fem_local[2] += P*a**2*(a+3*b)/L**3
                        fem_local[3] -= P*a**2*b/L**2
                    elif load['type'] == 'U':
                        w = load['w']
                        fem_local[0] += w*L/2; fem_local[1] += w*L**2/12
                        fem_local[2] += w*L/2; fem_local[3] -= w*L**2/12
                
                f_local = k_el @ u_el + fem_local
                
                # f_local = [Fy_start, M_start, Fy_end, M_end]
                # Walk from left (x=0) to current x
                V_x = f_local[0] # Start with left reaction
                M_x = -f_local[1] # Start with left moment (note sign convention)
                
                # Subtract loads found before x
                for load in span_loads:
                    if load['type'] == 'P':
                        if x >= load['x']:
                            V_x -= load['P']
                            M_x += load['P'] * (x - load['x']) # Moment reduces (hogging)
                    elif load['type'] == 'U':
                         # UDL active length
                         if x > 0:
                             w = load['w']
                             len_act = x
                             V_x -= w * len_act
                             M_x += w * len_act * (len_act / 2)
                
                # M_x from reaction
                M_x -= V_x_initial_reaction(f_local[0]) * x  ---> Wait, simplier:
                
                # Let's use strict Statics from the left node of the element
                V_calc = f_local[0]
                M_calc = f_local[1] # Counter-Clockwise is positive
                
                # Shear at x: V(x) = Ry_left - Sum(Loads)
                # Moment at x: M(x) = M_left + Ry_left*x - Sum(Load * arm)
                
                # Re-loop loads for precise x
                current_V = V_calc
                current_M = -M_calc + V_calc * x # Standard Beam Convention: M(+) sags
                
                for load in span_loads:
                    if load['type'] == 'P':
                        if x > load['x'] + 1e-6: # Past the load
                            current_V -= load['P']
                            current_M -= load['P'] * (x - load['x'])
                    elif load['type'] == 'U':
                        w = load['w']
                        current_V -= w * x
                        current_M -= (w * x) * (x/2)
                
                results.append({
                    'x': self.nodes[i] + x, # Global coordinate
                    'shear': current_V,
                    'moment': current_M
                })
                
        # Calculate Reactions (K*u - F_ext_nodes)
        Reactions = K_global @ U_global # Note: Need to handle nodal loads if any
        # Simplified: Just return the computed values at supports from f_local logic if needed
        # But for global reaction display:
        
        return pd.DataFrame(results), Reactions
