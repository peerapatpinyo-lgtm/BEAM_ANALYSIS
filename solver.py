import numpy as np
import pandas as pd

class BeamSolver:
    def __init__(self, spans, supports, loads, E=2e6, I=1e-4):
        self.spans = spans
        # supports: DataFrame with 'id' and 'type'
        self.supports = supports 
        self.loads = loads
        self.E = E
        self.I = I
        self.nodes = [0] + list(np.cumsum(spans))
        self.n_nodes = len(self.nodes)
        
    def solve(self):
        n_dof = 2 * self.n_nodes 
        K_global = np.zeros((n_dof, n_dof))
        F_global = np.zeros(n_dof)
        
        # --- 1. Stiffness Matrix Assembly ---
        for i, L in enumerate(self.spans):
            # Element Stiffness
            k_el = (self.E * self.I / L**3) * np.array([
                [12, 6*L, -12, 6*L],
                [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L],
                [6*L, 2*L**2, -6*L, 4*L**2]
            ])
            
            # Add to Global K
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            for r in range(4):
                for c in range(4):
                    K_global[idx[r], idx[c]] += k_el[r, c]
            
            # FEM (Fixed End Moments) Calculation
            fem = np.zeros(4)
            span_loads = [l for l in self.loads if int(l['span_idx']) == i]
            
            for load in span_loads:
                if load['type'] == 'P':
                    a = load['x']
                    b = L - a
                    P = load['P']
                    L2, L3 = L**2, L**3
                    fem[0] += P * b**2 * (3*a + b) / L3
                    fem[1] += P * a * b**2 / L2
                    fem[2] += P * a**2 * (a + 3*b) / L3
                    fem[3] -= P * a**2 * b / L2
                elif load['type'] == 'U':
                    w = load['w']
                    fem[0] += w * L / 2
                    fem[1] += w * L**2 / 12
                    fem[2] += w * L / 2
                    fem[3] -= w * L**2 / 12
            
            # Subtract FEM from Global Force Vector
            F_global[idx] -= fem 

        # --- 2. Boundary Conditions ---
        free_dofs = list(range(n_dof))
        
        for _, row in self.supports.iterrows():
            node_idx = int(row['id'])
            # Fix Vertical Y (Even index)
            if 2*node_idx in free_dofs: free_dofs.remove(2*node_idx)
            
            # Fix Rotation (Odd index) if Fixed
            if row['type'] == 'Fixed':
                if 2*node_idx+1 in free_dofs: free_dofs.remove(2*node_idx+1)

        # --- 3. Solve Equations ---
        U_global = np.zeros(n_dof)
        if free_dofs:
            K_ff = K_global[np.ix_(free_dofs, free_dofs)]
            F_f = F_global[free_dofs]
            try:
                u_solved = np.linalg.solve(K_ff, F_f)
                U_global[free_dofs] = u_solved
            except np.linalg.LinAlgError:
                # Unstable structure
                return pd.DataFrame(), np.zeros(n_dof)

        # Calculate Reactions: R = K*u - F_external (Here F_ext is just FEM related in opposite)
        # Actually simpler: R = K_global * U_global + FEM_global
        # But we will re-calculate element forces node by node for better precision below.
        
        # --- 4. Post-Processing (Internal Forces) ---
        results = []
        
        for i, L in enumerate(self.spans):
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            u_el = U_global[idx]
            
            # Re-calculate FEM for this element to get member end forces
            fem_local = np.zeros(4)
            span_loads = [l for l in self.loads if int(l['span_idx']) == i]
            
            for load in span_loads:
                if load['type'] == 'P':
                    a = load['x']; b = L - a; P = load['P']
                    fem_local[0] += P * b**2 * (3*a + b) / L**3
                    fem_local[1] += P * a * b**2 / L**2
                    fem_local[2] += P * a**2 * (a + 3*b) / L**3
                    fem_local[3] -= P * a**2 * b / L**2
                elif load['type'] == 'U':
                    w = load['w']
                    fem_local[0] += w * L / 2; fem_local[1] += w * L**2 / 12
                    fem_local[2] += w * L / 2; fem_local[3] -= w * L**2 / 12
            
            # Stiffness matrix again
            k_el = (self.E * self.I / L**3) * np.array([
                [12, 6*L, -12, 6*L],
                [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L],
                [6*L, 2*L**2, -6*L, 4*L**2]
            ])
            
            # Member End Forces (Forces exerted by nodes ON the beam)
            # f_local = k * u + fem
            f_local = k_el @ u_el + fem_local
            
            # Start of Element Forces
            V_start = f_local[0]   # Shear at left end (Up = +)
            M_start = f_local[1]   # Moment at left end (CCW = +)
            
            # Create dense mesh for plotting
            x_eval = np.linspace(0, L, 50)
            for l in span_loads:
                if l['type'] == 'P':
                    # Add point exactly at load and slightly around it for vertical jump
                    x_eval = np.append(x_eval, [l['x'] - 1e-5, l['x'], l['x'] + 1e-5])
            x_eval = np.sort(np.unique(x_eval))
            x_eval = x_eval[x_eval >= 0]
            x_eval = x_eval[x_eval <= L]
            
            # Method of Sections at distance x
            for x in x_eval:
                # 1. Effect from Start Node
                # V(x) starts with V_start
                # M(x) convention: Sagging is positive.
                # If M_start is CCW (+), it hogs the beam (negative moment).
                # So initial moment = -M_start.
                # Force V_start (Up) causes Sagging (+) at distance x.
                
                V_x = V_start
                M_x = -M_start + V_start * x
                
                # 2. Effect from Loads within the span up to x
                for load in span_loads:
                    if load['type'] == 'P':
                        if x > load['x']: # Passed the load
                            V_x -= load['P']
                            M_x -= load['P'] * (x - load['x'])
                    elif load['type'] == 'U':
                        if x > 0:
                            # UDL covers from 0 to x
                            w = load['w']
                            V_x -= w * x
                            M_x -= (w * x) * (x / 2)
                
                results.append({
                    'x': self.nodes[i] + x, # Global X
                    'shear': V_x,
                    'moment': M_x
                })

        # Calculate Reactions for display
        # We can sum up nodal forces from the Element calculations to be precise
        Reactions = np.zeros(n_dof)
        # Re-loop to sum assembly forces (Internal forces at nodes)
        # Or simpler: R = K * U - F_ext (applied loads)
        # Note: F_global construction earlier subtracted FEM.
        # R = K_global @ U_global matches the external forces required.
        # But F_global was (Load - FEM). 
        # Correct Reaction Calculation:
        Reactions = K_global @ U_global
        # Now we need to subtract the equivalent nodal loads from member loads to get the actual reaction 
        # at the support.
        # Wait, the K*u gives the TOTAL force at the node.
        # This includes Reaction + Applied Nodal Load.
        # If we have no direct nodal loads (only member loads), Reactions = K*u - (Forces from Member Loads on Nodes)
        # Forces from Member Loads on Nodes = FEM.
        
        # Let's reconstruct Global FEM vector
        FEM_global = np.zeros(n_dof)
        for i, L in enumerate(self.spans):
            fem = np.zeros(4)
            span_loads = [l for l in self.loads if int(l['span_idx']) == i]
            for load in span_loads:
                if load['type'] == 'P':
                    a = load['x']; b = L - a; P = load['P']
                    fem[0] += P*b**2*(3*a+b)/L**3; fem[1] += P*a*b**2/L**2
                    fem[2] += P*a**2*(a+3*b)/L**3; fem[3] -= P*a**2*b/L**2
                elif load['type'] == 'U':
                    w = load['w']
                    fem[0] += w*L/2; fem[1] += w*L**2/12
                    fem[2] += w*L/2; fem[3] -= w*L**2/12
            
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            FEM_global[idx] += fem
            
        final_reactions = K_global @ U_global + FEM_global
        
        return pd.DataFrame(results), final_reactions
