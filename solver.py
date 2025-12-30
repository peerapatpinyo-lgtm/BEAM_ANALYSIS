import numpy as np
import pandas as pd
from scipy import integrate

class BeamSolver:
    def __init__(self, spans, supports, loads, E=2e6, I=1e-4):
        self.spans = spans
        self.supports = supports 
        self.loads = loads
        self.E = E
        self.I = I
        self.nodes = [0] + list(np.cumsum(spans))
        self.n_nodes = len(self.nodes)
        
    def solve(self):
        # 1. Global Stiffness Setup
        n_dof = 2 * self.n_nodes 
        K_global = np.zeros((n_dof, n_dof))
        F_global = np.zeros(n_dof)
        
        for i, L in enumerate(self.spans):
            # Element Stiffness
            k_el = (self.E * self.I / L**3) * np.array([
                [12, 6*L, -12, 6*L],
                [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L],
                [6*L, 2*L**2, -6*L, 4*L**2]
            ])
            
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            for r in range(4):
                for c in range(4):
                    K_global[idx[r], idx[c]] += k_el[r, c]
            
            # FEM (Fixed End Moments)
            fem = np.zeros(4)
            span_loads = [l for l in self.loads if int(l['span_idx']) == i]
            
            for load in span_loads:
                # Map keys from input_handler (mag -> P/w)
                val = load['mag']
                
                if load['type'] == 'P':
                    a = load['x']; b = L - a; P = val
                    fem[0] += P * b**2 * (3*a + b) / L**3
                    fem[1] += P * a * b**2 / L**2
                    fem[2] += P * a**2 * (a + 3*b) / L**3
                    fem[3] -= P * a**2 * b / L**2
                elif load['type'] == 'U':
                    w = val
                    fem[0] += w * L / 2
                    fem[1] += w * L**2 / 12
                    fem[2] += w * L / 2
                    fem[3] -= w * L**2 / 12
            
            F_global[idx] -= fem 

        # 2. Boundary Conditions
        free_dofs = list(range(n_dof))
        for _, row in self.supports.iterrows():
            node_idx = int(row['id'])
            if 2*node_idx in free_dofs: free_dofs.remove(2*node_idx) 
            if row['type'] == 'Fixed':
                if 2*node_idx+1 in free_dofs: free_dofs.remove(2*node_idx+1)

        # 3. Solve
        U_global = np.zeros(n_dof)
        if free_dofs:
            K_ff = K_global[np.ix_(free_dofs, free_dofs)]
            F_f = F_global[free_dofs]
            try:
                u_solved = np.linalg.solve(K_ff, F_f)
                U_global[free_dofs] = u_solved
            except np.linalg.LinAlgError:
                return pd.DataFrame(), np.zeros(n_dof)

        # 4. Post-Processing with Critical Points (Fix Slanted Graph)
        results = []
        
        for i, L in enumerate(self.spans):
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            u_el = U_global[idx]
            
            # Recover Local Start Forces
            fem_local = np.zeros(4)
            span_loads = [l for l in self.loads if int(l['span_idx']) == i]
            
            # Create Evaluation Points (Mesh)
            # Add specific points where loads occur to prevent "slanted" shear graphs
            eval_points = set(np.linspace(0, L, 100)) # Base resolution
            
            for load in span_loads:
                val = load['mag']
                if load['type'] == 'P':
                    a = load['x']; b = L - a; P = val
                    fem_local[0] += P*b**2*(3*a+b)/L**3; fem_local[1] += P*a*b**2/L**2
                    fem_local[2] += P*a**2*(a+3*b)/L**3; fem_local[3] -= P*a**2*b/L**2
                    
                    # Add Critical Points for Graphing (Before, At, After)
                    eval_points.add(a)
                    if a > 0: eval_points.add(a - 1e-6)
                    if a < L: eval_points.add(a + 1e-6)
                    
                elif load['type'] == 'U':
                    w = val
                    fem_local[0] += w*L/2; fem_local[1] += w*L**2/12
                    fem_local[2] += w*L/2; fem_local[3] -= w*L**2/12
            
            # Sort points
            x_eval = sorted(list(eval_points))
            
            # Calculate Start Forces
            k_el = (self.E * self.I / L**3) * np.array([
                [12, 6*L, -12, 6*L], [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L], [6*L, 2*L**2, -6*L, 4*L**2]
            ])
            f_local = k_el @ u_el + fem_local
            V_start = f_local[0]
            M_start = f_local[1]

            # Calculate V, M along beam
            M_vals = []
            V_vals = []
            
            for x in x_eval:
                V_x = V_start
                M_x = -M_start + V_start * x
                
                for load in span_loads:
                    val = load['mag']
                    if load['type'] == 'P':
                        # Use a tiny tolerance for step function
                        if x >= load['x'] + 1e-9: 
                            V_x -= val
                            M_x -= val * (x - load['x'])
                    elif load['type'] == 'U':
                        if x > 0:
                            w = val
                            d = x
                            V_x -= w * d
                            M_x -= (w * d) * (d / 2)
                M_vals.append(M_x)
                V_vals.append(V_x)
            
            # Deflection (Double Integration)
            M_vals = np.array(M_vals)
            curv = M_vals / (self.E * self.I)
            
            if hasattr(integrate, 'cumulative_trapezoid'):
                trapz = integrate.cumulative_trapezoid
            else:
                trapz = integrate.cumtrapz
                
            # Note: Integration needs handling of duplicate x-coords (steps), 
            # but trapezoidal rule handles zero-width intervals safely (area=0)
            theta_vals = u_el[1] + trapz(curv, x_eval, initial=0)
            deflection_vals = u_el[0] + trapz(theta_vals, x_eval, initial=0)
            
            for k in range(len(x_eval)):
                results.append({
                    'x': self.nodes[i] + x_eval[k],
                    'shear': V_vals[k],
                    'moment': M_vals[k],
                    'deflection': deflection_vals[k]
                })

        Reactions = K_global @ U_global 
        return pd.DataFrame(results), Reactions
