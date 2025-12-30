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
        # 1. Initialize Stiffness Matrix & Force Vector
        n_dof = 2 * self.n_nodes 
        K_global = np.zeros((n_dof, n_dof))
        F_global = np.zeros(n_dof)
        
        for i, L in enumerate(self.spans):
            # Element Stiffness Matrix
            k_val = (self.E * self.I / L**3)
            k_el = k_val * np.array([
                [12, 6*L, -12, 6*L],
                [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L],
                [6*L, 2*L**2, -6*L, 4*L**2]
            ])
            
            # Map to Global Indices
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            for r in range(4):
                for c in range(4):
                    K_global[idx[r], idx[c]] += k_el[r, c]
            
            # Calculate Fixed End Moments (FEM)
            fem = np.zeros(4)
            span_loads = [l for l in self.loads if int(l['span_idx']) == i]
            
            for load in span_loads:
                val = load['mag']
                
                if load['type'] == 'P':
                    # Point Load
                    a = load['x']; b = L - a; P = val
                    fem[0] += P * b**2 * (3*a + b) / L**3  
                    fem[1] += P * a * b**2 / L**2          
                    fem[2] += P * a**2 * (a + 3*b) / L**3  
                    fem[3] -= P * a**2 * b / L**2          
                    
                elif load['type'] == 'U':
                    # UDL
                    w = val
                    fem[0] += w * L / 2
                    fem[1] += w * L**2 / 12
                    fem[2] += w * L / 2
                    fem[3] -= w * L**2 / 12
                    
                elif load['type'] == 'M':
                    # Moment Load
                    M_app = val 
                    a = load['x']; b = L - a
                    
                    # FEM Formulas for Moment Load
                    # V_A = -6*M*a*b/L^3, V_B = 6*M*a*b/L^3
                    # M_A = M*b*(2a-b)/L^2, M_B = M*a*(2b-a)/L^2
                    
                    fem[0] += -6 * M_app * a * b / L**3
                    fem[1] += M_app * b * (2*a - b) / L**2  
                    fem[2] += 6 * M_app * a * b / L**3
                    # *** FIXED SYNTAX ERROR HERE (2b -> 2*b) ***
                    fem[3] += M_app * a * (2*b - a) / L**2   
            
            F_global[idx] -= fem 

        # 2. Boundary Conditions
        free_dofs = list(range(n_dof))
        for _, row in self.supports.iterrows():
            node_idx = int(row['id'])
            # Fix Vertical Translation (y)
            if 2*node_idx in free_dofs: 
                free_dofs.remove(2*node_idx) 
            
            # Fix Rotation (theta) if Fixed Support
            if row['type'] == 'Fixed':
                if 2*node_idx+1 in free_dofs: 
                    free_dofs.remove(2*node_idx+1)

        # 3. Solve System
        U_global = np.zeros(n_dof)
        if free_dofs:
            K_ff = K_global[np.ix_(free_dofs, free_dofs)]
            F_f = F_global[free_dofs]
            try:
                u_solved = np.linalg.solve(K_ff, F_f)
                U_global[free_dofs] = u_solved
            except np.linalg.LinAlgError:
                return pd.DataFrame(), np.zeros(n_dof)

        # 4. Post-Processing
        results = []
        
        for i, L in enumerate(self.spans):
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            u_el = U_global[idx]
            
            fem_local = np.zeros(4)
            span_loads = [l for l in self.loads if int(l['span_idx']) == i]
            
            eval_points = set(np.linspace(0, L, 100)) 
            
            for load in span_loads:
                val = load['mag']
                if load['type'] == 'P':
                    a = load['x']; b = L - a; P = val
                    fem_local[0] += P*b**2*(3*a+b)/L**3; fem_local[1] += P*a*b**2/L**2
                    fem_local[2] += P*a**2*(a+3*b)/L**3; fem_local[3] -= P*a**2*b/L**2
                    eval_points.add(a); 
                    if a>0: eval_points.add(a-1e-6)
                    if a<L: eval_points.add(a+1e-6)
                    
                elif load['type'] == 'U':
                    w = val
                    fem_local[0] += w*L/2; fem_local[1] += w*L**2/12
                    fem_local[2] += w*L/2; fem_local[3] -= w*L**2/12
                    
                elif load['type'] == 'M':
                    a = load['x']; b = L - a; M_app = val
                    fem_local[0] += -6 * M_app * a * b / L**3
                    fem_local[1] += M_app * b * (2*a - b) / L**2
                    fem_local[2] += 6 * M_app * a * b / L**3
                    fem_local[3] += M_app * a * (2*b - a) / L**2
                    eval_points.add(a); 
                    if a>0: eval_points.add(a-1e-6)
                    if a<L: eval_points.add(a+1e-6)

            x_eval = sorted(list(eval_points))
            
            k_el = (self.E * self.I / L**3) * np.array([
                [12, 6*L, -12, 6*L], [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L], [6*L, 2*L**2, -6*L, 4*L**2]
            ])
            f_nodal = k_el @ u_el + fem_local
            
            V_start = f_nodal[0]
            M_start = f_nodal[1] 
            
            M_vals = []
            V_vals = []
            
            for x in x_eval:
                V_x = V_start
                M_x = -M_start + V_start * x 
                
                for load in span_loads:
                    val = load['mag']
                    if load['type'] == 'P':
                        if x >= load['x'] + 1e-9: 
                            V_x -= val
                            M_x -= val * (x - load['x'])
                    elif load['type'] == 'U':
                        if x > 0:
                            d = x 
                            V_x -= val * d
                            M_x -= (val * d) * (d / 2)
                    elif load['type'] == 'M':
                        if x >= load['x'] + 1e-9:
                            M_x += val 
                            
                V_vals.append(V_x)
                M_vals.append(M_x)
            
            # Deflection Calculation
            M_arr = np.array(M_vals)
            curv = M_arr / (self.E * self.I)
            
            if hasattr(integrate, 'cumulative_trapezoid'):
                trapz = integrate.cumulative_trapezoid
            else:
                trapz = integrate.cumtrapz
            
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
        
        # 5. Snap to Zero (Cleanup)
        df_res = pd.DataFrame(results)
        
        tol = 1e-5
        if not df_res.empty:
            df_res.loc[df_res['moment'].abs() < tol, 'moment'] = 0.0
            df_res.loc[df_res['shear'].abs() < tol, 'shear'] = 0.0
            df_res.loc[df_res['deflection'].abs() < 1e-9, 'deflection'] = 0.0
        
        return df_res, Reactions
