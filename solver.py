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
                    
                elif load['type'] == 'M':
                    # Moment Load (Concentrated Moment)
                    # Input Convention: Positive = Clockwise (CW)
                    # FEM Formulas for CW Moment 'M' at distance 'a':
                    # M_L = M * b * (2a - b) / L^2
                    # M_R = M * a * (2b - a) / L^2
                    # Vertical forces balance the moments
                    
                    M_app = val
                    a = load['x']
                    b = L - a
                    
                    # Nodal Moments (Reaction Moments at fixed ends)
                    m_fix_left = M_app * b * (2*a - b) / L**2
                    m_fix_right = M_app * a * (2*b - a) / L**2
                    
                    # Vertical reactions to balance these moments + applied moment
                    # Sum M_B = 0 -> R_A * L + M_fix_left + M_fix_right - M_app (careful with signs)
                    # Let's use standard coefficients for nodal forces vector directly:
                    
                    # Force Vector {Fy1, M1, Fy2, M2} for CW Moment M at a:
                    # Fy1 = -6*M*a*b / L^3
                    # M1  = b*(2*a - b)*M / L^2
                    # Fy2 = 6*M*a*b / L^3
                    # M2  = a*(2*b - a)*M / L^2
                    
                    fem[0] += -6 * M_app * a * b / L**3
                    fem[1] += M_app * b * (2*a - b) / L**2
                    fem[2] += 6 * M_app * a * b / L**3
                    fem[3] += M_app * a * (2*b - a) / L**2

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

        # 4. Post-Processing
        results = []
        
        for i, L in enumerate(self.spans):
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            u_el = U_global[idx]
            
            # Recover Local Start Forces (need to subtract FEM again)
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
                    
                    # Critical points for Moment Load (Jump in BMD)
                    eval_points.add(a)
                    if a>0: eval_points.add(a-1e-6)
                    if a<L: eval_points.add(a+1e-6)
            
            x_eval = sorted(list(eval_points))
            
            k_el = (self.E * self.I / L**3) * np.array([
                [12, 6*L, -12, 6*L], [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L], [6*L, 2*L**2, -6*L, 4*L**2]
            ])
            f_local = k_el @ u_el + fem_local
            V_start = f_local[0]
            M_start = f_local[1] # CCW is Positive in solver vector, but usually we plot Sagging as +

            M_vals = []
            V_vals = []
            
            for x in x_eval:
                V_x = V_start
                # Initial Moment at x from start forces
                # Note: M_start from matrix is Nodal Moment (CCW+). 
                # Internal Moment convention: Sagging +. 
                # If we cut at x: M_internal = -M_node + V_node*x ...
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
                        # Applied Moment M (CW)
                        # Equilibrium: M_internal + M_app (if x>a) + ... = 0
                        # Thus M_internal reduces by M_app? 
                        # Standard Jump: CW Moment creates Positive Jump in BMD (Sagging increases)
                        if x >= load['x'] + 1e-9:
                            M_x += val # Add CW moment value creates upward jump
                            
                M_vals.append(M_x)
                V_vals.append(V_x)
            
            # Deflection
            M_vals = np.array(M_vals)
            curv = M_vals / (self.E * self.I)
            
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
        return pd.DataFrame(results), Reactions
