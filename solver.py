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
        # 1. Setup Global Stiffness Matrix
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
            
            # Assembly indices
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            for r in range(4):
                for c in range(4):
                    K_global[idx[r], idx[c]] += k_el[r, c]
            
            # Fixed End Moments (FEM) from Loads
            fem = np.zeros(4)
            span_loads = [l for l in self.loads if int(l['span_idx']) == i]
            
            for load in span_loads:
                if load['type'] == 'P':
                    a = load['x']; b = L - a; P = load['P']
                    # FEM Formula for Point Load
                    fem[0] += P * b**2 * (3*a + b) / L**3  # Fy1
                    fem[1] += P * a * b**2 / L**2          # M1
                    fem[2] += P * a**2 * (a + 3*b) / L**3  # Fy2
                    fem[3] -= P * a**2 * b / L**2          # M2
                elif load['type'] == 'U':
                    w = load['w']
                    # FEM Formula for UDL
                    fem[0] += w * L / 2
                    fem[1] += w * L**2 / 12
                    fem[2] += w * L / 2
                    fem[3] -= w * L**2 / 12
            
            F_global[idx] -= fem 

        # 2. Apply Boundary Conditions
        free_dofs = list(range(n_dof))
        for _, row in self.supports.iterrows():
            node_idx = int(row['id'])
            # Fix Vertical (Y)
            if 2*node_idx in free_dofs: 
                free_dofs.remove(2*node_idx) 
            
            # Fix Rotation (If Fixed Support)
            if row['type'] == 'Fixed':
                if 2*node_idx+1 in free_dofs: 
                    free_dofs.remove(2*node_idx+1)

        # 3. Solve System (Ku = F)
        U_global = np.zeros(n_dof)
        if free_dofs:
            K_ff = K_global[np.ix_(free_dofs, free_dofs)]
            F_f = F_global[free_dofs]
            try:
                u_solved = np.linalg.solve(K_ff, F_f)
                U_global[free_dofs] = u_solved
            except np.linalg.LinAlgError:
                return pd.DataFrame(), np.zeros(n_dof) # Return empty if unstable

        # 4. Post-Processing (Shear, Moment, Deflection)
        results = []
        
        for i, L in enumerate(self.spans):
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            u_el = U_global[idx]
            
            # Boundary conditions for this span
            v_start = u_el[0]      # Deflection at start
            theta_start = u_el[1]  # Rotation at start
            
            # Recalculate forces at start of beam element
            # To get V_start and M_start for integration method
            fem_local = np.zeros(4)
            span_loads = [l for l in self.loads if int(l['span_idx']) == i]
            # (Re-calculate FEM for local force recovery - code dup simplified for clarity)
            for load in span_loads:
                if load['type'] == 'P':
                    a = load['x']; b = L - a; P = load['P']
                    fem_local[0] += P*b**2*(3*a+b)/L**3; fem_local[1] += P*a*b**2/L**2
                    fem_local[2] += P*a**2*(a+3*b)/L**3; fem_local[3] -= P*a**2*b/L**2
                elif load['type'] == 'U':
                    w = load['w']
                    fem_local[0] += w*L/2; fem_local[1] += w*L**2/12
                    fem_local[2] += w*L/2; fem_local[3] -= w*L**2/12
            
            k_el = (self.E * self.I / L**3) * np.array([
                [12, 6*L, -12, 6*L], [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L], [6*L, 2*L**2, -6*L, 4*L**2]
            ])
            
            f_local = k_el @ u_el + fem_local
            V_start = f_local[0]
            M_start = f_local[1] # Convention: Counter-Clockwise +
            
            # Calculate values along the beam
            x_eval = np.linspace(0, L, 50) # 50 points per span
            
            M_vals = []
            V_vals = []
            
            for x in x_eval:
                V_x = V_start
                M_x = -M_start + V_start * x # Moment equilibrium at x
                
                for load in span_loads:
                    if load['type'] == 'P':
                        if x > load['x']:
                            V_x -= load['P']
                            M_x -= load['P'] * (x - load['x'])
                    elif load['type'] == 'U':
                        if x > 0:
                            w = load['w']
                            # Force UDL only up to x
                            d = x 
                            V_x -= w * d
                            M_x -= (w * d) * (d / 2)
                M_vals.append(M_x)
                V_vals.append(V_x)
            
            # Double Integration for Deflection
            # Curvature = M / EI
            M_vals = np.array(M_vals)
            curv = M_vals / (self.E * self.I)
            
            # Integrate Curvature -> Slope (Theta)
            # Use cumulative_trapezoid or cumtrapz depending on scipy version
            if hasattr(integrate, 'cumulative_trapezoid'):
                trapz = integrate.cumulative_trapezoid
            else:
                trapz = integrate.cumtrapz

            theta_vals = theta_start + trapz(curv, x_eval, initial=0)
            
            # Integrate Slope -> Deflection (v)
            deflection_vals = v_start + trapz(theta_vals, x_eval, initial=0)
            
            # Store in list
            for k in range(len(x_eval)):
                results.append({
                    'x': self.nodes[i] + x_eval[k],
                    'shear': V_vals[k],
                    'moment': M_vals[k],
                    'deflection': deflection_vals[k] # <-- บรรทัดนี้สำคัญมากสำหรับการแก้ error
                })

        Reactions = K_global @ U_global 
        return pd.DataFrame(results), Reactions
