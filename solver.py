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
            # Element Stiffness Matrix (Standard Beam Element)
            # |  12   6L  -12   6L |
            # |  6L  4L^2 -6L  2L^2| ...
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
            # Convention for Vector: Force UP +, Moment CCW +
            fem = np.zeros(4)
            span_loads = [l for l in self.loads if int(l['span_idx']) == i]
            
            for load in span_loads:
                val = load['mag']
                
                if load['type'] == 'P':
                    # Point Load P (Down is usually input as positive magnitude in UI, 
                    # but FEM formulas assume P acts down. Reaction opposes load.)
                    a = load['x']; b = L - a; P = val
                    fem[0] += P * b**2 * (3*a + b) / L**3  # Fy1
                    fem[1] += P * a * b**2 / L**2          # M1 (CCW)
                    fem[2] += P * a**2 * (a + 3*b) / L**3  # Fy2
                    fem[3] -= P * a**2 * b / L**2          # M2 (CW -> Negative)
                    
                elif load['type'] == 'U':
                    # UDL w
                    w = val
                    fem[0] += w * L / 2
                    fem[1] += w * L**2 / 12
                    fem[2] += w * L / 2
                    fem[3] -= w * L**2 / 12
                    
                elif load['type'] == 'M':
                    # Moment Load M (Input: CW is Positive in UI convention usually)
                    # Let's assume input 'val' > 0 means Clockwise Moment on Beam.
                    # FEM Reactions to a Clockwise Moment M at 'a':
                    # Based on standard tables:
                    # M_A (Left, CCW) = M * b * (2a - b) / L^2
                    # M_B (Right, CCW) = M * a * (2b - a) / L^2
                    # Note: Both reactions are typically CCW to resist the CW twist in middle.
                    
                    M_app = val 
                    a = load['x']; b = L - a
                    
                    # Vertical forces to balance moments:
                    # R_A + R_B = 0
                    # Sum M_B = 0 -> R_A*L - M_app + M_A + M_B = 0 ... simplified:
                    # V_A = -6*M*a*b/L^3
                    # V_B =  6*M*a*b/L^3
                    
                    fem[0] += -6 * M_app * a * b / L**3
                    fem[1] += M_app * b * (2*a - b) / L**2  # CCW Reaction
                    fem[2] += 6 * M_app * a * b / L**3
                    fem[3] += M_app * a * (2b - a) / L**2   # CCW Reaction
            
            # Subtract FEM from Global Force Vector (F = F_ext - FEM)
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

        # 3. Solve System (Ku = F)
        U_global = np.zeros(n_dof)
        if free_dofs:
            K_ff = K_global[np.ix_(free_dofs, free_dofs)]
            F_f = F_global[free_dofs]
            try:
                u_solved = np.linalg.solve(K_ff, F_f)
                U_global[free_dofs] = u_solved
            except np.linalg.LinAlgError:
                return pd.DataFrame(), np.zeros(n_dof)

        # 4. Post-Processing & Result Generation
        results = []
        
        for i, L in enumerate(self.spans):
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            u_el = U_global[idx]
            
            # Re-calculate Local Forces (Standard Slope-Deflection)
            # Need local FEM again for superposition
            fem_local = np.zeros(4)
            span_loads = [l for l in self.loads if int(l['span_idx']) == i]
            
            # Determine Evaluation Points (Critical X)
            eval_points = set(np.linspace(0, L, 100)) # Standard resolution
            
            for load in span_loads:
                val = load['mag']
                if load['type'] == 'P':
                    a = load['x']; b = L - a; P = val
                    fem_local[0] += P*b**2*(3*a+b)/L**3; fem_local[1] += P*a*b**2/L**2
                    fem_local[2] += P*a**2*(a+3*b)/L**3; fem_local[3] -= P*a**2*b/L**2
                    # Critical Points
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
                    # Critical Points for Jump
                    eval_points.add(a); 
                    if a>0: eval_points.add(a-1e-6)
                    if a<L: eval_points.add(a+1e-6)

            x_eval = sorted(list(eval_points))
            
            # Stiffness Force Recovery
            k_el = (self.E * self.I / L**3) * np.array([
                [12, 6*L, -12, 6*L], [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L], [6*L, 2*L**2, -6*L, 4*L**2]
            ])
            f_nodal = k_el @ u_el + fem_local
            
            # Starting Values
            V_start = f_nodal[0]
            M_start = f_nodal[1] # Nodal Moment (CCW+)
            
            M_vals = []
            V_vals = []
            
            for x in x_eval:
                # Equilibrium at cut x
                # V(x) = V_start - Sum(Loads)
                # M(x) = -M_start + V_start*x - Sum(Load Moments)
                # Note: M_start is CCW, so it acts Negative in Sagging convention?
                # Beam Convention: Sagging is positive (+).
                # A positive Nodal Moment M1 (CCW) tends to lift the beam -> Sagging -> Positive?
                # No, CCW at left end pushes down -> Hogging -> Negative.
                # So M_internal = -M_node_CCW + ...
                
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
                        # Applied Moment M (CW).
                        # If we cut section after M, the applied moment M is in equilibrium equation.
                        # Internal M + Applied M_CW = External Moments...
                        # A CW applied moment causes an UPWARD jump in BMD (Sagging increases).
                        if x >= load['x'] + 1e-9:
                            M_x += val 
                            
                V_vals.append(V_x)
                M_vals.append(M_x)
            
            # Double Integration for Deflection
            # y'' = M/EI
            M_arr = np.array(M_vals)
            curv = M_arr / (self.E * self.I)
            
            if hasattr(integrate, 'cumulative_trapezoid'):
                trapz = integrate.cumulative_trapezoid
            else:
                trapz = integrate.cumtrapz
            
            # Integrate Curvature -> Slope (theta)
            # theta(x) = theta_start + int(M/EI)
            theta_start = u_el[1]
            theta_vals = theta_start + trapz(curv, x_eval, initial=0)
            
            # Integrate Slope -> Deflection (v)
            v_start_node = u_el[0]
            deflection_vals = v_start_node + trapz(theta_vals, x_eval, initial=0)
            
            for k in range(len(x_eval)):
                results.append({
                    'x': self.nodes[i] + x_eval[k],
                    'shear': V_vals[k],
                    'moment': M_vals[k],
                    'deflection': deflection_vals[k]
                })

        Reactions = K_global @ U_global 
        
        # 5. Result Cleanup (Snap to Zero)
        df_res = pd.DataFrame(results)
        
        # Snap values very close to zero -> 0.0 (Cosmetic & Accuracy)
        tol = 1e-5
        df_res.loc[df_res['moment'].abs() < tol, 'moment'] = 0.0
        df_res.loc[df_res['shear'].abs() < tol, 'shear'] = 0.0
        df_res.loc[df_res['deflection'].abs() < 1e-9, 'deflection'] = 0.0
        
        return df_res, Reactions
