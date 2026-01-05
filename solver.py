import numpy as np
import pandas as pd
from scipy import integrate

class BeamSolver:
    def __init__(self, spans, supports, loads, E=2e6, I=1e-4):
        self.spans = spans
        self.supports = supports 
        self.loads = loads if loads is not None else pd.DataFrame()
        self.E = E
        self.I = I
        self.nodes = [0] + list(np.cumsum(spans))
        self.n_nodes = len(self.nodes)
        self.total_len = self.nodes[-1]
        
    def solve(self):
        n_dof = 2 * self.n_nodes 
        K_global = np.zeros((n_dof, n_dof))
        F_global = np.zeros(n_dof)
        
        # --- 1. Stiffness & Load Vector (FEM) ---
        for i, L in enumerate(self.spans):
            # Stiffness Matrix
            k_val = (self.E * self.I / L**3)
            k_el = k_val * np.array([
                [12, 6*L, -12, 6*L], [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L], [6*L, 2*L**2, -6*L, 4*L**2]
            ])
            
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            for r in range(4):
                for c in range(4):
                    K_global[idx[r], idx[c]] += k_el[r, c]
            
            # Fixed End Forces (FEM) due to Loads
            fem = np.zeros(4)
            if not self.loads.empty:
                span_loads = self.loads[self.loads['span_idx'] == i].to_dict('records')
            else:
                span_loads = []
            
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
                    fem[0] += w * L / 2; fem[1] += w * L**2 / 12
                    fem[2] += w * L / 2; fem[3] -= w * L**2 / 12
                elif load['type'] == 'M':
                    M_app = val 
                    a = load['x']; b = L - a
                    # FEM for Moment Load (Counter-clockwise +)
                    fem[0] += -6 * M_app * a * b / L**3
                    fem[1] += M_app * b * (2*a - b) / L**2
                    fem[2] += 6 * M_app * a * b / L**3
                    fem[3] += M_app * a * (2*b - a) / L**2
            
            F_global[idx] -= fem 

        # --- 2. Boundary Conditions ---
        free_dofs = list(range(n_dof))
        if not self.supports.empty:
            for _, row in self.supports.iterrows():
                node_idx = int(row['id'])
                # Constrain Vertical Displacement (y)
                if 2*node_idx in free_dofs: free_dofs.remove(2*node_idx) 
                
                # Constrain Rotation (theta) if Fixed
                if row['type'] == 'Fixed':
                    if 2*node_idx+1 in free_dofs: free_dofs.remove(2*node_idx+1)

        # --- 3. Solve for Displacements ---
        U_global = np.zeros(n_dof)
        if free_dofs:
            try:
                U_global[free_dofs] = np.linalg.solve(K_global[np.ix_(free_dofs, free_dofs)], F_global[free_dofs])
            except np.linalg.LinAlgError:
                raise ValueError("Structure is unstable or singular matrix.")

        # --- 4. Calculate Reactions ---
        # R = K * U - F_equivalent (F_global used negative FEM, so R = K*U + FEM_accumulated)
        # Easier: R = K_global @ U_global - (F_global_original_loads)
        # Actually, standard matrix EQ: F_external + Reactions = K * U
        # So Reactions = K * U - F_external
        Reactions = np.dot(K_global, U_global) - F_global

        # --- 5. Post-Processing for Diagrams (Integration Method) ---
        # We will discretize the beam and integrate Shear -> Moment -> Slope -> Deflection
        
        # A. Create dense x array including key points
        x_points = set([0, self.total_len])
        for n in self.nodes: x_points.add(n)
        if not self.loads.empty:
            for _, l in self.loads.iterrows():
                abs_x = self.nodes[int(l['span_idx'])] + l['x']
                x_points.add(abs_x)
                x_points.add(abs_x - 1e-6) # Discontinuity handling
                x_points.add(abs_x + 1e-6)
        
        # Add dense points for smooth curves
        dense_x = np.linspace(0, self.total_len, 501)
        x_final = np.unique(np.concatenate((list(x_points), dense_x)))
        x_final.sort()
        x_final = x_final[x_final >= 0]
        x_final = x_final[x_final <= self.total_len]

        V_vals = np.zeros_like(x_final)
        M_vals = np.zeros_like(x_final)

        # B. Calculate V and M using Statics (Walking from Left)
        for i, x in enumerate(x_final):
            # Sum Vertical Forces (Reactions + Loads) to the left
            v_sum = 0
            m_sum = 0
            
            # 1. Reactions
            for n_i, node_x in enumerate(self.nodes):
                if node_x <= x + 1e-9: # Reaction is to the left or at x
                    # Vertical Reaction (Index 2*n_i)
                    ry = Reactions[2*n_i]
                    # Moment Reaction (Index 2*n_i+1) - Note: FEM Moment is CCW+, Beam M is Sagging+
                    rm = Reactions[2*n_i+1]
                    
                    v_sum += ry
                    m_sum -= rm # Reaction moment opposes internal moment
                    m_sum += ry * (x - node_x)
            
            # 2. Applied Loads
            if not self.loads.empty:
                for _, l in self.loads.iterrows():
                    l_start_x = self.nodes[int(l['span_idx'])]
                    abs_loc = l_start_x + l['x']
                    
                    if l['type'] == 'P':
                        if abs_loc <= x + 1e-9: # Point load to the left
                            v_sum -= l['mag']
                            m_sum -= l['mag'] * (x - abs_loc)
                            
                    elif l['type'] == 'U':
                        l_end_x = self.nodes[int(l['span_idx']) + 1]
                        # Intersection of load span and current x
                        start_eff = max(l_start_x, 0) # simplified
                        end_eff = min(x, l_end_x)
                        
                        if end_eff > start_eff:
                            dist = end_eff - start_eff
                            load_mag = l['mag'] * dist
                            centroid = start_eff + dist/2
                            v_sum -= load_mag
                            m_sum -= load_mag * (x - centroid)
                            
                    elif l['type'] == 'M':
                        if abs_loc <= x + 1e-9:
                            m_sum += l['mag'] # Applied Moment

            V_vals[i] = v_sum
            M_vals[i] = m_sum

        # C. Calculate Deflection via Integration of M/EI
        # slope = integral(M/EI) + C1
        # defl  = integral(slope) + C2
        
        if hasattr(integrate, 'cumulative_trapezoid'):
            cumtrapz = integrate.cumulative_trapezoid
        else:
            cumtrapz = integrate.cumtrapz

        # Integrate M to get Curvature/Slope
        theta_rel = cumtrapz(M_vals, x_final, initial=0) / (self.E * self.I)
        
        # Integrate Slope to get Deflection shape
        delta_rel = cumtrapz(theta_rel, x_final, initial=0)
        
        # D. Apply Boundary Conditions to Integration Constants
        # We know the true displacement and rotation at Node 0 from the Matrix Solver
        y0_true = U_global[0]      # Vertical displacement at x=0
        theta0_true = U_global[1]  # Rotation at x=0
        
        # Corrected Slope = theta_rel + C1 -> C1 = theta0_true
        # Corrected Defl  = delta_rel + C1*x + C2 -> C2 = y0_true
        
        slope_final = theta_rel + theta0_true
        defl_final = delta_rel + theta0_true * x_final + y0_true

        # --- 6. Pack Results ---
        df_results = pd.DataFrame({
            'x': x_final,
            'shear': V_vals,
            'moment': M_vals,
            'deflection': defl_final
        })
        
        return df_results, Reactions
