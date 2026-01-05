import numpy as np
import pandas as pd
from scipy import integrate

class BeamSolver:
    def __init__(self, spans, supports, loads, E=2e6, I=1e-4):
        """
        Custom Matrix Stiffness Solver for Continuous Beams
        """
        self.spans = spans
        self.supports = supports 
        self.loads = loads if loads is not None else pd.DataFrame()
        self.E = E
        self.I = I
        
        # Geometry
        self.nodes = [0] + list(np.cumsum(spans))
        self.n_nodes = len(self.nodes)
        self.total_len = self.nodes[-1]
        
    def solve(self):
        # --- A. Matrix Stiffness Method (FEM) ---
        n_dof = 2 * self.n_nodes # 2 DOF per node (y, theta)
        K_global = np.zeros((n_dof, n_dof))
        F_global = np.zeros(n_dof)
        
        # 1. Build Stiffness Matrix & Load Vector
        for i, L in enumerate(self.spans):
            # Element Stiffness (Bernoulli Beam)
            k = (self.E * self.I / L**3)
            k_el = k * np.array([
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
            
            # Fixed End Forces (FEM) calculation
            fem = np.zeros(4)
            if not self.loads.empty:
                # Filter loads in this span
                # We need to check coordinate carefully. 
                # Input 'x' is relative to span start.
                span_loads = self.loads[self.loads['span_idx'] == i].to_dict('records')
                
                for load in span_loads:
                    val = load['mag']
                    a = load['x']
                    b = L - a
                    
                    if load['type'] == 'P': # Point Load
                        # Ry_L, M_L, Ry_R, M_R
                        fem[0] += val * b**2 * (3*a + b) / L**3
                        fem[1] += val * a * b**2 / L**2
                        fem[2] += val * a**2 * (a + 3*b) / L**3
                        fem[3] -= val * a**2 * b / L**2
                        
                    elif load['type'] == 'U': # Uniform Load
                        w = val
                        # Treat full span uniform for simplicity or partial
                        # Assuming full span based on UI, but if partial logic is needed:
                        # (Simplified for full span w)
                        fem[0] += w * L / 2
                        fem[1] += w * L**2 / 12
                        fem[2] += w * L / 2
                        fem[3] -= w * L**2 / 12
            
            # Add FEM to Global Force Vector (Subtraction because F_node = -FEM)
            F_global[idx] -= fem

        # 2. Apply Boundary Conditions
        free_dofs = list(range(n_dof))
        
        # Support logic
        if not self.supports.empty:
            for _, row in self.supports.iterrows():
                node_idx = int(row['id'])
                stype = row['type']
                
                # All supports constrain Vertical Y (index 2*node)
                dof_y = 2*node_idx
                if dof_y in free_dofs: free_dofs.remove(dof_y)
                
                # Fixed support constrains Rotation (index 2*node + 1)
                if stype == 'Fixed':
                    dof_th = 2*node_idx + 1
                    if dof_th in free_dofs: free_dofs.remove(dof_th)

        # 3. Solve System (KU = F)
        U_global = np.zeros(n_dof)
        if free_dofs:
            K_free = K_global[np.ix_(free_dofs, free_dofs)]
            F_free = F_global[free_dofs]
            try:
                U_free = np.linalg.solve(K_free, F_free)
                U_global[free_dofs] = U_free
            except np.linalg.LinAlgError:
                raise ValueError("Structure is unstable (Singular Matrix)")

        # 4. Calculate Reactions (R = K*U - F_applied)
        # Note: F_global currently holds -FEM. 
        # Correct statics: Reactions + F_external = K * U
        # Reactions = K*U - F_external
        Reactions = np.dot(K_global, U_global) - F_global

        # --- B. Post-Processing (Generate Diagrams) ---
        # Strategy: Use Method of Sections (Integration) from Left to Right
        # utilizing the calculated Reactions and Applied Loads.
        
        # Create dense X coordinates
        x_points = set([0, self.total_len])
        for n in self.nodes: x_points.add(n)
        if not self.loads.empty:
            for _, l in self.loads.iterrows():
                abs_x = self.nodes[int(l['span_idx'])] + l['x']
                x_points.add(abs_x)
                x_points.add(abs_x - 1e-5) # For discontinuities
                x_points.add(abs_x + 1e-5)
        
        dense_x = np.linspace(0, self.total_len, 501)
        x_final = np.unique(np.concatenate((list(x_points), dense_x)))
        x_final = np.sort(x_final)
        x_final = x_final[(x_final >= 0) & (x_final <= self.total_len)]

        V_vals = np.zeros_like(x_final)
        M_vals = np.zeros_like(x_final)

        for i, x in enumerate(x_final):
            v_sum = 0
            m_sum = 0
            
            # 1. Add Reactions (to the left of x)
            for n_i, node_x in enumerate(self.nodes):
                if node_x <= x + 1e-9:
                    ry = Reactions[2*n_i]
                    rm = Reactions[2*n_i+1] # Reaction Moment
                    
                    v_sum += ry
                    # Reaction moment direction check:
                    # Matrix: CCW+. Beam Statics: Sagging+. 
                    # If Support creates CCW reaction, it pushes beam up? No.
                    # Standard: M_internal = Sum(M_forces) + Sum(M_reactions)
                    m_sum += ry * (x - node_x)
                    m_sum += rm # Add concentrated moment reaction
            
            # 2. Add Loads (to the left of x)
            if not self.loads.empty:
                for _, l in self.loads.iterrows():
                    l_start_x = self.nodes[int(l['span_idx'])]
                    abs_loc = l_start_x + l['x']
                    
                    if l['type'] == 'P':
                        if abs_loc <= x + 1e-9:
                            load_val = l['mag'] # Downward load is positive in input? 
                            # Usually input P is magnitude. Gravity is down.
                            # Let's assume input is positive for gravity load
                            v_sum -= load_val 
                            m_sum -= load_val * (x - abs_loc)
                            
                    elif l['type'] == 'U':
                        l_end_x = self.nodes[int(l['span_idx']) + 1]
                        # Overlap between [l_start, l_end] and [0, x]
                        eff_start = l_start_x
                        eff_end = min(x, l_end_x)
                        
                        if eff_end > eff_start:
                            dist = eff_end - eff_start
                            w_mag = l['mag'] * dist
                            centroid = eff_start + dist/2
                            v_sum -= w_mag
                            m_sum -= w_mag * (x - centroid)

            V_vals[i] = v_sum
            M_vals[i] = m_sum

        # --- C. Deflection (Double Integration) ---
        # Integrate M/EI -> Slope -> Deflection
        # Using cumulative trapezoid integration
        
        # 1. Slope (theta) = Integral(M/EI) + C1
        # We know theta at node 0 from Matrix Solver (U_global[1])
        if hasattr(integrate, 'cumulative_trapezoid'):
            cumtrapz = integrate.cumulative_trapezoid
        else:
            cumtrapz = integrate.cumtrapz

        curvature = M_vals / (self.E * self.I)
        theta_rel = cumtrapz(curvature, x_final, initial=0)
        
        # 2. Deflection (delta) = Integral(theta) + C2
        # We know delta at node 0 from Matrix Solver (U_global[0])
        delta_rel = cumtrapz(theta_rel, x_final, initial=0)
        
        # 3. Apply Boundary Constants from Matrix Solution
        # True Slope = theta_rel + theta_start
        # True Defl  = delta_rel + theta_start*x + delta_start
        
        theta_start = U_global[1] # Rotation at Node 0
        delta_start = U_global[0] # Displacement at Node 0
        
        defl_final = delta_rel + theta_start * x_final + delta_start

        # Pack Data
        df_results = pd.DataFrame({
            'x': x_final,
            'shear': V_vals,
            'moment': M_vals,
            'deflection': defl_final
        })
        
        return df_results, Reactions
