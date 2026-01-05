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
        # --- 1. Matrix Stiffness Method (FEM) ---
        n_dof = 2 * self.n_nodes
        K_global = np.zeros((n_dof, n_dof))
        F_global = np.zeros(n_dof)
        
        for i, L in enumerate(self.spans):
            # Stiffness Matrix
            k_val = (self.E * self.I / L**3)
            k_el = k_val * np.array([
                [12, 6*L, -12, 6*L], [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L], [6*L, 2*L**2, -6*L, 4*L**2]
            ])
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            
            # Assembly
            for r in range(4):
                for c in range(4):
                    K_global[idx[r], idx[c]] += k_el[r, c]
            
            # Fixed End Forces (FEM)
            fem = np.zeros(4)
            if not self.loads.empty:
                span_loads = self.loads[self.loads['span_idx'] == i].to_dict('records')
                for load in span_loads:
                    val = load['mag']
                    if load['type'] == 'P':
                        a = load['x']; b = L - a
                        fem[0] += val * b**2 * (3*a + b) / L**3
                        fem[1] += val * a * b**2 / L**2
                        fem[2] += val * a**2 * (a + 3*b) / L**3
                        fem[3] -= val * a**2 * b / L**2
                    elif load['type'] == 'U':
                        w = val
                        fem[0] += w * L / 2; fem[1] += w * L**2 / 12
                        fem[2] += w * L / 2; fem[3] -= w * L**2 / 12
            
            F_global[idx] -= fem

        # --- 2. Boundary Conditions ---
        free_dofs = list(range(n_dof))
        if not self.supports.empty:
            for _, row in self.supports.iterrows():
                node_idx = int(row['id'])
                # Fix Vertical (Y)
                if 2*node_idx in free_dofs: free_dofs.remove(2*node_idx)
                # Fix Rotation if Fixed
                if row['type'] == 'Fixed':
                    if 2*node_idx+1 in free_dofs: free_dofs.remove(2*node_idx+1)

        # --- 3. Solve Displacements ---
        U_global = np.zeros(n_dof)
        if free_dofs:
            try:
                U_global[free_dofs] = np.linalg.solve(K_global[np.ix_(free_dofs, free_dofs)], F_global[free_dofs])
            except np.linalg.LinAlgError:
                return None, None # Singular matrix

        # --- 4. Reactions ---
        Reactions = np.dot(K_global, U_global) - F_global

        # --- 5. Post-Processing (Integration Method) ---
        # Generate dense x array
        x_points = set([0, self.total_len])
        for n in self.nodes: x_points.add(n)
        if not self.loads.empty:
            for _, l in self.loads.iterrows():
                abs_x = self.nodes[int(l['span_idx'])] + l['x']
                x_points.add(abs_x)
                x_points.add(abs_x - 1e-5)
                x_points.add(abs_x + 1e-5)
        
        dense_x = np.linspace(0, self.total_len, 501)
        x_final = np.sort(np.unique(np.concatenate((list(x_points), dense_x))))
        x_final = x_final[(x_final >= 0) & (x_final <= self.total_len)]

        V_vals = np.zeros_like(x_final)
        M_vals = np.zeros_like(x_final)

        for i, x in enumerate(x_final):
            v_sum = 0; m_sum = 0
            
            # Reactions
            for n_i, node_x in enumerate(self.nodes):
                if node_x <= x + 1e-9:
                    ry = Reactions[2*n_i]
                    rm = Reactions[2*n_i+1]
                    v_sum += ry
                    m_sum += ry * (x - node_x) + rm
            
            # Loads
            if not self.loads.empty:
                for _, l in self.loads.iterrows():
                    l_start = self.nodes[int(l['span_idx'])]
                    if l['type'] == 'P':
                        abs_loc = l_start + l['x']
                        if abs_loc <= x + 1e-9:
                            v_sum -= l['mag']
                            m_sum -= l['mag'] * (x - abs_loc)
                    elif l['type'] == 'U':
                        l_end = self.nodes[int(l['span_idx']) + 1]
                        start_eff = l_start
                        end_eff = min(x, l_end)
                        if end_eff > start_eff:
                            dist = end_eff - start_eff
                            load = l['mag'] * dist
                            cent = start_eff + dist/2
                            v_sum -= load
                            m_sum -= load * (x - cent)

            V_vals[i] = v_sum
            M_vals[i] = m_sum

        # Deflection
        if hasattr(integrate, 'cumulative_trapezoid'): cumtrapz = integrate.cumulative_trapezoid
        else: cumtrapz = integrate.cumtrapz

        theta_rel = cumtrapz(M_vals, x_final, initial=0) / (self.E * self.I)
        delta_rel = cumtrapz(theta_rel, x_final, initial=0)
        
        # Apply Boundary Constants (Use U_global[0] and U_global[1])
        y0_true = U_global[0]
        theta0_true = U_global[1]
        
        defl_final = delta_rel + theta0_true * x_final + y0_true

        return pd.DataFrame({'x': x_final, 'shear': V_vals, 'moment': M_vals, 'deflection': defl_final}), Reactions
