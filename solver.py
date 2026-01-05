import numpy as np
import pandas as pd

class BeamSolver:
    def __init__(self, spans, supports, loads, E, I):
        """
        Matrix Stiffness Method Solver (Exact FEM)
        """
        self.spans = np.array(spans, dtype=float)
        self.supports = supports
        self.loads = loads
        self.E = float(E)
        self.I = float(I)
        self.nodes = np.concatenate(([0], np.cumsum(self.spans)))
        self.n_nodes = len(self.nodes)
        self.total_len = self.nodes[-1]
        
    def _get_fixed_end_reactions(self, L, load):
        """
        Calculate Exact Fixed End Reactions (Forces & Moments from Support -> Beam)
        Positive convention: Upward Force (+y), Counter-Clockwise Moment (+Mz)
        """
        fem = np.zeros(4) # [Fy1, M1, Fy2, M2]
        mag = load['mag']
        
        if load['type'] == 'P':
            # Point Load (Standard Formulas)
            a = load['x']
            b = L - a
            # Reaction Forces (Upward is positive)
            fem[0] = (mag * b**2 * (3*a + b)) / L**3
            fem[1] = (mag * a * b**2) / L**2
            fem[2] = (mag * a**2 * (a + 3*b)) / L**3
            fem[3] = -(mag * a**2 * b) / L**2
            
        elif load['type'] == 'U':
            # Uniform Load
            start = load['x']
            dist = load.get('dist', L - start) 
            end = start + dist
            w = mag
            
            # Use Gauss-Legendre Quadrature for Exact Integration of Partial/Full UDL
            # We calculate Reactions (Force needed to hold the load)
            # Reaction = Integral( ShapeFunction * w * dx )
            
            gl_x = np.array([-0.774596669, 0, 0.774596669])
            gl_w = np.array([0.555555556, 0.888888889, 0.555555556])
            
            mid = (start + end) / 2
            jac = (end - start) / 2
            
            for i in range(3):
                xi_global = mid + jac * gl_x[i] # Position on beam 0..L
                weight = gl_w[i] * jac * w 
                
                # Hermite Shape Functions at xi_global
                xi = xi_global / L
                n1 = 1 - 3*xi**2 + 2*xi**3
                n2 = L * (xi - 2*xi**2 + xi**3)
                n3 = 3*xi**2 - 2*xi**3
                n4 = L * (-xi**2 + xi**3)
                
                # Add to reaction vector
                fem += weight * np.array([n1, n2, n3, n4])

        return fem

    def solve(self):
        n_dof = 2 * self.n_nodes
        K = np.zeros((n_dof, n_dof))
        F_node = np.zeros(n_dof) # Global Nodal Force Vector
        
        # 1. Assemble Stiffness Matrix & Load Vector
        for i, L in enumerate(self.spans):
            k = self.E * self.I / L**3
            k_el = k * np.array([
                [12, 6*L, -12, 6*L],
                [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L],
                [6*L, 2*L**2, -6*L, 4*L**2]
            ])
            
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            for r in range(4):
                for c in range(4):
                    K[idx[r], idx[c]] += k_el[r, c]
                    
            # 2. Process Loads
            if not self.loads.empty:
                span_loads = self.loads[self.loads['span_idx'] == i]
                for _, l in span_loads.iterrows():
                    # Calculate Fixed End Reactions (Forces from Support -> Beam)
                    reactions = self._get_fixed_end_reactions(L, l)
                    
                    # 🔴 CRITICAL FIX HERE:
                    # Nodal Force F = F_external - F_reactions
                    # We subtract the reactions to get the equivalent nodal loads
                    F_node[idx] -= reactions

        # 3. Apply Boundary Conditions
        active_dof = list(range(n_dof))
        
        for _, s in self.supports.iterrows():
            node_idx = int(s['id'])
            # Vertical Restraint (Ty)
            if 2*node_idx in active_dof:
                active_dof.remove(2*node_idx)
            
            # Rotational Restraint (Rz) - Only for Fixed support
            if s['type'] == 'Fixed':
                if 2*node_idx+1 in active_dof:
                    active_dof.remove(2*node_idx+1)

        # 4. Solve for Displacements
        U = np.zeros(n_dof)
        if len(active_dof) > 0:
            K_reduced = K[np.ix_(active_dof, active_dof)]
            F_reduced = F_node[active_dof]
            try:
                U[active_dof] = np.linalg.solve(K_reduced, F_reduced)
            except np.linalg.LinAlgError:
                raise Exception("Structure is unstable or singular matrix.")

        # 5. Calculate Reactions
        # R = K*U - F_node
        # Note: F_node already contains the "-Reactions" from loads
        R = K @ U - F_node
        
        return self._post_process(U, R)

    def _post_process(self, U, R):
        """
        Generate Diagram Data
        """
        x_plot = []
        v_plot = []
        m_plot = []
        d_plot = []
        
        num_points = 200 
        
        for i, L in enumerate(self.spans):
            x_local = np.linspace(0, L, num_points)
            x_global = self.nodes[i] + x_local
            
            # Get Element Displacements
            u_el = U[2*i : 2*i+4] 
            
            # --- Deflection (Hermite) ---
            xi = x_local / L
            N1 = 1 - 3*xi**2 + 2*xi**3
            N2 = L * (xi - 2*xi**2 + xi**3)
            N3 = 3*xi**2 - 2*xi**3
            N4 = L * (-xi**2 + xi**3)
            
            y_elastic = N1*u_el[0] + N2*u_el[1] + N3*u_el[2] + N4*u_el[3]
            
            # Particular solution for UDL (Simply Supported Curvature correction)
            y_particular = np.zeros_like(x_local)
            if not self.loads.empty:
                span_loads = self.loads[self.loads['span_idx'] == i]
                for _, l in span_loads.iterrows():
                    if l['type'] == 'U':
                         w = l['mag']
                         # Approximate shape for visualization (Standard UDL eq)
                         # This adds the "belly" to the deflection curve
                         y_particular += -w * x_local * (L**3 - 2*L*x_local**2 + x_local**3) / (24 * self.E * self.I)

            d_total = y_elastic + y_particular
            
            # --- Shear & Moment (Statics) ---
            v_seg = []
            m_seg = []
            
            for xg in x_global:
                V_val = 0
                M_val = 0
                
                # Sum Reactions
                for n_idx, nx in enumerate(self.nodes):
                    if nx <= xg + 1e-5: 
                        V_val += R[2*n_idx]
                        M_val += R[2*n_idx] * (xg - nx) + R[2*n_idx+1]
                
                # Subtract Loads
                if not self.loads.empty:
                    for _, l in self.loads.iterrows():
                        lx = self.nodes[int(l['span_idx'])] + l['x']
                        
                        if l['type'] == 'P':
                            if lx <= xg - 1e-5:
                                V_val -= l['mag']
                                M_val -= l['mag'] * (xg - lx)
                                
                        elif l['type'] == 'U':
                            l_start = lx
                            if xg > l_start + 1e-5:
                                dist = l.get('dist', self.spans[int(l['span_idx'])] - l['x'])
                                l_end = l_start + dist
                                eff_end = min(xg, l_end)
                                eff_len = eff_end - l_start
                                
                                if eff_len > 0:
                                    load_mag = l['mag'] * eff_len
                                    centroid = l_start + eff_len / 2
                                    V_val -= load_mag
                                    M_val -= load_mag * (xg - centroid)
                
                v_seg.append(V_val)
                m_seg.append(M_val)
            
            x_plot.extend(x_global)
            v_plot.extend(v_seg)
            m_plot.extend(m_seg)
            d_plot.extend(d_total)

        return pd.DataFrame({
            'x': x_plot, 
            'shear': v_plot, 
            'moment': m_plot, 
            'deflection': d_plot
        }), R
