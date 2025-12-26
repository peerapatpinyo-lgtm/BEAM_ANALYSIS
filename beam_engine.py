import numpy as np
import pandas as pd

class SimpleBeamSolver:
    """
    Custom Direct Stiffness Method Engine for Continuous Beams.
    Supports: Point Loads, Uniform Loads, Support Settlements (future), etc.
    """
    def __init__(self, spans, supports, loads, E=200e9, I=500e-6):
        self.spans = spans
        self.supports = supports
        self.loads = loads
        self.E = E
        self.I = I
        self.nodes = len(spans) + 1
        self.dof = 2 * self.nodes  # 2 DOF per node (Y, Theta)
        
        # Global Matrices
        self.K = np.zeros((self.dof, self.dof))
        self.F = np.zeros(self.dof)
        self.u = None
        self.R = None
        
    def solve(self):
        # 1. Assemble Stiffness Matrix (K)
        for i, L in enumerate(self.spans):
            # Stiffness Matrix for Beam Element
            k = (self.E * self.I / L**3) * np.array([
                [12, 6*L, -12, 6*L],
                [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L],
                [6*L, 2*L**2, -6*L, 4*L**2]
            ])
            
            # Map local indices to global indices
            idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
            for r in range(4):
                for c in range(4):
                    self.K[idx[r], idx[c]] += k[r, c]

        # 2. Apply Loads (Superposition)
        for load in self.loads:
            span_idx = load['span_idx']
            L = self.spans[span_idx]
            
            n1, n2 = span_idx, span_idx + 1
            idx = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
            fem = np.zeros(4)
            
            if load['type'] == 'Uniform':
                w = load['total_w']
                Ry = (w * L) / 2.0
                M = (w * L**2) / 12.0
                # Fixed End Actions -> Equivalent Nodal Forces
                fem = np.array([-Ry, -M, -Ry, M])
                
            elif load['type'] == 'Point':
                P = load['total_w']
                a = load['pos']
                b = L - a
                M1 = (P * a * b**2) / (L**2)
                M2 = (P * a**2 * b) / (L**2)
                R1 = (P * b**2 * (3*a + b)) / (L**3)
                R2 = (P * a**2 * (a + 3*b)) / (L**3)
                fem = np.array([-R1, -M1, -R2, M2])

            self.F[idx] += fem

        # 3. Boundary Conditions
        fixed_dof = []
        for i, supp in enumerate(self.supports):
            if supp in ['Pin', 'Roller']: 
                fixed_dof.append(2*i)       # Fix Y translation
            elif supp == 'Fix': 
                fixed_dof.extend([2*i, 2*i+1]) # Fix Y and Theta
        
        free_dof = [x for x in range(self.dof) if x not in fixed_dof]
        
        # 4. Partition and Solve
        if len(free_dof) == 0:
            return None, "Structure is fully constrained or invalid."

        K_ff = self.K[np.ix_(free_dof, free_dof)]
        F_f = self.F[free_dof]
        
        try:
            u_f = np.linalg.solve(K_ff, F_f)
        except np.linalg.LinAlgError:
            return None, "Structure Unstable (Singular Matrix)"
            
        self.u = np.zeros(self.dof)
        self.u[free_dof] = u_f
        
        # Calculate Reactions: R = K*u - F
        self.R = self.K @ self.u - self.F
        
        return self.u, None

    def get_internal_forces(self, num_points=100):
        """
        Calculates Shear (V) and Moment (M) along the beam using Statics (Section Method).
        """
        x_total_coords = []
        shear_vals = []
        moment_vals = []
        
        # Map reactions for easy access
        node_ry = {i: self.R[2*i] for i in range(self.nodes)}
        node_rm = {i: self.R[2*i+1] for i in range(self.nodes)}
            
        current_x_start = 0
        
        for i, L in enumerate(self.spans):
            x_local = np.linspace(0, L, num_points)
            x_global = current_x_start + x_local
            
            v_span, m_span = [], []
            
            for x_curr in x_global:
                V, M = 0.0, 0.0
                
                # --- 1. Effect of Reactions to the left ---
                for node_i in range(i + 1):
                    rx = sum(self.spans[:node_i])
                    # Include reaction if it's strictly to the left or at the point
                    if rx <= x_curr + 1e-6:
                        V += node_ry[node_i]
                        # Moment arm = (x_curr - rx)
                        # Reaction Moment is subtracted (standard sign convention)
                        M += -node_rm[node_i] + node_ry[node_i] * (x_curr - rx)
                
                # --- 2. Effect of Loads to the left ---
                for load in self.loads:
                    load_start_x = sum(self.spans[:load['span_idx']])
                    
                    if load['type'] == 'Point':
                        px = load_start_x + load['pos']
                        if px <= x_curr:
                            P = load['total_w']
                            V -= P
                            M -= P * (x_curr - px)
                            
                    elif load['type'] == 'Uniform':
                        lx_start = load_start_x
                        lx_end = load_start_x + self.spans[load['span_idx']]
                        
                        # Determine the segment of load active to the left of x_curr
                        eff_start = lx_start
                        eff_end = min(x_curr, lx_end)
                        
                        if eff_end > eff_start:
                            w = load['total_w']
                            length = eff_end - eff_start
                            load_force = w * length
                            # Centroid of the load segment
                            centroid_dist = x_curr - (eff_start + length/2.0)
                            
                            V -= load_force
                            M -= load_force * centroid_dist
                
                v_span.append(V)
                m_span.append(M)
                
            x_total_coords.extend(x_global)
            shear_vals.extend(v_span)
            moment_vals.extend(m_span)
            
            current_x_start += L
            
        return pd.DataFrame({
            'x': x_total_coords,
            'shear': np.array(shear_vals) / 1000.0,   # Convert N -> kN
            'moment': np.array(moment_vals) / 1000.0  # Convert Nm -> kNm
        })
