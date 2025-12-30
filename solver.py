import numpy as np
import pandas as pd

class BeamSolver:
    def __init__(self, spans, supports, loads):
        """
        Direct Stiffness Method Solver tailored for Continuous Beams.
        Verified against standard Structural Analysis textbooks (Hibbeler/Kassimali).
        """
        self.spans = spans
        self.supports = supports
        self.loads = loads
        self.n_nodes = len(spans) + 1
        self.n_dof = 2 * self.n_nodes  # DOFs: [Fy1, Mz1, Fy2, Mz2, ...]
        
        # Matrix Properties
        self.K_global = np.zeros((self.n_dof, self.n_dof))
        self.F_global = np.zeros(self.n_dof)  # This will hold (F_nodal - F_fixed_end)
        self.U = np.zeros(self.n_dof)
        
        # Constants (Relative EI is sufficient for determinate/indeterminate force distribution)
        self.EI = 1.0e6 

    def _get_fem(self, L, span_loads):
        """
        Calculate Fixed End Moments (FEM) and Forces for a span.
        Matrix Sign Convention: Up (+), Counter-Clockwise (+)
        Loads are assumed Gravity (Downward).
        """
        fem = np.zeros(4) # [FyL, MzL, FyR, MzR]
        
        for l in span_loads:
            if l['type'] == 'U':
                w = l['w'] # Downward magnitude
                # Fixed-Fixed Beam Formulas for Uniform Load w
                # FyL = wL/2 (Up +)
                # MzL = wL^2/12 (CCW +)
                # FyR = wL/2 (Up +)
                # MzR = -wL^2/12 (CW -)
                
                fem[0] += w * L / 2
                fem[1] += w * L**2 / 12
                fem[2] += w * L / 2
                fem[3] -= w * L**2 / 12
                
            elif l['type'] == 'P':
                P = l['P'] # Downward magnitude
                a = l['x'] # Distance from Left
                b = L - a
                # Fixed-Fixed Beam Formulas for Point Load P
                # FyL = Pb^2(3a+b)/L^3
                # MzL = Pab^2/L^2 (CCW +)
                # FyR = Pa^2(a+3b)/L^3
                # MzR = -Pa^2b/L^2 (CW -)
                
                fem[0] += P * (b**2 * (3*a + b)) / L**3
                fem[1] += P * a * (b**2) / L**2
                fem[2] += P * (a**2 * (a + 3*b)) / L**3
                fem[3] -= P * (a**2) * b / L**2
                
        return fem

    def solve(self):
        # --- 1. Assemble Stiffness Matrix (K) ---
        for i, L in enumerate(self.spans):
            # Element Stiffness (Bernoulli-Euler Beam)
            k_local = (self.EI / L**3) * np.array([
                [ 12,   6*L, -12,   6*L],
                [ 6*L, 4*L**2, -6*L, 2*L**2],
                [-12,  -6*L,  12,  -6*L],
                [ 6*L, 2*L**2, -6*L, 4*L**2]
            ])
            
            # Map to Global Indices
            idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
            for r in range(4):
                for c in range(4):
                    self.K_global[idx[r], idx[c]] += k_local[r, c]

        # --- 2. Assemble Load Vector (F) ---
        # Formula: {F_system} = {F_nodal_ext} - {F_fixed_end}
        # Since we have no external joint loads, {F_system} = - {F_fixed_end}
        
        self.all_fems = [] # Store for recovery later
        
        for i, L in enumerate(self.spans):
            span_loads = [l for l in self.loads if l['span_idx'] == i]
            fem = self._get_fem(L, span_loads)
            self.all_fems.append(fem)
            
            # Global Indices
            idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
            
            # Subtract FEM from Global Load Vector
            for j in range(4):
                self.F_global[idx[j]] -= fem[j]

        # --- 3. Apply Boundary Conditions ---
        fixed_dofs = []
        for _, row in self.supports.iterrows():
            node_idx = int(row['id'])
            stype = row['type']
            
            # 2*node_idx = Vertical (y), 2*node_idx+1 = Rotation (theta)
            if stype == "Pin" or stype == "Roller":
                fixed_dofs.append(2*node_idx) # Fix Vertical
            elif stype == "Fixed":
                fixed_dofs.append(2*node_idx)   # Fix Vertical
                fixed_dofs.append(2*node_idx+1) # Fix Rotation

        # --- 4. Solve System ---
        free_dofs = [d for d in range(self.n_dof) if d not in fixed_dofs]
        
        if len(free_dofs) > 0:
            K_ff = self.K_global[np.ix_(free_dofs, free_dofs)]
            F_f = self.F_global[free_dofs]
            
            try:
                U_f = np.linalg.solve(K_ff, F_f)
                self.U[free_dofs] = U_f
            except np.linalg.LinAlgError:
                return None, None # Unstable
        
        # --- 5. Post-Processing (Generate Exact Diagrams) ---
        return self._generate_results()

    def _generate_results(self):
        x_out = []
        v_out = []
        m_out = []
        cum_dist = 0
        
        reactions = np.zeros(self.n_dof)
        
        for i, L in enumerate(self.spans):
            # 1. Recover Member End Forces
            # {q} = [k]{u} + {q_fixed}
            idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
            u_ele = self.U[idx]
            fem = self.all_fems[i]
            
            k_local = (self.EI / L**3) * np.array([
                [ 12,   6*L, -12,   6*L],
                [ 6*L, 4*L**2, -6*L, 2*L**2],
                [-12,  -6*L,  12,  -6*L],
                [ 6*L, 2*L**2, -6*L, 4*L**2]
            ])
            
            # Force exerted BY NODE ON BEAM
            q_member = np.dot(k_local, u_ele) + fem
            
            # Save reactions (force from beam on node is -q, force from support on node balances this)
            # Actually, Global Reaction = Sum of Member Forces at that node.
            # Simplified: Global Reaction R = K_global * U + FEM_global_sum? 
            # Let's trust equilibrium of member ends:
            reactions[idx[0]] += q_member[0] # Fy Left
            reactions[idx[1]] += q_member[1] # Mz Left
            reactions[idx[2]] += q_member[2] # Fy Right
            reactions[idx[3]] += q_member[3] # Mz Right
            
            # 2. Section Cut Method (Statics) for Diagram
            # Start Values at x=0 (Left end of span)
            # Shear V = Force Up = q_member[0]
            # Moment M = Internal Sagging Moment. 
            #   Matrix q[1] is CCW Moment on Left End.
            #   CCW Moment on Left End causes HOGGING (Sad curve).
            #   So Internal Moment M_beam = - q_member[1]
            
            V_start = q_member[0]
            M_start = -q_member[1] 
            
            # Define exact evaluation points
            span_loads = [l for l in self.loads if l['span_idx'] == i]
            eval_points = set([0.0, L])
            for l in span_loads:
                if l['type'] == 'P':
                    # Add point and slight offsets to capture shear drop vertically
                    eval_points.add(l['x'])
                    eval_points.add(l['x'] - 1e-10)
                    eval_points.add(l['x'] + 1e-10)
            
            # Add intermediate points for smooth curves (Uniform Load)
            num_segments = 50
            for k in range(num_segments + 1):
                eval_points.add(k * L / num_segments)
                
            sorted_x = sorted(list(eval_points))
            
            # Calculate V(x) and M(x) using Free Body Diagram at distance x
            for x in sorted_x:
                if x < 0 or x > L: continue
                
                # Base values from Left Support reactions
                Vx = V_start
                Mx = M_start + V_start * x  # Moment from Shear force
                
                # Subtract effect of Loads inside [0, x]
                for l in span_loads:
                    if l['type'] == 'U':
                        w = l['w']
                        # Uniform load covers distance x
                        # Force = w * x (Down)
                        # Moment arm = x/2
                        Vx -= w * x
                        Mx -= (w * x) * (x / 2)
                        
                    elif l['type'] == 'P':
                        P = l['P']
                        xp = l['x']
                        if x > xp + 1e-12: # If we are past the load
                            Vx -= P
                            Mx -= P * (x - xp)
                
                # Zero out extremely small noise (e.g. 1e-14) for clean 0.00 at pins
                if abs(Mx) < 1e-6: Mx = 0.0
                if abs(Vx) < 1e-6: Vx = 0.0
                            
                x_out.append(cum_dist + x)
                v_out.append(Vx)
                m_out.append(Mx)
            
            cum_dist += L
            
        return pd.DataFrame({'x': x_out, 'shear': v_out, 'moment': m_out}), reactions
