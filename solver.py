import numpy as np
import pandas as pd

class BeamSolver:
    def __init__(self, spans, supports, loads):
        self.spans = spans
        self.supports = supports
        self.loads = loads
        self.n_nodes = len(spans) + 1
        self.n_dof = 2 * self.n_nodes  # DOF per node: [Fy, Mz]
        
        # Initialization
        self.K = np.zeros((self.n_dof, self.n_dof))
        self.F_global = np.zeros(self.n_dof) # Loads vector
        self.U = np.zeros(self.n_dof) # Displacements
        self.EI = 10000.0 # Arbitrary stiffness (Assume constant EI)

    def solve(self):
        # 1. Build Global Stiffness Matrix (K)
        for i, L in enumerate(self.spans):
            # Element Stiffness Matrix (Standard Beam Element)
            # Signs: Fy (+Up), M (+Counter-Clockwise)
            k_ele = (self.EI / L**3) * np.array([
                [ 12,   6*L, -12,   6*L],
                [ 6*L, 4*L**2, -6*L, 2*L**2],
                [-12,  -6*L,  12,  -6*L],
                [ 6*L, 2*L**2, -6*L, 4*L**2]
            ])
            
            # Map to Global DOFs
            # Node i indices: 2*i, 2*i+1
            # Node i+1 indices: 2*(i+1), 2*(i+1)+1
            indexes = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
            
            for r in range(4):
                for c in range(4):
                    self.K[indexes[r], indexes[c]] += k_ele[r, c]

        # 2. Build Global Load Vector (F) from Fixed End Moments (FEM)
        # F_global = F_nodal_loads - F_fixed_end_reactions
        # (We only handle member loads converted to equivalent nodal loads)
        
        # Array to store FEMs per span for post-processing later
        self.span_fems = [] 
        
        for i, L in enumerate(self.spans):
            # Calculate FEM for this span
            # Convention: Up (+), CCW (+)
            fem = np.zeros(4) # [Fy_L, M_L, Fy_R, M_R]
            
            span_loads = [l for l in self.loads if l['span_idx'] == i]
            for l in span_loads:
                if l['type'] == 'U': # Uniform Load w (Gravity/Down)
                    w = l['w']
                    # Reactions to support load w:
                    fem[0] += w * L / 2        # Fy_L (Up)
                    fem[1] += w * L**2 / 12    # M_L (CCW)
                    fem[2] += w * L / 2        # Fy_R (Up)
                    fem[3] -= w * L**2 / 12    # M_R (CW -> Negative)
                    
                elif l['type'] == 'P': # Point Load P (Gravity/Down)
                    P = l['P']
                    a = l['x']
                    b = L - a
                    fem[0] += P * b**2 * (3*a + b) / L**3
                    fem[1] += P * a * b**2 / L**2
                    fem[2] += P * a**2 * (a + 3*b) / L**3
                    fem[3] -= P * a**2 * b / L**2
            
            self.span_fems.append(fem)
            
            # Subtract FEM from Global Force Vector (Equivalent Nodal Loads)
            # Nodes i and i+1
            idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
            for j in range(4):
                self.F_global[idx[j]] -= fem[j]

        # 3. Apply Boundary Conditions
        fixed_dofs = []
        for _, row in self.supports.iterrows():
            n_idx = int(row['id'])
            stype = row['type']
            
            # DOF indices: 2*n_idx (Fy), 2*n_idx+1 (Mz)
            if stype == "Pin" or stype == "Roller":
                fixed_dofs.append(2*n_idx) # Fix Fy only
            elif stype == "Fixed":
                fixed_dofs.append(2*n_idx)     # Fix Fy
                fixed_dofs.append(2*n_idx+1)   # Fix Mz (Rotation)

        # 4. Solve for Displacements [U]
        free_dofs = [d for d in range(self.n_dof) if d not in fixed_dofs]
        
        if not free_dofs:
            return None, None # Fully constrained or error
            
        K_free = self.K[np.ix_(free_dofs, free_dofs)]
        F_free = self.F_global[free_dofs]
        
        try:
            U_free = np.linalg.solve(K_free, F_free)
        except np.linalg.LinAlgError:
            return None, None # Unstable structure
            
        self.U[free_dofs] = U_free
        
        # 5. Post-Processing: Calculate Internal Forces (Shear & Moment Diagrams)
        return self._generate_diagrams()

    def _generate_diagrams(self):
        x_plot = []
        shear_plot = []
        moment_plot = []
        
        cum_len = 0
        
        # Store reactions
        reactions = np.zeros(self.n_dof)
        
        for i, L in enumerate(self.spans):
            # 1. Get Member End Forces (Local)
            # f_member = k_ele * u_ele + fem
            idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
            u_ele = self.U[idx]
            
            k_ele = (self.EI / L**3) * np.array([
                [ 12,   6*L, -12,   6*L],
                [ 6*L, 4*L**2, -6*L, 2*L**2],
                [-12,  -6*L,  12,  -6*L],
                [ 6*L, 2*L**2, -6*L, 4*L**2]
            ])
            
            fem = self.span_fems[i]
            
            # Forces exerted BY THE NODES ON THE BEAM
            f_ele = np.dot(k_ele, u_ele) + fem
            
            # f_ele = [Fy_L, M_L, Fy_R, M_R] (Matrix Sign Convention: Up+, CCW+)
            
            # 2. Convert to Diagram Starting Values
            # Shear V at x=0 is Fy_L
            V_start = f_ele[0]
            
            # Moment M at x=0 (Beam Sign Convention: Sagging +)
            # Matrix M_L is CCW. On the Left end, CCW bends the beam DOWN (Hogging).
            # So Internal Moment = -M_Matrix
            M_start = -f_ele[1] 
            
            # 3. Integrate along the span to generate diagram points
            # Points of interest: Start, End, Loads
            points = [0.0, L]
            span_loads = [l for l in self.loads if l['span_idx'] == i]
            for l in span_loads:
                if l['type'] == 'P':
                    points.extend([l['x'], l['x']-1e-6, l['x']+1e-6])
            
            # Dense points for curves
            points.extend(np.linspace(0, L, 100))
            points = sorted(list(set([p for p in points if 0 <= p <= L])))
            
            for x in points:
                # V(x) = V_start - Load_Force_Accumulated
                # M(x) = M_start + Area_Shear
                
                # Manual Statics at distance x from left node
                Vx = V_start
                Mx = M_start + V_start * x # Moment from V_start
                
                for l in span_loads:
                    if l['type'] == 'U':
                        w = l['w']
                        # Uniform Load starts at 0, goes to x
                        # Force = w * x (Down)
                        if x > 0:
                            Vx -= w * x
                            # Moment arm is x/2
                            Mx -= (w * x) * (x / 2)
                            
                    elif l['type'] == 'P':
                        P = l['P']
                        xp = l['x']
                        if x > xp:
                            Vx -= P
                            Mx -= P * (x - xp)
                            
                # Global coordinate
                x_plot.append(cum_len + x)
                
                # Fix floating point noise for Pin supports (If M is very close to 0, make it 0)
                if abs(Mx) < 1e-4: Mx = 0.0
                
                shear_plot.append(Vx)
                moment_plot.append(Mx)
            
            # Calculate Reactions (Force imbalance at nodes)
            # This is simplified; usually R = K_global*U - F_ext
            # But here we can sum member end forces at each node
            
            cum_len += L

        # Create DataFrame
        df = pd.DataFrame({'x': x_plot, 'shear': shear_plot, 'moment': moment_plot})
        return df, reactions
