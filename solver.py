import numpy as np
import pandas as pd

class BeamSolver:
    def __init__(self, spans, supports, loads):
        self.spans = spans
        self.supports = supports
        self.loads = loads
        self.n_nodes = len(spans) + 1
        self.n_dof = 2 * self.n_nodes  # 2 DOF per node (Deflection Y, Rotation Theta)
        
        # Global Stiffness Matrix & Force Vector
        self.K = np.zeros((self.n_dof, self.n_dof))
        self.F = np.zeros(self.n_dof)
        self.U = np.zeros(self.n_dof)
        
    def solve(self):
        # 1. Build Global Stiffness Matrix (Assemble Elements)
        # EI assumed constant = 1.0 (Relative stiffness matters for force distribution)
        EI = 1.0 
        
        for i, L in enumerate(self.spans):
            # Element Stiffness Matrix (4x4)
            k_local = (EI / L**3) * np.array([
                [12, 6*L, -12, 6*L],
                [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L],
                [6*L, 2*L**2, -6*L, 4*L**2]
            ])
            
            # Map to Global Indices
            # Node i -> DOFs 2*i, 2*i+1
            # Node i+1 -> DOFs 2*(i+1), 2*(i+1)+1
            idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
            
            for r in range(4):
                for c in range(4):
                    self.K[idx[r], idx[c]] += k_local[r, c]

        # 2. Apply Loads (Fixed End Forces)
        cum_len = [0] + list(np.cumsum(self.spans))
        
        for l in self.loads:
            span_idx = l['span_idx']
            L = self.spans[span_idx]
            
            # Indices for Left and Right nodes of this span
            i_L, i_R = 2*span_idx, 2*(span_idx+1)
            
            fem_shear_L = 0
            fem_moment_L = 0
            fem_shear_R = 0
            fem_moment_R = 0
            
            if l['type'] == 'U':
                w = l['w']
                # FEM for Uniform Load
                fem_shear_L = -w * L / 2
                fem_moment_L = -w * L**2 / 12
                fem_shear_R = -w * L / 2
                fem_moment_R = w * L**2 / 12
                
            elif l['type'] == 'P':
                P = l['P']
                a = l['x']  # distance from left node
                b = L - a
                # FEM for Point Load
                fem_shear_L = -P * b**2 * (3*a + b) / L**3
                fem_moment_L = -P * a * b**2 / L**2
                fem_shear_R = -P * a**2 * (a + 3*b) / L**3
                fem_moment_R = P * a**2 * b / L**2

            # Add to Global Force Vector (Subtract FEM because F = K*u + FEM -> K*u = F_ext - FEM)
            # Since external joint loads are 0, F = -FEM
            self.F[i_L]   -= fem_shear_L
            self.F[i_L+1] -= fem_moment_L
            self.F[i_R]   -= fem_shear_R
            self.F[i_R+1] -= fem_moment_R

        # 3. Apply Boundary Conditions
        fixed_dofs = []
        for _, row in self.supports.iterrows():
            node_idx = int(row['id'])
            stype = row['type']
            
            if stype == "Pin":
                fixed_dofs.append(2*node_idx)     # Fix Y
            elif stype == "Roller":
                fixed_dofs.append(2*node_idx)     # Fix Y
            elif stype == "Fixed":
                fixed_dofs.append(2*node_idx)     # Fix Y
                fixed_dofs.append(2*node_idx+1)   # Fix Theta

        # 4. Solve Matrix (Partitioning)
        free_dofs = [i for i in range(self.n_dof) if i not in fixed_dofs]
        
        K_ff = self.K[np.ix_(free_dofs, free_dofs)]
        F_f = self.F[free_dofs]
        
        U_f = np.linalg.solve(K_ff, F_f)
        
        self.U[free_dofs] = U_f  # Fill back calculated displacements

        # 5. Post-Processing (Calculate Internal Forces along the beam)
        return self._calculate_internal_forces(cum_len)

    def _calculate_internal_forces(self, cum_len):
        x_plot = []
        shear_plot = []
        moment_plot = []
        
        EI = 1.0
        
        for i, L in enumerate(self.spans):
            # Nodal Displacements for this element
            u_ele = self.U[[2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]]
            
            # Element Stiffness
            k_local = (EI / L**3) * np.array([
                [12, 6*L, -12, 6*L],
                [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L],
                [6*L, 2*L**2, -6*L, 4*L**2]
            ])
            
            # Nodal Forces (from stiffness)
            f_ele = np.dot(k_local, u_ele) # [Fy1, M1, Fy2, M2] convention
            
            # Start Forces (Left Node)
            V_start = f_ele[0]
            M_start = f_ele[1] # Counter-clockwise positive
            
            # Discretize span
            x_local = np.linspace(0, L, 100)
            
            # Add exact points for loads to catch peaks
            span_loads = [l for l in self.loads if l['span_idx'] == i]
            for l in span_loads:
                if l['type'] == 'P':
                    x_local = np.append(x_local, [l['x'], l['x']-1e-6, l['x']+1e-6])
            x_local = np.sort(np.unique(x_local))
            x_local = x_local[(x_local>=0) & (x_local<=L)]

            # Calculate V(x) and M(x) using statics
            for x in x_local:
                # Base from Node Forces
                # Beam Sign Convention: Upward Shear (+), Sagging Moment (+)
                # Matrix Convention: Upward Force (+), CCW Moment (+)
                
                # Shear at x = V_start + sum(Loads)
                # Moment at x = -M_start + V_start*x + sum(Load Moments)
                # Note: Matrix M1 is CCW. Beam Theory M is Sagging(+). So M_beam = -M_matrix
                
                V_x = V_start
                M_x = -M_start + V_start * x 
                
                # Add Load Effects
                for l in span_loads:
                    if l['type'] == 'U':
                        # Dist Load w (downward positive in input, but math needs direction)
                        # Input w is magnitude. Assuming gravity loads -> Downward.
                        # Force = -w * x_covered
                        if x > 0:
                            w = l['w']
                            V_x -= w * x
                            M_x -= w * x**2 / 2
                    elif l['type'] == 'P':
                        if x > l['x']:
                            P = l['P']
                            V_x -= P
                            M_x -= P * (x - l['x'])
                
                x_plot.append(cum_len[i] + x)
                shear_plot.append(V_x)
                moment_plot.append(M_x)
                
        # Handle Reaction Calculation (Optional/Simple)
        reactions = {} # Simplified for now
        
        return pd.DataFrame({'x': x_plot, 'shear': shear_plot, 'moment': moment_plot}), reactions
