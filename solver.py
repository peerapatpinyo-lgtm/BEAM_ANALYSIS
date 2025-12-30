import numpy as np
import pandas as pd

class BeamSolver:
    def __init__(self, spans, supports, loads):
        self.spans = spans
        self.supports = supports
        self.loads = loads
        self.n_nodes = len(spans) + 1
        self.n_dof = 2 * self.n_nodes  # 2 DOF per node (Deflection Y, Rotation Theta)
        
        self.K = np.zeros((self.n_dof, self.n_dof))
        self.F = np.zeros(self.n_dof) # Global Force Vector
        self.U = np.zeros(self.n_dof) # Global Displacement Vector
        
    def _get_fixed_end_forces(self, span_idx, L):
        """
        Calculate Fixed End Actions (Reactions) for a single span.
        Sign Convention for Vector: Y (Up+), M (CCW+)
        Input Load is assumed Gravity (Down).
        """
        # fem = [Fy_L, M_L, Fy_R, M_R]
        fem = np.zeros(4)
        
        span_loads = [l for l in self.loads if l['span_idx'] == span_idx]
        
        for l in span_loads:
            if l['type'] == 'U':
                w = l['w'] # Magnitude
                # Reactions for uniform load w (down)
                # Fy = wL/2 (Up+)
                # M_L = wL^2/12 (CCW+)
                # M_R = -wL^2/12 (CW-)
                fem[0] += w * L / 2
                fem[1] += w * L**2 / 12
                fem[2] += w * L / 2
                fem[3] += -w * L**2 / 12
                
            elif l['type'] == 'P':
                P = l['P'] # Magnitude
                a = l['x']
                b = L - a
                # Reactions for point load P (down)
                # Fy_L = Pb^2(3a+b)/L^3
                # M_L = Pab^2/L^2
                # Fy_R = Pa^2(a+3b)/L^3
                # M_R = -Pa^2b/L^2
                fem[0] += P * b**2 * (3*a + b) / L**3
                fem[1] += P * a * b**2 / L**2
                fem[2] += P * a**2 * (a + 3*b) / L**3
                fem[3] += -P * a**2 * b / L**2
                
        return fem

    def solve(self):
        # 1. Build Global Stiffness Matrix
        EI = 1.0e6 # Dummy EI (high enough), relative stiffness matters
        
        for i, L in enumerate(self.spans):
            # Element Stiffness
            k_local = (EI / L**3) * np.array([
                [12, 6*L, -12, 6*L],
                [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L],
                [6*L, 2*L**2, -6*L, 4*L**2]
            ])
            
            idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
            for r in range(4):
                for c in range(4):
                    self.K[idx[r], idx[c]] += k_local[r, c]
                    
            # 2. Add Fixed End Forces (Load Vector)
            # Global Force = External Nodal Force - Fixed End Reactions
            # F_node = K*u + F_fixed_reaction
            # So, K*u = F_node - F_fixed_reaction
            
            fem = self._get_fixed_end_forces(i, L)
            
            self.F[idx[0]] -= fem[0]
            self.F[idx[1]] -= fem[1]
            self.F[idx[2]] -= fem[2]
            self.F[idx[3]] -= fem[3]

        # 3. Apply Supports (Boundary Conditions)
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

        # 4. Solve for Displacements
        free_dofs = [i for i in range(self.n_dof) if i not in fixed_dofs]
        
        if len(free_dofs) > 0:
            K_ff = self.K[np.ix_(free_dofs, free_dofs)]
            F_f = self.F[free_dofs]
            try:
                U_f = np.linalg.solve(K_ff, F_f)
                self.U[free_dofs] = U_f
            except:
                return None, None # Unstable
        
        # 5. Post-Process (Internal Forces)
        return self._calculate_internal_forces(EI)

    def _calculate_internal_forces(self, EI):
        x_plot = []
        shear_plot = []
        moment_plot = []
        
        cum_len = [0] + list(np.cumsum(self.spans))
        
        # Array to store reaction forces at each node
        node_reactions = np.zeros(self.n_dof)
        
        for i, L in enumerate(self.spans):
            # Element Displacements
            u_ele = self.U[[2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]]
            
            # Stiffness Force (K * u)
            k_local = (EI / L**3) * np.array([
                [12, 6*L, -12, 6*L],
                [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L],
                [6*L, 2*L**2, -6*L, 4*L**2]
            ])
            f_stiff = np.dot(k_local, u_ele)
            
            # Add Fixed End Forces back to get Total Nodal Forces
            # F_total = K*u + FEM
            f_fixed = self._get_fixed_end_forces(i, L)
            f_total = f_stiff + f_fixed
            
            # Save these forces to calculate Reactions later
            # Force exerted BY MEMBER ON NODE
            # Reaction is Force exerted BY SUPPORT ON NODE
            # So Reaction = Sum(Member Forces) (roughly, handled at global level usually)
            
            # Start Forces for Statics
            # V_start (Left Up+) = f_total[0]
            # M_start (Left CCW+) = f_total[1]
            V_start = f_total[0]
            M_start = f_total[1]
            
            # Sampling Points (High Resolution + Key Points)
            x_local = [0.0, L] # Boundary
            
            # Add Load Points
            span_loads = [l for l in self.loads if l['span_idx'] == i]
            for l in span_loads:
                if l['type'] == 'P':
                    x_local.extend([l['x'], l['x']-1e-5, l['x']+1e-5])
            
            # Add dense points
            x_local.extend(np.linspace(0, L, 101))
            x_local = np.sort(np.unique(x_local))
            x_local = x_local[(x_local >= 0) & (x_local <= L)]
            
            for x in x_local:
                # Calculate V(x), M(x) using Free Body Diagram from Left Node
                # V(x) = V_start - Sum(Loads)
                # M(x) = -M_start + V_start*x - Sum(Load Moments)
                # Note: M_start is CCW(+), Internal Moment Sagging(+) 
                # At x=0, Moment should be -M_start (because CCW reaction bends beam down/hogging)
                
                V_x = V_start
                M_x = -M_start + V_start * x
                
                for l in span_loads:
                    if l['type'] == 'U':
                        w = l['w']
                        # Force = w * x
                        # Moment arm = x/2
                        if x > 0:
                            V_x -= w * x
                            M_x -= w * x**2 / 2
                    elif l['type'] == 'P':
                        P = l['P']
                        if x > l['x']:
                            V_x -= P
                            M_x -= P * (x - l['x'])
                
                x_plot.append(cum_len[i] + x)
                shear_plot.append(V_x)
                moment_plot.append(M_x)
                
            # Accumulate Global Reactions (Global Force Balance)
            # R = K*U - F_ext + F_fixed ??
            # Easier: R = Total Force at node from elements
            # Left Node of span
            node_reactions[2*i] += f_total[0]   # Fy
            node_reactions[2*i+1] += f_total[1] # Mz
            # Right Node of span
            node_reactions[2*(i+1)] += f_total[2]   # Fy
            node_reactions[2*(i+1)+1] += f_total[3] # Mz

        return pd.DataFrame({'x': x_plot, 'shear': shear_plot, 'moment': moment_plot}), node_reactions
