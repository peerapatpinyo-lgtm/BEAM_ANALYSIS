import numpy as np
import pandas as pd

class BeamSolver:
    def __init__(self, spans, supports, loads, E, I):
        """
        spans: list of span lengths [L1, L2, ...]
        supports: DataFrame [{'id': node_idx, 'type': 'Pin'/'Roller'/'Fixed'}]
        loads: DataFrame of loads
        E: Elastic Modulus
        I: Moment of Inertia
        """
        self.spans = spans
        self.supports = supports
        self.loads = loads
        self.E = float(E)
        self.I = float(I)
        self.num_nodes = len(spans) + 1
        self.num_dof = 2 * self.num_nodes  # 2 DOF per node (Vertical Y, Rotation Z)

    def solve(self):
        # 1. Prepare Global Stiffness Matrix (K) & Force Vector (F)
        K = np.zeros((self.num_dof, self.num_dof))
        F = np.zeros(self.num_dof)
        
        # 2. Build Stiffness Matrix (Element by Element)
        for i, L in enumerate(self.spans):
            k_local = self._get_element_stiffness(L)
            
            # Global Indices for this element
            idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
            
            # Assembly K
            for r in range(4):
                for c in range(4):
                    K[idx[r], idx[c]] += k_local[r, c]

            # 3. Process Loads for Fixed End Forces (FEF)
            # --- FIX: Ensure span_loads is always a list ---
            if not self.loads.empty:
                span_loads_df = self.loads[self.loads['span_idx'] == i]
                span_loads = span_loads_df.to_dict('records')
            else:
                span_loads = []
            # -----------------------------------------------
            
            for load in span_loads:
                f_local = self._calc_fixed_end_forces(load, L)
                
                # Add equivalent nodal forces to F vector
                F[idx[0]] += f_local[0] # Fy1
                F[idx[1]] += f_local[1] # M1
                F[idx[2]] += f_local[2] # Fy2
                F[idx[3]] += f_local[3] # M2

        # 4. Apply Boundary Conditions
        fixed_dofs = []
        # Ensure supports is DataFrame
        if isinstance(self.supports, list): 
            df_sup = pd.DataFrame(self.supports)
        else:
            df_sup = self.supports

        for _, sup in df_sup.iterrows():
            node_idx = int(sup['id'])
            sType = sup['type']
            
            if sType == 'Pin':
                fixed_dofs.append(2*node_idx) # Fix Vertical
            elif sType == 'Roller':
                fixed_dofs.append(2*node_idx) # Fix Vertical
            elif sType == 'Fixed':
                fixed_dofs.append(2*node_idx)   # Fix Vertical
                fixed_dofs.append(2*node_idx+1) # Fix Rotation

        fixed_dofs = sorted(list(set(fixed_dofs)))
        free_dofs = [i for i in range(self.num_dof) if i not in fixed_dofs]
        
        # Partition and Solve
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        F_f = F[free_dofs]
        
        try:
            u_f = np.linalg.solve(K_ff, F_f)
        except np.linalg.LinAlgError:
            raise ValueError("Structure is unstable or singular matrix.")

        # Reconstruct full displacement vector
        U_global = np.zeros(self.num_dof)
        U_global[free_dofs] = u_f
        
        # Calculate Reactions: R = K*U - F_applied
        Reactions = K @ U_global - F
        
        # 5. Post-Processing
        results = []
        x_cursor = 0
        
        for i, L in enumerate(self.spans):
            # Element displacements
            idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
            u_elem = U_global[idx]
            
            # --- FIX: Ensure span_loads is always a list ---
            if not self.loads.empty:
                span_loads_df = self.loads[self.loads['span_idx'] == i]
                span_loads = span_loads_df.to_dict('records')
            else:
                span_loads = []
            # -----------------------------------------------

            # Discretize element for smooth diagrams
            x_eval = np.linspace(0, L, 50)
            
            for x in x_eval:
                V, M, D = self._get_internal_forces(x, L, u_elem, span_loads)
                results.append({
                    'x': x_cursor + x,
                    'shear': V,
                    'moment': M,
                    'deflection': D
                })
            
            x_cursor += L

        return pd.DataFrame(results), Reactions

    def _get_element_stiffness(self, L):
        E, I = self.E, self.I
        k = np.array([
            [12*E*I/L**3,  6*E*I/L**2, -12*E*I/L**3, 6*E*I/L**2],
            [6*E*I/L**2,   4*E*I/L,    -6*E*I/L**2,  2*E*I/L],
            [-12*E*I/L**3, -6*E*I/L**2, 12*E*I/L**3, -6*E*I/L**2],
            [6*E*I/L**2,   2*E*I/L,    -6*E*I/L**2,  4*E*I/L]
        ])
        return k

    def _calc_fixed_end_forces(self, load, L):
        f = np.zeros(4)
        mag = float(load['mag'])
        a = float(load['x']) 
        b = L - a
        
        if load['type'] == 'P': 
            P = mag 
            F_y = -P
            
            fem_1 = -F_y * a * b**2 / L**2
            fem_2 = +F_y * a**2 * b / L**2
            fea_1 = -F_y * b**2 * (3*a + b) / L**3
            fea_2 = -F_y * a**2 * (a + 3*b) / L**3
            
            f[0] = -fea_1; f[1] = -fem_1; f[2] = -fea_2; f[3] = -fem_2

        elif load['type'] == 'U': 
            # Uniform Load (Full Span Logic for robustness)
            w = -mag 
            f[0] = w * L / 2
            f[1] = -w * L**2 / 12
            f[2] = w * L / 2
            f[3] = +w * L**2 / 12

        elif load['type'] == 'M': 
            M = mag 
            R_A = -6*M*a*b / L**3
            R_B = +6*M*a*b / L**3
            M_A = M*b*(b - 2*a) / L**2
            M_B = M*a*(a - 2*b) / L**2
            
            f[0] = -R_A; f[1] = -M_A; f[2] = -R_B; f[3] = -M_B

        return f

    def _get_internal_forces(self, x, L, u_elem, loads):
        E, I = self.E, self.I
        
        # Shape Functions for Deflection (Hermite)
        xi = x/L
        N_vec = np.array([
            1 - 3*xi**2 + 2*xi**3,
            L * (xi - 2*xi**2 + xi**3),
            3*xi**2 - 2*xi**3,
            L * (-xi**2 + xi**3)
        ])
        D_elastic = np.dot(N_vec, u_elem)
        
        # Calculate forces at Start Node (Left) from Stiffness
        # F_start_elastic = k_local * u
        k_loc = self._get_element_stiffness(L)
        f_int = k_loc @ u_elem 
        
        # Subtract Fixed End Actions (to get Total Nodal Force)
        fe_actions = np.zeros(4)
        for load in loads:
            f_equiv = self._calc_fixed_end_forces(load, L)
            fe_actions -= f_equiv 
            
        F_total_start = f_int + fe_actions
        
        V_start = F_total_start[0] # Upward Force on beam start
        M_start = F_total_start[1] # CCW Moment on beam start
        
        # Statics walk from 0 to x
        V_x = V_start
        M_x = M_start 
        D_x = D_elastic # Approximation
        
        for load in loads:
            mag = float(load['mag'])
            lx = float(load['x'])
            
            if x >= lx:
                if load['type'] == 'P':
                    P = mag
                    V_x -= P 
                    M_x -= P * (x - lx)
                elif load['type'] == 'U':
                    w = mag
                    dist = float(load.get('dist', L))
                    # Overlap length
                    dx_overlap = min(x, lx + dist) - lx
                    if dx_overlap > 0:
                        load_total = w * dx_overlap
                        centroid_dist = (x - lx) - dx_overlap/2 # Dist from x to load centroid
                        V_x -= load_total
                        # M_x -= load_total * centroid_dist 
                        # Correct arm calculation:
                        # Load acts at center of overlap block.
                        # Center of overlap block is lx + dx_overlap/2
                        # Arm = x - (lx + dx_overlap/2)
                        arm = x - (lx + dx_overlap/2)
                        M_x -= load_total * arm
                elif load['type'] == 'M':
                    M_ext = mag 
                    M_x += M_ext 
        
        # Sign Convention Adjustment for Plotting
        # Standard: Sagging Moment (+), Hogging Moment (-)
        # Our M_start is CCW. CCW at left end causes Hogging (Top Tension).
        # So M_plot should be -M_calc
        
        return V_x, -M_x, D_x
