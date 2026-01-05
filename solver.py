import numpy as np
import pandas as pd

class BeamSolver:
    def __init__(self, spans, supports, loads, E, I):
        """
        spans: list of span lengths [L1, L2, ...]
        supports: DataFrame or list of dicts [{'id': node_idx, 'type': 'Pin'/'Roller'/'Fixed'}]
        loads: DataFrame of loads
        E: Elastic Modulus
        I: Moment of Inertia
        """
        self.spans = spans
        # Ensure supports is a DataFrame
        self.supports = supports if isinstance(supports, pd.DataFrame) else pd.DataFrame(supports)
        self.loads = loads
        self.E = float(E)
        self.I = float(I)
        self.num_nodes = len(spans) + 1
        self.num_dof = 2 * self.num_nodes  # 2 DOF per node (Vertical Y, Rotation Z)

    def solve(self):
        # 1. Setup Global Matrix
        num_dof = self.num_dof
        K = np.zeros((num_dof, num_dof))
        F = np.zeros(num_dof)
        
        # 2. Build Stiffness & Load Vector
        for i, L in enumerate(self.spans):
            # Local Stiffness
            k_local = self._get_element_stiffness(L)
            
            # DOF mapping
            idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
            
            # Assemble K
            for r in range(4):
                for c in range(4):
                    K[idx[r], idx[c]] += k_local[r, c]

            # Assemble F (Equivalent Nodal Forces)
            if not self.loads.empty:
                span_loads = self.loads[self.loads['span_idx'] == i].to_dict('records')
            else:
                span_loads = []
            
            for load in span_loads:
                f_equiv = self._calc_equivalent_nodal_forces(load, L)
                F[idx[0]] += f_equiv[0]
                F[idx[1]] += f_equiv[1]
                F[idx[2]] += f_equiv[2]
                F[idx[3]] += f_equiv[3]

        # 3. Apply Boundary Conditions
        fixed_dofs = []
        for _, sup in self.supports.iterrows():
            node_idx = int(sup['id'])
            sType = sup['type']
            if sType in ['Pin', 'Roller']:
                fixed_dofs.append(2*node_idx) # Fix Vertical
            elif sType == 'Fixed':
                fixed_dofs.append(2*node_idx)   # Fix Vertical
                fixed_dofs.append(2*node_idx+1) # Fix Rotation
        
        fixed_dofs = sorted(list(set(fixed_dofs)))
        free_dofs = [i for i in range(num_dof) if i not in fixed_dofs]
        
        # 4. Solve for Displacements
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        F_f = F[free_dofs]
        
        try:
            u_f = np.linalg.solve(K_ff, F_f)
        except np.linalg.LinAlgError:
            return pd.DataFrame(), np.zeros(num_dof) # Singular Matrix

        U_global = np.zeros(num_dof)
        U_global[free_dofs] = u_f
        
        # Calculate Reactions
        Reactions = K @ U_global - F

        # 5. Post-Processing (Internal Forces with Epsilon Points for Sharp Graphs)
        results = []
        x_cursor = 0
        
        for i, L in enumerate(self.spans):
            idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
            u_elem = U_global[idx]
            
            if not self.loads.empty:
                span_loads = self.loads[self.loads['span_idx'] == i].to_dict('records')
            else:
                span_loads = []

            # Calculate Member Start Forces (Statics Initialization)
            k_loc = self._get_element_stiffness(L)
            f_elastic = k_loc @ u_elem
            
            f_fea = np.zeros(4)
            for load in span_loads:
                f_eq = self._calc_equivalent_nodal_forces(load, L)
                f_fea -= f_eq 
            
            f_total_start = f_elastic + f_fea
            
            V0 = f_total_start[0]
            M0 = -f_total_start[1] # Convert FEM moment to Beam Convention
            
            # --- FIX: Generate Dense Points + Epsilon Points ---
            # 1. Basic points
            x_eval = set(np.linspace(0, L, 100))
            x_eval.update([0, L])
            
            # 2. Add Critical Load Locations
            load_locs = set()
            for l in span_loads:
                lx = float(l['x'])
                load_locs.add(lx)
                if l['type'] == 'U':
                    load_locs.add(lx + float(l.get('dist', L)))
            
            # 3. Add Epsilon Points (+/- small value) to force vertical lines
            eps = 1e-10
            for loc in load_locs:
                if 0 <= loc <= L:
                    x_eval.add(loc)
                    # Check bounds before adding epsilon points
                    if loc - eps >= 0: x_eval.add(loc - eps)
                    if loc + eps <= L: x_eval.add(loc + eps)
            
            unique_points = sorted(list(x_eval))
            
            for x in unique_points:
                V, M = self._calculate_statics_at_x(x, V0, M0, span_loads)
                D = self._get_deflection(x, L, u_elem)
                
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

    def _calc_equivalent_nodal_forces(self, load, L):
        f = np.zeros(4)
        mag = float(load['mag']) 
        a = float(load['x']) 
        # Convert User Input (Down+) to FEM Y-axis (Up+)
        F_load = -mag 
        
        if load['type'] == 'P': 
            xi = a/L
            N1 = 1 - 3*xi**2 + 2*xi**3
            N2 = L * (xi - 2*xi**2 + xi**3)
            N3 = 3*xi**2 - 2*xi**3
            N4 = L * (-xi**2 + xi**3)
            
            f[0] = N1 * F_load
            f[1] = N2 * F_load
            f[2] = N3 * F_load
            f[3] = N4 * F_load

        elif load['type'] == 'U': 
            # Full UDL approximation for Nodal Loads
            # Exact Fixed End Actions for w (Down):
            # Fy = wL/2 (Up reaction) -> Equiv Load = Down
            # M_left = wL^2/12 (CCW reaction) -> Equiv Load = CW (-M)
            # M_right = -wL^2/12 (CW reaction) -> Equiv Load = CCW (+M)
            
            # F_load is negative (Down).
            # So Total Force is negative.
            f[0] = F_load * L / 2
            f[1] = -abs(F_load) * L**2 / 12 # Ensure Negative Moment (CW) at Node 1
            f[2] = F_load * L / 2
            f[3] = +abs(F_load) * L**2 / 12 # Ensure Positive Moment (CCW) at Node 2

        return f

    def _calculate_statics_at_x(self, x, V0, M0, loads):
        V_x = V0
        M_x = M0 + V0 * x 
        
        for load in loads:
            lx = float(load['x'])
            mag = float(load['mag']) # User input (+ = Down)
            
            # Use strict inequality for P loads to allow 'jump' logic
            if x > lx:
                if load['type'] == 'P':
                    V_x -= mag
                    M_x -= mag * (x - lx)
                    
                elif load['type'] == 'U':
                    w = mag
                    dist = float(load.get('dist', 1e9))
                    start_load = lx
                    end_load = lx + dist
                    
                    # Effective overlap
                    x_eff_end = min(x, end_load)
                    length = x_eff_end - start_load
                    
                    if length > 0:
                        force = w * length
                        V_x -= force
                        centroid = start_load + length/2
                        arm = x - centroid
                        M_x -= force * arm
        
        return V_x, M_x

    def _get_deflection(self, x, L, u_elem):
        xi = x/L
        N_vec = np.array([
            1 - 3*xi**2 + 2*xi**3,
            L * (xi - 2*xi**2 + xi**3),
            3*xi**2 - 2*xi**3,
            L * (-xi**2 + xi**3)
        ])
        return np.dot(N_vec, u_elem)
