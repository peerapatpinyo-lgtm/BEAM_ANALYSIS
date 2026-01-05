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
            
            # DOF mapping: Node i -> 2*i, 2*i+1
            idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
            
            # Assemble K
            for r in range(4):
                for c in range(4):
                    K[idx[r], idx[c]] += k_local[r, c]

            # Assemble F (Equivalent Nodal Forces from Loads)
            # Filter loads on this span
            if not self.loads.empty:
                span_loads = self.loads[self.loads['span_idx'] == i].to_dict('records')
            else:
                span_loads = []
            
            for load in span_loads:
                # Get Equivalent Nodal Loads (Forces applied TO nodes)
                f_equiv = self._calc_equivalent_nodal_forces(load, L)
                # Add to Global F
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
            return pd.DataFrame(), np.zeros(num_dof) # Singular matrix

        U_global = np.zeros(num_dof)
        U_global[free_dofs] = u_f
        
        # Calculate Global Reactions: R = K*U - F_applied
        Reactions = K @ U_global - F

        # 5. Post-Processing (Internal Forces Calculation)
        results = []
        x_cursor = 0
        
        for i, L in enumerate(self.spans):
            # Element DOFs
            idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
            u_elem = U_global[idx]
            
            # Loads on this span
            if not self.loads.empty:
                span_loads = self.loads[self.loads['span_idx'] == i].to_dict('records')
            else:
                span_loads = []

            # --- KEY FIX: Calculate Member End Forces correctly ---
            # F_member_elastic = k_local * u_local
            # Total F_member_start = F_member_elastic + Fixed_End_Actions
            
            k_loc = self._get_element_stiffness(L)
            f_elastic = k_loc @ u_elem
            
            # Fixed End Actions = - Equivalent Nodal Forces
            f_fea = np.zeros(4)
            for load in span_loads:
                f_eq = self._calc_equivalent_nodal_forces(load, L)
                f_fea -= f_eq 
            
            f_total_start = f_elastic + f_fea
            
            # Extract Start Forces for Statics Walk
            # Node Forces: Fy (Up+), M (CCW+)
            # Beam Internal Forces at x=0:
            # Shear V = Fy_start (Up is positive Shear on left face)
            # Moment M = - M_start (CCW Moment at left support causes HOGGING/Tension Top. 
            #                       Standard sign convention: Sagging/Tension Bottom is Positive.
            #                       Therefore, Internal Moment = - Node Moment)
            
            V0 = f_total_start[0]
            M0 = -f_total_start[1] 
            
            # Create evaluation points (add load locations for precision)
            num_points = 50
            x_eval = np.linspace(0, L, num_points)
            
            # Add critical points
            load_locs = [float(l['x']) for l in span_loads]
            for l in span_loads:
                if l['type'] == 'U':
                    load_locs.append(float(l['x']) + float(l.get('dist', L)))
            
            unique_points = sorted(list(set(list(x_eval) + [0, L] + [loc for loc in load_locs if 0 <= loc <= L])))
            
            for x in unique_points:
                V, M = self._calculate_statics_at_x(x, V0, M0, span_loads)
                
                # Simple elastic deflection curve (good approximation)
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
        # Calculates forces applied TO NODES to represent the load
        f = np.zeros(4)
        mag = float(load['mag']) # Assumed Input: + for Downward Gravity Load
        a = float(load['x']) 
        
        # Convert to FEM Coordinate System (Up +, Down -)
        # If input 1000 means 1000kg Down, then Force F = -1000
        F_load = -mag 
        
        if load['type'] == 'P': 
            # Use Shape Functions for Exact Nodal Load allocation
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
            # Uniform Load w (Force/Length)
            # Total Force = w * L
            # Equivalent Nodal Forces for Full UDL:
            # Fy = F_total / 2
            # M = F_total * L / 12 (Check signs: Left is -M, Right is +M for Down load)
            
            w_total = F_load * L # Total force
            f[0] = w_total / 2
            f[1] = w_total * L / 12  # FEM Moment at node 1 (CCW is +). Down load causes CCW reaction? No.
                                     # Fixed End Reaction for Down load: M_A = +wL^2/12 (CCW).
                                     # Equiv Load = - Reaction = -wL^2/12.
            
            # Let's double check Fixed End Moment Signs.
            # Downward Load (-w). 
            # Fixed End Reaction Moment at Left (MA): Positive (CCW) to resist rotation.
            # Equivalent Load Moment = - Reaction. So Negative.
            
            # Correct Standard Formula for Equiv Nodal Load of Downward UDL:
            # Fy = -wL/2
            # M1 = -wL^2/12
            # Fy2 = -wL/2
            # M2 = +wL^2/12
            
            f[0] = F_load * L / 2
            f[1] = -abs(F_load) * L**2 / 12 # Force negative, so this ensures -
            f[2] = F_load * L / 2
            f[3] = +abs(F_load) * L**2 / 12

        return f

    def _calculate_statics_at_x(self, x, V0, M0, loads):
        # Calculate V and M at distance x from left node using Statics
        # V(x) = V_start + Sum(Forces)
        # M(x) = M_start + V_start*x + Sum(Moment of Forces)
        
        # Start with Reaction effects
        V_x = V0
        M_x = M0 + V0 * x 
        
        for load in loads:
            lx = float(load['x'])
            mag = float(load['mag']) # Input (+ = Down)
            
            if x > lx:
                if load['type'] == 'P':
                    # Point Load P (Down)
                    # V drops by P
                    V_x -= mag
                    # M drops by P * arm
                    M_x -= mag * (x - lx)
                    
                elif load['type'] == 'U':
                    # UDL w (Down)
                    w = mag
                    dist = float(load.get('dist', 1e9)) # Full length if not specified
                    
                    # Calculate overlap length
                    start_load = lx
                    end_load = lx + dist
                    
                    # Where does the load effectively act relative to x?
                    if x > start_load:
                        x_eff_end = min(x, end_load)
                        length = x_eff_end - start_load
                        
                        if length > 0:
                            force = w * length
                            V_x -= force
                            
                            # Moment arm: distance from x to centroid of the load block
                            # Centroid of block is at (start + length/2)
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
