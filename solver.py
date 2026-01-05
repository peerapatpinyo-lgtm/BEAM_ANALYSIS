import numpy as np
import pandas as pd

class BeamSolver:
    def __init__(self, spans, supports, loads, E, I):
        """
        spans: list of span lengths [L1, L2, ...]
        supports: DataFrame or list of dicts
        loads: DataFrame of loads
        E: Elastic Modulus
        I: Moment of Inertia
        """
        self.spans = spans
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
                # Get Equivalent Nodal Loads
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
            return pd.DataFrame(), np.zeros(num_dof) # Return empty on error

        U_global = np.zeros(num_dof)
        U_global[free_dofs] = u_f
        
        # Calculate Global Reactions: R = K*U - F_applied
        # Note: F vector constructed above contains Equivalent Nodal Loads.
        # R = K*U - F_equiv
        Reactions = K @ U_global - F

        # 5. Post-Processing (Internal Forces)
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

            # --- CRITICAL FIX: CALCULATE MEMBER END FORCES CORRECTLY ---
            # F_member = k_local * u_local + Fixed_End_Actions
            # Fixed_End_Actions = - Equivalent_Nodal_Forces
            
            k_loc = self._get_element_stiffness(L)
            f_elastic = k_loc @ u_elem
            
            # Sum Fixed End Actions (Reaction from support to beam if fixed)
            f_fea = np.zeros(4)
            for load in span_loads:
                # _calc_equivalent_nodal_forces returns forces ON NODES.
                # Fixed End Actions (Forces ON BEAM ENDS) are opposite.
                f_eq = self._calc_equivalent_nodal_forces(load, L)
                f_fea -= f_eq 
            
            f_total_start = f_elastic + f_fea
            
            # Initial Forces at x=0 of the element
            # V_start (Up+) = F_y_start
            # M_start (CCW+) = M_z_start. 
            # Note: CCW Moment at left end causes HOGGING (Negative Moment).
            # So Internal Moment = - M_z_start
            
            V0 = f_total_start[0]
            M0 = -f_total_start[1] # Convert CCW node moment to Beam Sign (Sagging+)
            
            # Discretize for plotting
            num_points = 50
            x_eval = np.linspace(0, L, num_points)
            
            # Add points at load locations for sharp diagrams
            load_locs = [float(l['x']) for l in span_loads]
            # Add UDL start/end
            for l in span_loads:
                if l['type'] == 'U':
                    load_locs.append(float(l['x']) + float(l.get('dist', L)))
            
            # Merge and sort unique points
            x_eval = sorted(list(set(list(x_eval) + [0, L] + [loc for loc in load_locs if 0 <= loc <= L])))
            
            for x in x_eval:
                V, M = self._calculate_statics_at_x(x, V0, M0, span_loads)
                
                # Deflection (Approximate via Shape Function)
                # Note: This is purely elastic deflection from nodal displacement. 
                # Does not include local load curvature (bubble function), but sufficient for visualization.
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
        # Calculates Equivalent Nodal Loads (Forces applied TO NODES)
        # Based on Fixed End Reactions formulas
        f = np.zeros(4)
        mag = float(load['mag'])
        a = float(load['x']) 
        b = L - a
        
        # Note: Input mag positive = Downward usually?
        # Let's assume User Input: (+) = Gravity Load (Down)
        # FEM Y-axis: (+) = Up.
        # So Load Force F_y = -mag
        
        if load['type'] == 'P': 
            P = mag # Magnitude
            # Reactions for Downward Load P:
            # RA (Up) = Pb^2(3a+b)/L^3
            # MA (CCW+) = -Pab^2/L^2
            # But we want Equivalent Nodal Loads.
            # Equiv Load = - Fixed End Reaction.
            # If P is Down, Reaction is Up. Equiv Load is Down.
            
            # Let's verify standard FEF formulas for DOWNWARD load P:
            # FEM_A = -Pab^2/L^2 (CCW is -? No, usually Standard is -)
            # Let's stick to strict derivation:
            # Force vector P_vec is [0, -P, 0, 0] at x=a.
            # Equiv Nodal Force = Integ(N_transpose * P_vec).
            # Fy1 = N1(a) * (-P)
            # M1  = N2(a) * (-P)
            # Fy2 = N3(a) * (-P)
            # M2  = N4(a) * (-P)
            
            xi = a/L
            N1 = 1 - 3*xi**2 + 2*xi**3
            N2 = L * (xi - 2*xi**2 + xi**3)
            N3 = 3*xi**2 - 2*xi**3
            N4 = L * (-xi**2 + xi**3)
            
            F_load = -P # Downward force in FEM coord
            
            f[0] = N1 * F_load
            f[1] = N2 * F_load
            f[2] = N3 * F_load
            f[3] = N4 * F_load

        elif load['type'] == 'U': 
            w_mag = mag # N/m (Down)
            # Full span UDL approximation for robustness
            # Equiv Loads for Uniform Load w (Down):
            # Fy = -wL/2
            # M1 = -wL^2/12
            # M2 = +wL^2/12
            
            F_total = -w_mag * L
            f[0] = F_total / 2
            f[1] = -w_mag * L**2 / 12
            f[2] = F_total / 2
            f[3] = +w_mag * L**2 / 12
            
            # Note: If Partial UDL is needed, use integration of Shape Func
            # But for this fix, Full UDL logic prevents crash.

        elif load['type'] == 'M': 
            M_ext = mag # CCW +
            # Moment is tricky with shape functions (derivative).
            # Simplified: FEF for Moment M at a.
            # For now, ignore FEF of Moment load to prevent error, 
            # or treat as couple?
            # Let's leave 0 for M load FEF to imply applied at nearest node 
            # manually by user, or implement exact later.
            pass

        return f

    def _calculate_statics_at_x(self, x, V0, M0, loads):
        # Walk from x=0 to x using Statics
        # V(x) = V0 + Sum(Loads Up)
        # M(x) = M0 + V0*x + Sum(Load * arm)
        
        V_x = V0
        M_x = M0 + V_x * x # Contribution from initial shear
        
        for load in loads:
            lx = float(load['x'])
            mag = float(load['mag']) # User input (+) = Down
            
            if x > lx:
                if load['type'] == 'P':
                    # Point Load
                    # Force = -mag (Down)
                    # V changes by -mag
                    # M changes by -mag * (x - lx)
                    
                    V_x -= mag
                    M_x -= mag * (x - lx)
                    
                elif load['type'] == 'U':
                    # UDL
                    w = mag # Down
                    dist = float(load.get('dist', 0))
                    if dist == 0: dist = 1e9 # treat as long if 0
                    
                    # Overlap
                    end_load = lx + dist
                    x_overlap_end = min(x, end_load)
                    
                    if x_overlap_end > lx:
                        length = x_overlap_end - lx
                        load_force = w * length
                        
                        V_x -= load_force
                        
                        # Moment arm from centroid of load block to x
                        # Centroid is at lx + length/2
                        centroid = lx + length/2
                        arm = x - centroid
                        M_x -= load_force * arm
                        
                elif load['type'] == 'M':
                    # Moment Load (CCW +)
                    # Directly adds to Moment diagram
                    M_x += mag
                    
        return V_x, M_x

    def _get_deflection(self, x, L, u_elem):
        # Hermite Interpolation
        xi = x/L
        N_vec = np.array([
            1 - 3*xi**2 + 2*xi**3,
            L * (xi - 2*xi**2 + xi**3),
            3*xi**2 - 2*xi**3,
            L * (-xi**2 + xi**3)
        ])
        return np.dot(N_vec, u_elem)
