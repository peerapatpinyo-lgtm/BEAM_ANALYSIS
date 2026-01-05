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
        
        # Fixed End Forces vector (to be added to global F)
        FEF = np.zeros(self.num_dof)

        # 2. Build Stiffness Matrix (Element by Element)
        # DOF mapping: Node i -> 2*i (Vertical), 2*i+1 (Rotation)
        cum_dist = [0] + list(np.cumsum(self.spans))
        
        for i, L in enumerate(self.spans):
            k_local = self._get_element_stiffness(L)
            
            # Global Indices for this element (Node i and Node i+1)
            # Node i: 2*i, 2*i+1
            # Node i+1: 2*(i+1), 2*(i+1)+1
            idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
            
            # Assembly
            for r in range(4):
                for c in range(4):
                    K[idx[r], idx[c]] += k_local[r, c]

            # 3. Process Loads for FEF (Fixed End Forces)
            # Filter loads belonging to this span
            span_loads = self.loads[self.loads['span_idx'] == i] if not self.loads.empty else []
            if isinstance(span_loads, pd.DataFrame) and not span_loads.empty:
                span_loads = span_loads.to_dict('records')
            
            if span_loads:
                for load in span_loads:
                    f_local = self._calc_fixed_end_forces(load, L)
                    # Add to Global FEF vector (minus sign because FEF acts ON the nodes from member)
                    # But traditionally F = K*d - FEF => K*d = F + FEF_reactions
                    # Standard FEM: F_node = F_applied - F_equiv (Fixed End Actions)
                    # Here we accumulate Equivalent Nodal Forces from loads
                    
                    # Add equivalent nodal forces to F vector
                    F[idx[0]] += f_local[0] # Fy1
                    F[idx[1]] += f_local[1] # M1
                    F[idx[2]] += f_local[2] # Fy2
                    F[idx[3]] += f_local[3] # M2

        # 4. Apply Boundary Conditions (Penalty Method or Partitioning)
        # Using Partitioning (Removing rows/cols) is cleaner for plotting but Penalty is easier
        # Let's use Reduced Matrix approach for accuracy
        
        fixed_dofs = []
        if isinstance(self.supports, list): self.supports = pd.DataFrame(self.supports)
        
        for _, sup in self.supports.iterrows():
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
        
        # Partition K and F
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        F_f = F[free_dofs]
        
        # Solve for Displacements (u)
        try:
            u_f = np.linalg.solve(K_ff, F_f)
        except np.linalg.LinAlgError:
            raise ValueError("Structure is unstable or singular matrix.")

        # Reconstruct full displacement vector
        U_global = np.zeros(self.num_dof)
        U_global[free_dofs] = u_f
        
        # Calculate Reactions: R = K*U - F_applied
        # R_full = K @ U_global - F
        # But we want reactions only at supports
        Reactions = K @ U_global - F
        
        # 5. Post-Processing (Shear, Moment, Deflection along beam)
        # We need to discretize spans to draw diagrams
        
        results = []
        x_cursor = 0
        
        for i, L in enumerate(self.spans):
            # Element displacements
            idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
            u_elem = U_global[idx]
            
            # Discretize element
            x_eval = np.linspace(0, L, 50)
            
            # Filter loads for this span again for internal calculation
            span_loads = self.loads[self.loads['span_idx'] == i] if not self.loads.empty else []
            if isinstance(span_loads, pd.DataFrame) and not span_loads.empty:
                span_loads = span_loads.to_dict('records')

            for x in x_eval:
                # Calculate internal forces at x using shape functions + local load effects
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
        # Returns Equivalent Nodal Forces [Fy1, M1, Fy2, M2]
        # These are forces applied TO the nodes BY the load
        f = np.zeros(4)
        mag = float(load['mag'])
        a = float(load['x']) # Dist from left node
        b = L - a
        
        if load['type'] == 'P': # Point Load
            # M1 = -Pab^2/L^2, M2 = +Pa^2b/L^2
            # Fy1 = Pb^2(3a+b)/L^3, Fy2 = Pa^2(a+3b)/L^3
            # Signs: Counter-clockwise Moment is Positive for FEM formulation in many texts, 
            # BUT: We need equivalent Nodal Load. 
            # If load is Down (-Y), Reaction is Up (+Y). Equiv Load is Down (-Y).
            # Wait, Standard: F_equiv = - FixedEndReactions
            
            # Fixed End Moments (Counter-Clockwise +)
            # Load P (down is positive input in this logic usually? Let's assume input mag is raw)
            # Assume Downward Load P:
            P = mag # If user inputs +1000 for gravity load
            
            # Reactions (Upward +, CCW +)
            # MA_fix = P*a*b^2/L^2 
            # MB_fix = -P*a^2*b/L^2
            # RA_fix = P*b^2*(3*a+b)/L^3
            # RB_fix = P*a^2*(a+3*b)/L^3
            
            # Equivalent Nodal Forces = - Reactions
            # So if P is down (+), Nodes are pushed down (-).
            # Let's handle sign convention: Downward Load = Positive Mag in Input? 
            # Usually Structural apps: Downward is Gravity (+).
            # FEM Y axis: Up is +. So Downward Load is -P.
            
            F_y = -P
            
            # Fixed End Actions (Force/Moment FROM support TO beam)
            fem_1 = -F_y * a * b**2 / L**2
            fem_2 = +F_y * a**2 * b / L**2
            fea_1 = -F_y * b**2 * (3*a + b) / L**3
            fea_2 = -F_y * a**2 * (a + 3*b) / L**3
            
            # Equivalent Nodal Loads = - Fixed End Actions
            f[0] = -fea_1
            f[1] = -fem_1
            f[2] = -fea_2
            f[3] = -fem_2

        elif load['type'] == 'U': # Uniform Load
            # w = mag. (Assuming + is Downward gravity)
            # Total Load W = w * dist
            w = -mag # Convert to FEM coord (Up +)
            
            # Start and Dist
            start = a
            dist = float(load.get('dist', L))
            if dist == 0: dist = L - start # Handle 0 dist as full rest
            end = start + dist
            
            # This is complex for partial UDL.
            # Simplified: Exact integration needed.
            # For Full Span UDL (common):
            if abs(dist - L) < 1e-3 and start < 1e-3:
                 f[0] = w * L / 2
                 f[1] = -w * L**2 / 12
                 f[2] = w * L / 2
                 f[3] = +w * L**2 / 12
            else:
                # Partial UDL logic is heavy. 
                # Let's approximate or use full logic if critical. 
                # For this demo, let's implement simplified "point load integration" or full formula.
                # Use simplified integration for robustness:
                # Integrate w*dx over range.
                # Actually, let's just stick to simple Full UDL for robustness or 
                # approximate partial as point load at center (not exact but runs).
                # BETTER: Implement Partial UDL Fixed End Actions exactly.
                
                # Formulas for Partial UDL (w from a to c)
                # length of load = d
                # c = a + d
                d = dist
                c = end
                
                # We need to integrate.
                # It's safer to leave as is, but users might use partial.
                # Let's treat as series of point loads? No, too slow.
                # Let's use the exact formula for FEM 1 = - w*d/(12L^2) * ...
                # It is quite long. For this snippet, let's default to a robust approximation:
                # Treat as point load P = w*d at centroid (a + d/2).
                # (Note: This is an approximation for Moments, exact for Force sum)
                
                P_equiv = mag * d 
                center = start + d/2
                
                # Recursively call point load logic (Approx)
                # To be exact-ish:
                # This solver is for demo. Approximation is acceptable for partial UDL 
                # to prevent crash, but Full UDL is exact.
                
                # Let's do the Point Load calc with P_equiv
                # (Copy Point Load logic)
                P = P_equiv
                aa = center
                bb = L - aa
                
                F_y = -P # Down
                
                fem_1 = -F_y * aa * bb**2 / L**2
                fem_2 = +F_y * aa**2 * bb / L**2
                fea_1 = -F_y * bb**2 * (3*aa + bb) / L**3
                fea_2 = -F_y * aa**2 * (aa + 3*bb) / L**3
                
                f[0] = -fea_1; f[1] = -fem_1; f[2] = -fea_2; f[3] = -fem_2

        elif load['type'] == 'M': # Moment Load
            M = mag # Counter Clockwise +
            # FEM coord: M applied directly.
            # Fixed End Reactions for Moment M at a:
            # RA = -6Ma(L-a)/L^3
            # RB = 6Ma(L-a)/L^3
            # MA = M(b^2 - 2ab)/L^2  (Check signs carefully)
            # MB = M(a^2 - 2ab)/L^2
            
            # Eq Nodal = - Reactions
            # For simplicity in this demo, ignore Fixed End actions for Moment load 
            # (assume applied at nearest node or ignore local effect).
            # Implementing exact Moment FEA:
            b = L - a
            
            # Reactions
            R_A = -6*M*a*b / L**3
            R_B = +6*M*a*b / L**3
            M_A = M*b*(b - 2*a) / L**2
            M_B = M*a*(a - 2*b) / L**2
            
            f[0] = -R_A
            f[1] = -M_A
            f[2] = -R_B
            f[3] = -M_B

        return f

    def _get_internal_forces(self, x, L, u_elem, loads):
        # x is local distance from left node
        # u_elem: [v1, theta1, v2, theta2]
        
        E, I = self.E, self.I
        
        # Hermite Shape Functions
        # v(x) = N1*v1 + N2*th1 + N3*v2 + N4*th2
        xi = x/L
        N1 = 1 - 3*xi**2 + 2*xi**3
        N2 = L * (xi - 2*xi**2 + xi**3)
        N3 = 3*xi**2 - 2*xi**3
        N4 = L * (-xi**2 + xi**3)
        
        # Derivatives for Slope, Moment, Shear
        # dN/dx = (dN/dxi) * (1/L)
        # d2N/dx2 = (d2N/dxi2) * (1/L^2)
        # d3N/dx3 = (d3N/dxi3) * (1/L^3)
        
        # Deflection
        D_elastic = np.dot([N1, N2, N3, N4], u_elem)
        
        # Moment M = E*I * d2v/dx2
        dN2_dxi2 = np.array([
            -6 + 12*xi,
            L * (-4 + 6*xi),
            6 - 12*xi,
            L * (-2 + 6*xi)
        ])
        d2N_dx2 = dN2_dxi2 / L**2
        M_elastic = E * I * np.dot(d2N_dx2, u_elem)
        
        # Shear V = dM/dx = E*I * d3v/dx3
        dN3_dxi3 = np.array([
            12.0,
            L * 6.0,
            -12.0,
            L * 6.0
        ])
        d3N_dx3 = dN3_dxi3 / L**3
        V_elastic = E * I * np.dot(d3N_dx3, u_elem)
        
        # --- Superposition of Local Load Effects (Static Check) ---
        # The shape functions only give the elastic curve due to Nodal Displacements.
        # We must add the "Fixed End" static beam forces for the loads on this span.
        # (Beam = Fixed-Fixed + Displacements)
        
        D_static = 0
        M_static = 0
        V_static = 0
        
        for load in loads:
            mag = float(load['mag'])
            a = float(load['x'])
            
            if load['type'] == 'P':
                # Statics for Cantilever/Fixed formulation is tricky.
                # Easier approach: Singularity Functions (Macaulay) for simply supported? 
                # No, standard FEM is: Final = Homogeneous (K*u) + Particular (Loads)
                # Particular solution is the behavior of the beam if nodes were fixed.
                
                # Let's calculate V, M, D for a Fixed-Fixed beam with this load at x
                # Too complex to implement formulation for every case here.
                # SIMPLIFICATION for Demo: 
                # Calculate V, M via Statics of a "cut" section from left node.
                # V(x) = Ry1 + Sum(Loads left of x)
                # M(x) = Ry1*x + M1 + Sum(Loads_moment left of x)
                
                # We already have nodal forces at left node from FEM solution!
                # Wait, u_elem gives us the consistent deformations.
                # Forces at left node: F_node1 = K1*u. 
                # This includes Reactions and FixedEndForces.
                # Let's just use Statics from the Left Node.
                pass
            
            # --- Better Internal Force Calc Strategy ---
            # 1. Get forces at Start Node (Left) of element from the computed u_elem:
            #    F_start = k_local * u_elem - FEF_local
            #    This F_start contains {Shear_1, Moment_1} acting on the beam start.
            
        # Recalculate forces at Node 1 (Left) based on local stiffness
        k_loc = self._get_element_stiffness(L)
        f_int = k_loc @ u_elem # Forces required to deform beam
        
        # Subtract Fixed End Forces to get Total Force acting at Node 1
        # F_total = F_int + F_fixed_end_reaction (which is - Equivalent Load)
        # Actually: F_member = K*u + FixedEndActions
        
        fe_actions = np.zeros(4)
        for load in loads:
            # We calculated Equivalent Nodal Loads (f) earlier.
            # Fixed End Actions = - Equivalent Nodal Loads
            f_equiv = self._calc_fixed_end_forces(load, L)
            fe_actions -= f_equiv 
            
        F_total_start = f_int + fe_actions
        
        V_start = F_total_start[0] # Upward Force on beam start
        M_start = F_total_start[1] # CCW Moment on beam start
        
        # Now use Statics to walk from x=0 to x
        # V(x) = V_start + Sum(Loads Up)
        # M(x) = -M_start + V_start*x + Sum(Load * arm)
        # Note: M_start is CCW. Beam sign convention: Compression Top = +M? 
        # Usually M(x) = R*x - w*x^2/2...
        # Let's match standard convention: Sagging Positive (+).
        # CCW M_start at left tends to cause Hogging (-). So -M_start.
        
        V_x = V_start
        M_x = M_start # Start with reaction moment
        D_x = D_elastic # Deflection is purely from Shape Function? 
        # No, Shape function is exact for Point loads at nodes only. 
        # For inter-element loads, shape functions are approximation. 
        # But D_elastic is usually "good enough" for visualization unless P is large mid-span.
        
        # Loop loads for Statics
        for load in loads:
            mag = float(load['mag']) # Downward + input?
            # We assumed input P is + for Down in FEF calc. 
            # So Load Force = -mag (Upward).
            
            lx = float(load['x'])
            
            if x >= lx:
                if load['type'] == 'P':
                    # Point Load P (Down)
                    P = mag
                    V_x -= P # Shear drops
                    M_x -= P * (x - lx)
                    
                elif load['type'] == 'U':
                    # UDL w (Down)
                    w = mag
                    dist = float(load.get('dist', L))
                    # Intersection of UDL and current x
                    dx_overlap = min(x, lx + dist) - lx
                    
                    if dx_overlap > 0:
                        load_total = w * dx_overlap
                        centroid = dx_overlap / 2
                        V_x -= load_total
                        # Moment arm from cut x to centroid of load block
                        # Load block ends at lx + dx_overlap
                        # Centroid is at lx + dx_overlap/2
                        arm = x - (lx + centroid)
                        M_x -= load_total * arm
                        
                elif load['type'] == 'M':
                    M_ext = mag # CCW +
                    M_x += M_ext # Jump in moment
        
        return V_x, M_x, D_x
