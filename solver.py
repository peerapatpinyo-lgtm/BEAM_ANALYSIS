# solver.py
import numpy as np
import pandas as pd

class BeamSolver:
    def __init__(self, spans, supports, loads, E, b, h, I):
        """
        spans: list of lengths [L1, L2, ...]
        supports: list of dicts [{'id': 0, 'type': 'Pin'}, ...]
        loads: list of dicts [{'type': 'P', 'span_index': 0, 'x': 2.5, 'mag': 1000}, ...]
        """
        self.spans = spans
        self.supports = supports
        self.loads = loads
        self.E = E
        self.b = b
        self.h = h
        self.I = I
        
        # Timoshenko Parameters (Assume Concrete properties for shear deformation)
        self.nu = 0.2 
        self.G = self.E / (2 * (1 + self.nu))
        self.kappa = 5/6  # Shear correction factor for rectangular section
        self.A = self.b * self.h

        self.nodes_x = [0] + list(np.cumsum(spans))
        self.n_nodes = len(self.nodes_x)
        self.total_length = self.nodes_x[-1]

    def _get_timoshenko_stiffness(self, L):
        """Generates local stiffness matrix for a beam element including shear deformation."""
        E, I, G, A, kappa = self.E, self.I, self.G, self.A, self.kappa
        phi = (12 * E * I) / (kappa * G * A * L**2) # Shear deformation parameter
        const = (E * I) / ((1 + phi) * L**3)
        
        k = np.zeros((4, 4))
        k[0, 0] = 12;        k[0, 1] = 6 * L;           k[0, 2] = -12;         k[0, 3] = 6 * L
        k[1, 0] = 6 * L;     k[1, 1] = (4 + phi)*L**2;  k[1, 2] = -6 * L;      k[1, 3] = (2 - phi)*L**2
        k[2, 0] = -12;       k[2, 1] = -6 * L;          k[2, 2] = 12;          k[2, 3] = -6 * L
        k[3, 0] = 6 * L;     k[3, 1] = (2 - phi)*L**2;  k[3, 2] = -6 * L;      k[3, 3] = (4 + phi)*L**2
        
        return k * const

    def solve(self):
        try:
            n_dof = self.n_nodes * 2
            K_global = np.zeros((n_dof, n_dof))
            F_global = np.zeros(n_dof)
            
            # --- 1. Stiffness Matrix Assembly ---
            for i, L in enumerate(self.spans):
                k_local = self._get_timoshenko_stiffness(L)
                idxs = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
                for r in range(4):
                    for c in range(4):
                        K_global[idxs[r], idxs[c]] += k_local[r, c]

            # --- 2. Loads (Fixed End Forces Calculation) ---
            for load in self.loads:
                span_idx = load.get('span_index')
                if span_idx is None or span_idx >= len(self.spans): continue
                
                L = self.spans[span_idx]
                mag = load['mag']
                fem = np.zeros(4) # [Fy1, M1, Fy2, M2]
                
                if load['type'] == 'P':
                    # [FIXED] Point Load Logic
                    # a = distance from left node, b = distance from right node
                    a = load['x'] 
                    b_dist = L - a
                    
                    # Formulas for Fixed End Moments (FEM)
                    fem[1] = -mag * (a * b_dist**2) / L**2  # M_left
                    fem[3] = mag * (a**2 * b_dist) / L**2   # M_right
                    
                    # Forces from static equilibrium of the fixed beam
                    fem[0] = -mag * (b_dist**2 * (3*a + b_dist)) / L**3 # Fy_left
                    fem[2] = -mag * (a**2 * (a + 3*b_dist)) / L**3      # Fy_right

                elif load['type'] == 'U':
                    # [UPGRADED] Partial UDL Support
                    # Integrate FEM formulas for load starting at x1 and ending at x2
                    x1 = load['x']
                    x2 = load['x'] + load['dist']
                    
                    # Ensure within bounds
                    x1 = max(0, x1)
                    x2 = min(L, x2)
                    
                    if x2 > x1:
                        w = mag
                        # Helper integrals for FEM: 
                        # M_left = -integral(w * x * (L-x)^2 / L^2 dx)
                        # M_right = integral(w * x^2 * (L-x) / L^2 dx)
                        
                        # Term 1: Integral of x(L-x)^2 = x(L^2 - 2Lx + x^2) = L^2x - 2Lx^2 + x^3
                        # Int -> L^2*x^2/2 - 2L*x^3/3 + x^4/4
                        def int_term1(x): return (L**2 * x**2)/2 - (2*L * x**3)/3 + (x**4)/4
                        
                        # Term 2: Integral of x^2(L-x) = L*x^2 - x^3
                        # Int -> L*x^3/3 - x^4/4
                        def int_term2(x): return (L * x**3)/3 - (x**4)/4
                        
                        val1 = int_term1(x2) - int_term1(x1)
                        val2 = int_term2(x2) - int_term2(x1)
                        
                        fem[1] = -(w / L**2) * val1  # M_left
                        fem[3] = +(w / L**2) * val2  # M_right
                        
                        # Calculate Vertical Forces based on Moments + Static Load
                        total_load = w * (x2 - x1)
                        load_centroid = (x1 + x2) / 2
                        
                        # Take moment about Right support to find R_left
                        # R_left * L + M_left + M_right - Total_Load * (L - centroid) = 0
                        # R_left = (Total_Load * (L - centroid) - M_left - M_right) / L
                        
                        # Note: FEM moments are reaction moments on the nodes.
                        # Equation signs: Sum M_right = 0 => R_left*L + M_left_react + M_right_react - Force*(L-cent) = 0
                        # Here fem[1] and fem[3] are vector forces/moments acting ON THE NODE.
                        
                        fem[0] = -(total_load * (L - load_centroid) + fem[1] + fem[3]) / L
                        fem[2] = -(total_load - (-fem[0])) # Sum Fy = 0
                        
                # Add to Global Force Vector
                idxs = [2*span_idx, 2*span_idx+1, 2*(span_idx+1), 2*(span_idx+1)+1]
                for j in range(4): F_global[idxs[j]] += fem[j]

            # --- 3. Apply Boundary Conditions ---
            fixed_dofs = []
            for s in self.supports:
                nid = s.get('id', s.get('node_id'))
                if s['type'] in ['Pin', 'Roller']: fixed_dofs.append(2*nid) # Fix Y
                elif s['type'] == 'Fixed': fixed_dofs.extend([2*nid, 2*nid+1]) # Fix Y and Rotation

            free_dofs = [i for i in range(n_dof) if i not in fixed_dofs]
            d_global = np.zeros(n_dof)
            
            if free_dofs:
                K_free = K_global[np.ix_(free_dofs, free_dofs)]
                F_free = F_global[free_dofs]
                # Solve Kd = F
                try:
                    d_free = np.linalg.solve(K_free, F_free)
                    d_global[free_dofs] = d_free
                except np.linalg.LinAlgError:
                    return None, None, {"error": "Unstable Structure (Matrix Singular)"}
            
            # --- 4. Compute Reactions ---
            # R = K * d - F_applied (F_global contains equivalent nodal forces from loads)
            # Reaction is the force needed to maintain the displacement (usually 0 at supports)
            # The 'F_global' used here is the Equivalent Nodal Loads.
            # Actual Equation: K*d = F_external + R
            # So R = K*d - F_external
            # Since F_global was constructed as "Forces applied TO nodes", F_external = F_global
            R_global = np.dot(K_global, d_global) - F_global
            reactions = {i: R_global[2*i] for i in range(self.n_nodes)}

            # --- 5. Post-Processing (Method of Sections for Diagrams) ---
            x_plot, v_plot, m_plot, d_plot = [], [], [], []
            
            for span_i, L_span in enumerate(self.spans):
                x_start_node = self.nodes_x[span_i]
                
                # Get nodal displacements for this element (for shape function)
                u_ele = d_global[[2*span_i, 2*span_i+1, 2*(span_i+1), 2*(span_i+1)+1]]
                
                # Create evaluation points
                num_pts = 51
                x_evals = np.linspace(0, L_span, num_pts)
                
                for x_local in x_evals:
                    x_global = x_start_node + x_local
                    
                    # --- Internal Forces (V, M) using Statics (Left-hand section) ---
                    V_x, M_x = 0.0, 0.0
                    
                    # 5.1 Sum Reactions from left
                    for node_i in range(span_i + 1):
                        if node_i in reactions:
                            r_pos = self.nodes_x[node_i]
                            # Include if reaction is to the left (or at current point)
                            if r_pos <= x_global + 1e-5:
                                V_x += reactions[node_i]
                                M_x += reactions[node_i] * (x_global - r_pos)
                    
                    # 5.2 Sum Loads from left
                    for load in self.loads:
                        l_span_idx = load['span_index']
                        l_span_start_x = self.nodes_x[l_span_idx]
                        
                        if l_span_start_x > x_global: continue # Load starts after current section
                        
                        if load['type'] == 'P':
                            p_loc_global = l_span_start_x + load['x']
                            if p_loc_global <= x_global + 1e-5:
                                V_x -= load['mag']
                                M_x -= load['mag'] * (x_global - p_loc_global)
                                
                        elif load['type'] == 'U':
                            # Global start/end of this UDL
                            u_start_global = l_span_start_x + load['x']
                            u_end_global = u_start_global + load['dist']
                            
                            # Determine overlap with current section [0, x_global]
                            # Load acts from u_start_global to u_end_global
                            # We only care about the portion <= x_global
                            
                            eff_start = u_start_global
                            eff_end = min(x_global, u_end_global)
                            
                            if eff_end > eff_start + 1e-5:
                                w_len = eff_end - eff_start
                                force = load['mag'] * w_len
                                centroid = eff_start + w_len/2
                                
                                V_x -= force
                                M_x -= force * (x_global - centroid)
                    
                    # --- Deflection (Hermitian Interpolation) ---
                    xi = x_local / L_span
                    # Shape functions
                    N1 = 1 - 3*xi**2 + 2*xi**3
                    N2 = x_local * (1 - 2*xi + xi**2)
                    N3 = 3*xi**2 - 2*xi**3
                    N4 = x_local * (xi**2 - xi)
                    
                    def_val = (N1*u_ele[0] + N2*u_ele[1] + N3*u_ele[2] + N4*u_ele[3]) * 1000 # convert to mm

                    x_plot.append(x_global)
                    v_plot.append(V_x)
                    m_plot.append(M_x)
                    d_plot.append(def_val)

            df_res = pd.DataFrame({'x': x_plot, 'shear': v_plot, 'moment': m_plot, 'deflection': d_plot})
            return df_res, reactions, {"status": "success"}

        except Exception as e:
            return None, None, {"error": str(e)}
