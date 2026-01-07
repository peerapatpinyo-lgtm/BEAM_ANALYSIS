import numpy as np
import pandas as pd

class BeamSolver:
    def __init__(self, spans, supports, loads, E, b, h, I):
        """
        Initialization
        :param spans: list of span lengths [L1, L2, ...]
        :param supports: list of dicts [{'id': 0, 'type': 'Pin'}, ...]
        :param loads: list of dicts [{'type': 'U', 'mag': 1000, ...}, ...]
        :param E: Elastic Modulus (MPa or N/mm^2 -> input as N/m^2 in main)
        :param b: width (m)
        :param h: depth (m)
        :param I: Moment of Inertia (m^4)
        """
        self.spans = spans
        self.supports = supports
        self.loads = loads
        self.E = E
        self.b = b
        self.h = h
        self.I = I
        
        # Calculate Poisson's Ratio & Shear Modulus (Assume Concrete)
        self.nu = 0.2 
        self.G = self.E / (2 * (1 + self.nu))
        self.kappa = 5/6  # Shear correction factor for rectangular section
        self.A = self.b * self.h # Cross-sectional Area

        # Node Mapping
        self.nodes_x = [0] + list(np.cumsum(spans))
        self.n_nodes = len(self.nodes_x)
        self.total_length = self.nodes_x[-1]

    def _get_timoshenko_stiffness(self, L):
        """
        Generate Stiffness Matrix using Timoshenko Beam Theory
        Includes Shear Deformation Parameter (Phi)
        """
        E, I, G, A, kappa = self.E, self.I, self.G, self.A, self.kappa
        
        # Phi (Shear Deformation Parameter)
        # If Phi = 0, it reduces to Euler-Bernoulli
        phi = (12 * E * I) / (kappa * G * A * L**2)
        
        # Common constant
        const = (E * I) / ((1 + phi) * L**3)
        
        # Matrix Coefficients
        k = np.zeros((4, 4))
        
        # Row 1
        k[0, 0] = 12
        k[0, 1] = 6 * L
        k[0, 2] = -12
        k[0, 3] = 6 * L
        
        # Row 2
        k[1, 0] = 6 * L
        k[1, 1] = (4 + phi) * L**2
        k[1, 2] = -6 * L
        k[1, 3] = (2 - phi) * L**2
        
        # Row 3
        k[2, 0] = -12
        k[2, 1] = -6 * L
        k[2, 2] = 12
        k[2, 3] = -6 * L
        
        # Row 4
        k[3, 0] = 6 * L
        k[3, 1] = (2 - phi) * L**2
        k[3, 2] = -6 * L
        k[3, 3] = (4 + phi) * L**2
        
        return k * const

    def solve(self):
        try:
            n_dof = self.n_nodes * 2
            K_global = np.zeros((n_dof, n_dof))
            F_global = np.zeros(n_dof)
            
            # 1. ASSEMBLE GLOBAL STIFFNESS (Timoshenko)
            for i, L in enumerate(self.spans):
                k_local = self._get_timoshenko_stiffness(L)
                
                # DOF Mapping: [y1, th1, y2, th2]
                idxs = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
                
                for r in range(4):
                    for c in range(4):
                        K_global[idxs[r], idxs[c]] += k_local[r, c]

            # 2. ASSEMBLE LOAD VECTOR (Fixed End Actions)
            # Note: Using consistent load vector for Timoshenko is complex.
            # For standard applications, standard FEA loads are acceptable approximations
            # as the stiffness matrix controls the primary redistribution.
            for load in self.loads:
                span_idx = load.get('span_index')
                if span_idx is None: continue
                
                L = self.spans[span_idx]
                mag = load['mag']
                
                fem = np.zeros(4)
                
                if load['type'] == 'U': # UDL
                    fem[0] = -mag * L / 2
                    fem[1] = -mag * L**2 / 12
                    fem[2] = -mag * L / 2
                    fem[3] = mag * L**2 / 12
                    
                elif load['type'] == 'P': # Point Load
                    a = load['dist']
                    b_dist = L - a
                    # P at distance a
                    fem[0] = -mag * (b_dist**2 * (3*a + b_dist)) / L**3
                    fem[1] = -mag * (a * b_dist**2) / L**2
                    fem[2] = -mag * (a**2 * (a + 3*b_dist)) / L**3
                    fem[3] = mag * (a**2 * b_dist) / L**2

                # Add to Global
                idxs = [2*span_idx, 2*span_idx+1, 2*(span_idx+1), 2*(span_idx+1)+1]
                for j in range(4):
                    F_global[idxs[j]] += fem[j]

            # 3. BOUNDARY CONDITIONS
            fixed_dofs = []
            for s in self.supports:
                nid = s['id']
                if s['type'] in ['Pin', 'Roller']:
                    fixed_dofs.append(2*nid) # Fix Y
                elif s['type'] == 'Fixed':
                    fixed_dofs.append(2*nid)   # Fix Y
                    fixed_dofs.append(2*nid+1) # Fix Rotation

            free_dofs = [i for i in range(n_dof) if i not in fixed_dofs]
            
            # Partition & Solve
            K_free = K_global[np.ix_(free_dofs, free_dofs)]
            F_free = F_global[free_dofs]
            
            d_free = np.linalg.solve(K_free, F_free)
            
            d_global = np.zeros(n_dof)
            d_global[free_dofs] = d_free
            
            # Calculate Reactions (R = K*d - F)
            R_global = np.dot(K_global, d_global) - F_global
            
            # Store Reactions nicely
            reactions = {}
            for i in range(self.n_nodes):
                reactions[i] = R_global[2*i] # Store vertical reaction only

            # 4. GENERATE HIGH-RES RESULTS (Method of Sections / Statics)
            # นี่คือส่วนที่ทำให้กราฟ "สวย" และ "ถูกต้อง" ที่สุด
            # โดยการเดินตัด Section ตลอดความยาวคาน (Statics Check)
            
            x_plot = []
            v_plot = []
            m_plot = []
            d_plot = [] # Deflection (Approx via Shape Function for visual)
            
            # Generate points (e.g., 100 points per span)
            total_points = 0
            
            for span_i, L_span in enumerate(self.spans):
                x_start = self.nodes_x[span_i]
                
                # Shape function coeffs for this span (for Deflection only)
                u_ele = d_global[[2*span_i, 2*span_i+1, 2*(span_i+1), 2*(span_i+1)+1]]
                
                # Create dense points
                x_local_vals = np.linspace(0, L_span, 51)
                
                for x_loc in x_local_vals:
                    x_global = x_start + x_loc
                    
                    # --- A. Calculate Shear & Moment using STATIC EQUILIBRIUM (Method of Sections) ---
                    # Cut at x_global, sum forces from Left
                    
                    V_x = 0
                    M_x = 0
                    
                    # 1. Sum Reactions to the left
                    for node_i in range(span_i + 1): # Nodes up to current span start
                        if node_i * 0 <= x_global: # Safety check logic
                            if node_i in reactions:
                                r_val = reactions[node_i]
                                r_pos = self.nodes_x[node_i]
                                if r_pos <= x_global + 0.0001:
                                    V_x += r_val
                                    M_x += r_val * (x_global - r_pos)
                    
                    # 2. Sum Loads to the left
                    for load in self.loads:
                        l_span_idx = load['span_index']
                        l_x_start_global = self.nodes_x[l_span_idx]
                        
                        if l_x_start_global > x_global: continue # Load starts after cut
                        
                        dx = x_global - l_x_start_global # Distance from start of load's span
                        
                        if load['type'] == 'P':
                            p_pos_global = l_x_start_global + load['dist']
                            if p_pos_global <= x_global + 0.0001:
                                V_x -= load['mag']
                                M_x -= load['mag'] * (x_global - p_pos_global)
                                
                        elif load['type'] == 'U':
                            # Load starts at l_x_start_global
                            # Load ends at l_x_start_global + load['dist']
                            
                            start_load = l_x_start_global
                            end_load = start_load + load['dist']
                            
                            # Determine overlap with cut section
                            eff_start = start_load
                            eff_end = min(x_global, end_load)
                            
                            if eff_end > eff_start:
                                w_len = eff_end - eff_start
                                w_force = load['mag'] * w_len # Total load
                                centroid = eff_start + w_len/2
                                
                                V_x -= w_force
                                M_x -= w_force * (x_global - centroid)

                    # --- B. Deflection (Using Timoshenko Shape Functions or Approx) ---
                    # For plotting, standard Hermitian is often close enough visually, 
                    # but let's use the node displacements we found.
                    xi = x_loc / L_span
                    # Hermitian Shape Functions (Standard)
                    # Note: Using Exact Timoshenko shape functions for plotting is overkill 
                    # for visualization if nodes are correct. Using Hermitian for smooth curve.
                    N1 = 1 - 3*xi**2 + 2*xi**3
                    N2 = x_loc * (1 - 2*xi + xi**2)
                    N3 = 3*xi**2 - 2*xi**3
                    N4 = x_loc * (xi**2 - xi)
                    
                    def_val = (N1*u_ele[0] + N2*u_ele[1] + N3*u_ele[2] + N4*u_ele[3]) * 1000 # mm
                    
                    x_plot.append(x_global)
                    v_plot.append(V_x)
                    m_plot.append(M_x)
                    d_plot.append(def_val)

            # Create DataFrame
            df_res = pd.DataFrame({
                'x': x_plot,
                'shear': v_plot,
                'moment': m_plot,
                'deflection': d_plot
            })
            
            return df_res, reactions, {"status": "success"}

        except Exception as e:
            return None, None, {"error": str(e)}
