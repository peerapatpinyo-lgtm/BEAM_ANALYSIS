import numpy as np
import pandas as pd

class BeamSolver:
    def __init__(self, spans, supports, loads, E, b, h, I):
        self.spans = spans
        self.supports = supports
        self.loads = loads # User defined loads
        self.E = E
        self.b = b
        self.h = h
        self.I = I
        
        # Material Constants
        self.DENSITY_CONCRETE = 24.0 # kN/m3 (Unit Weight)
        
        # Timoshenko Parameters
        self.nu = 0.2 
        self.G = self.E / (2 * (1 + self.nu))
        self.kappa = 5/6 
        self.A = self.b * self.h

        self.nodes_x = [0] + list(np.cumsum(spans))
        self.n_nodes = len(self.nodes_x)
        self.total_length = self.nodes_x[-1]

    def _get_timoshenko_stiffness(self, L):
        E, I, G, A, kappa = self.E, self.I, self.G, self.A, self.kappa
        phi = (12 * E * I) / (kappa * G * A * L**2)
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
            
            # --- 0. PREPARE TOTAL LOADS (USER + SELF WEIGHT) ---
            # Calculate Self Weight (kN/m)
            w_self = self.b * self.h * self.DENSITY_CONCRETE
            
            # Combine User Loads with Self Weight
            # We treat Self Weight as a UDL on EVERY span
            all_loads = self.loads.copy()
            for i, L in enumerate(self.spans):
                all_loads.append({
                    'type': 'U',
                    'span_index': i,
                    'x': 0.0,
                    'dist': L,
                    'mag': w_self,
                    'is_self_weight': True # Tag for visualization if needed
                })

            # --- 1. Stiffness Matrix ---
            for i, L in enumerate(self.spans):
                k_local = self._get_timoshenko_stiffness(L)
                idxs = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
                for r in range(4):
                    for c in range(4):
                        K_global[idxs[r], idxs[c]] += k_local[r, c]

            # --- 2. Loads (Fixed End Forces) ---
            for load in all_loads:
                span_idx = load.get('span_index')
                if span_idx is None or span_idx >= len(self.spans): continue
                
                L = self.spans[span_idx]
                mag = load['mag']
                fem = np.zeros(4) 
                
                if load['type'] == 'P':
                    a = load['x']
                    b_dist = L - a
                    fem[1] = -mag * (a * b_dist**2) / L**2
                    fem[3] = mag * (a**2 * b_dist) / L**2
                    fem[0] = -mag * (b_dist**2 * (3*a + b_dist)) / L**3
                    fem[2] = -mag * (a**2 * (a + 3*b_dist)) / L**3

                elif load['type'] == 'U':
                    x1 = load['x']
                    x2 = load['x'] + load['dist']
                    x1 = max(0, x1); x2 = min(L, x2)
                    
                    if x2 > x1:
                        w = mag
                        # Integration for partial UDL FEM
                        def int_term1(x): return (L**2 * x**2)/2 - (2*L * x**3)/3 + (x**4)/4
                        def int_term2(x): return (L * x**3)/3 - (x**4)/4
                        
                        val1 = int_term1(x2) - int_term1(x1)
                        val2 = int_term2(x2) - int_term2(x1)
                        
                        fem[1] = -(w / L**2) * val1
                        fem[3] = +(w / L**2) * val2
                        
                        total_load = w * (x2 - x1)
                        load_centroid = (x1 + x2) / 2
                        fem[0] = -(total_load * (L - load_centroid) + fem[1] + fem[3]) / L
                        fem[2] = -(total_load - (-fem[0]))
                        
                idxs = [2*span_idx, 2*span_idx+1, 2*(span_idx+1), 2*(span_idx+1)+1]
                for j in range(4): F_global[idxs[j]] += fem[j]

            # --- 3. Boundary Conditions ---
            fixed_dofs = []
            for s in self.supports:
                nid = s.get('id', s.get('node_id'))
                if s['type'] in ['Pin', 'Roller']: fixed_dofs.append(2*nid)
                elif s['type'] == 'Fixed': fixed_dofs.extend([2*nid, 2*nid+1])

            free_dofs = [i for i in range(n_dof) if i not in fixed_dofs]
            d_global = np.zeros(n_dof)
            
            if free_dofs:
                K_free = K_global[np.ix_(free_dofs, free_dofs)]
                F_free = F_global[free_dofs]
                try:
                    d_free = np.linalg.solve(K_free, F_free)
                    d_global[free_dofs] = d_free
                except np.linalg.LinAlgError:
                    return None, None, {"error": "Structure Unstable (Singular Matrix)"}
            
            # --- 4. Reactions ---
            # R = K*d - F_equiv
            R_global = np.dot(K_global, d_global) - F_global
            reactions = {i: R_global[2*i] for i in range(self.n_nodes)}

            # --- 5. Results Generation ---
            x_plot, v_plot, m_plot, d_plot = [], [], [], []
            
            for span_i, L_span in enumerate(self.spans):
                x_start_node = self.nodes_x[span_i]
                u_ele = d_global[[2*span_i, 2*span_i+1, 2*(span_i+1), 2*(span_i+1)+1]]
                
                for x_local in np.linspace(0, L_span, 51):
                    x_global = x_start_node + x_local
                    V_x, M_x = 0.0, 0.0
                    
                    # 5.1 Reactions from Left
                    for node_i in range(span_i + 1):
                        if node_i in reactions:
                            r_pos = self.nodes_x[node_i]
                            if r_pos <= x_global + 1e-5:
                                V_x += reactions[node_i]
                                M_x += reactions[node_i] * (x_global - r_pos)
                    
                    # 5.2 ALL Loads (User + Self Weight) from Left
                    for load in all_loads:
                        l_span_idx = load['span_index']
                        l_start = self.nodes_x[l_span_idx]
                        if l_start > x_global: continue
                        
                        if load['type'] == 'P':
                            p_loc = l_start + load['x']
                            if p_loc <= x_global + 1e-5:
                                V_x -= load['mag']
                                M_x -= load['mag'] * (x_global - p_loc)
                        elif load['type'] == 'U':
                            u_start = l_start + load['x']
                            u_end = u_start + load['dist']
                            eff_start = u_start
                            eff_end = min(x_global, u_end)
                            if eff_end > eff_start + 1e-5:
                                force = load['mag'] * (eff_end - eff_start)
                                cent = eff_start + (eff_end - eff_start)/2
                                V_x -= force
                                M_x -= force * (x_global - cent)
                    
                    # Deflection (mm)
                    xi = x_local / L_span
                    N1 = 1 - 3*xi**2 + 2*xi**3
                    N2 = x_local * (1 - 2*xi + xi**2)
                    N3 = 3*xi**2 - 2*xi**3
                    N4 = x_local * (xi**2 - xi)
                    def_val = (N1*u_ele[0] + N2*u_ele[1] + N3*u_ele[2] + N4*u_ele[3]) * 1000 

                    x_plot.append(x_global)
                    v_plot.append(V_x)
                    m_plot.append(M_x)
                    d_plot.append(def_val)

            df_res = pd.DataFrame({'x': x_plot, 'shear': v_plot, 'moment': m_plot, 'deflection': d_plot})
            
            # Save final w_self to access from outside if needed
            self.w_self_used = w_self
            self.all_loads_used = all_loads # For equilibrium check
            
            return df_res, reactions, {"status": "success"}

        except Exception as e:
            return None, None, {"error": str(e)}

    def check_equilibrium(self, reactions):
        """
        Updated to include Self-Weight in equilibrium check
        """
        sum_fy_load = 0
        sum_m_load = 0 
        
        # Use self.all_loads_used which includes SW
        for load in getattr(self, 'all_loads_used', self.loads):
            span_idx = load['span_index']
            base_x = self.nodes_x[span_idx]
            
            if load['type'] == 'P':
                f = load['mag']
                x = base_x + load['x']
                sum_fy_load += f
                sum_m_load += f * x
            elif load['type'] == 'U':
                w = load['mag']
                x_start = base_x + load['x']
                length = load['dist']
                f = w * length
                x_cent = x_start + length/2
                sum_fy_load += f
                sum_m_load += f * x_cent
                
        sum_fy_reac = sum(reactions.values())
        diff_fy = sum_fy_reac - sum_fy_load
        
        return {
            "load_down": sum_fy_load,
            "react_up": sum_fy_reac,
            "diff_fy": diff_fy
        }
