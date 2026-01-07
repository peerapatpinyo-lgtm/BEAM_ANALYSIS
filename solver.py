import numpy as np
import pandas as pd

class BeamSolver:
    def __init__(self, spans, supports, loads, E, b, h, I):
        self.spans = spans
        self.supports = supports
        self.loads = loads
        self.E = E
        self.b = b
        self.h = h
        self.I = I
        
        # Parameters
        self.DENSITY = 24.0 # kN/m3 (Concrete)
        self.nu = 0.2 
        self.G = self.E / (2 * (1 + self.nu))
        self.kappa = 5/6 
        self.A = self.b * self.h

        self.nodes_x = [0] + list(np.cumsum(spans))
        self.n_nodes = len(self.nodes_x)
        self.total_length = self.nodes_x[-1]
        
        # Prepare Loads (User + Self Weight)
        self.final_loads = self._prepare_loads()

    def _prepare_loads(self):
        """Combines user loads with calculated self-weight."""
        w_sw = self.b * self.h * self.DENSITY # kN/m
        
        combined_loads = []
        
        # 1. Add Self Weight as UDL for every span
        for i, L in enumerate(self.spans):
            combined_loads.append({
                'type': 'U',
                'span_index': i,
                'x': 0.0,
                'dist': L,
                'mag': w_sw,
                'source': 'self_weight'
            })
            
        # 2. Add User Loads
        for l in self.loads:
            # Copy to avoid modifying original
            new_l = l.copy()
            new_l['source'] = 'user'
            combined_loads.append(new_l)
            
        return combined_loads

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
            
            # 1. Stiffness Matrix
            for i, L in enumerate(self.spans):
                k_local = self._get_timoshenko_stiffness(L)
                idxs = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
                for r in range(4):
                    for c in range(4):
                        K_global[idxs[r], idxs[c]] += k_local[r, c]

            # 2. Process All Loads (FEM)
            for load in self.final_loads:
                span_idx = load.get('span_index')
                if span_idx is None: continue
                
                L = self.spans[span_idx]
                mag = load['mag']
                fem = np.zeros(4)
                
                if load['type'] == 'P':
                    a = load['x']
                    b_dist = L - a
                    # FEM Formulas
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
                        # Integrated FEM for Partial UDL
                        def int_term1(x): return (L**2 * x**2)/2 - (2*L * x**3)/3 + (x**4)/4
                        def int_term2(x): return (L * x**3)/3 - (x**4)/4
                        
                        val1 = int_term1(x2) - int_term1(x1)
                        val2 = int_term2(x2) - int_term2(x1)
                        
                        fem[1] = -(w / L**2) * val1
                        fem[3] = +(w / L**2) * val2
                        
                        total_load = w * (x2 - x1)
                        centroid = (x1 + x2) / 2
                        fem[0] = -(total_load * (L - centroid) + fem[1] + fem[3]) / L
                        fem[2] = -(total_load - (-fem[0]))

                idxs = [2*span_idx, 2*span_idx+1, 2*(span_idx+1), 2*(span_idx+1)+1]
                for j in range(4): F_global[idxs[j]] += fem[j]

            # 3. Boundary Conditions
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
                    return None, None, {"error": "Structure Unstable"}

            # 4. Reactions
            R_global = np.dot(K_global, d_global) - F_global
            reactions = {i: R_global[2*i] for i in range(self.n_nodes)}

            # 5. Method of Sections (Visualization Data)
            x_plot, v_plot, m_plot, d_plot = [], [], [], []
            
            for span_i, L_span in enumerate(self.spans):
                x_start_node = self.nodes_x[span_i]
                u_ele = d_global[[2*span_i, 2*span_i+1, 2*(span_i+1), 2*(span_i+1)+1]]
                
                # Evaluation points
                pts = np.linspace(0, L_span, 51)
                for x_local in pts:
                    x_global = x_start_node + x_local
                    V_x, M_x = 0.0, 0.0
                    
                    # 5.1 Reactions from Left
                    for node_i in range(span_i + 1):
                        if node_i in reactions:
                            r_pos = self.nodes_x[node_i]
                            if r_pos <= x_global + 1e-6:
                                V_x += reactions[node_i]
                                M_x += reactions[node_i] * (x_global - r_pos)
                                
                    # 5.2 Loads from Left (Includes SW)
                    for load in self.final_loads:
                        l_span_idx = load['span_index']
                        l_start_global = self.nodes_x[l_span_idx]
                        
                        if l_start_global > x_global: continue
                        
                        if load['type'] == 'P':
                            p_loc_global = l_start_global + load['x']
                            if p_loc_global <= x_global + 1e-6:
                                V_x -= load['mag']
                                M_x -= load['mag'] * (x_global - p_loc_global)
                                
                        elif load['type'] == 'U':
                            u_start_global = l_start_global + load['x']
                            u_end_global = u_start_global + load['dist']
                            
                            eff_start = u_start_global
                            eff_end = min(x_global, u_end_global)
                            
                            if eff_end > eff_start + 1e-6:
                                w_len = eff_end - eff_start
                                force = load['mag'] * w_len
                                cent = eff_start + w_len/2
                                V_x -= force
                                M_x -= force * (x_global - cent)
                    
                    # Deflection
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
            return df_res, reactions, {"status": "success"}

        except Exception as e:
            return None, None, {"error": str(e)}

    def check_equilibrium(self, reactions):
        sum_fy_load = 0
        
        for load in self.final_loads:
            if load['type'] == 'P':
                sum_fy_load += load['mag']
            elif load['type'] == 'U':
                sum_fy_load += load['mag'] * load['dist']
                
        sum_fy_reac = sum(reactions.values())
        diff = sum_fy_reac - sum_fy_load
        
        return {
            "load_down": sum_fy_load,
            "react_up": sum_fy_reac,
            "diff_fy": diff
        }
