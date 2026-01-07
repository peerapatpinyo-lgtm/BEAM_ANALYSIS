
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
        
        # Timoshenko Parameters (Assume Concrete)
        self.nu = 0.2 
        self.G = self.E / (2 * (1 + self.nu))
        self.kappa = 5/6  # Rectangular section
        self.A = self.b * self.h

        self.nodes_x = [0] + list(np.cumsum(spans))
        self.n_nodes = len(self.nodes_x)
        self.total_length = self.nodes_x[-1]

    def _get_timoshenko_stiffness(self, L):
        E, I, G, A, kappa = self.E, self.I, self.G, self.A, self.kappa
        phi = (12 * E * I) / (kappa * G * A * L**2) # Shear deformation parameter
        const = (E * I) / ((1 + phi) * L**3)
        
        k = np.zeros((4, 4))
        k[0, 0] = 12;        k[0, 1] = 6 * L;           k[0, 2] = -12;         k[0, 3] = 6 * L
        k[1, 0] = 6 * L;     k[1, 1] = (4 + phi)*L**2;  k[1, 2] = -6 * L;      k[1, 3] = (2 - phi)*L**2
        k[2, 0] = -12;       k[2, 1] = -6 * L;          k[2, 2] = 12;          k[2, 3] = -6 * L
        k[3, 0] = 6 * L;     k[3, 1] = (2 - phi)*L**2;  k[3, 2] = -6 * L;      k[3, 3] = (4 + phi)*L**2
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

            # 2. Loads (Fixed End Forces - Simplified for Timoshenko)
            for load in self.loads:
                span_idx = load.get('span_index')
                if span_idx is None: continue
                L = self.spans[span_idx]
                mag = load['mag']
                fem = np.zeros(4)
                
                if load['type'] == 'U':
                    fem = np.array([-mag*L/2, -mag*L**2/12, -mag*L/2, mag*L**2/12])
                elif load['type'] == 'P':
                    a = load['dist']; b_dist = L - a
                    fem[0] = -mag * (b_dist**2 * (3*a + b_dist)) / L**3
                    fem[1] = -mag * (a * b_dist**2) / L**2
                    fem[2] = -mag * (a**2 * (a + 3*b_dist)) / L**3
                    fem[3] = mag * (a**2 * b_dist) / L**2
                
                idxs = [2*span_idx, 2*span_idx+1, 2*(span_idx+1), 2*(span_idx+1)+1]
                for j in range(4): F_global[idxs[j]] += fem[j]

            # 3. Boundary Conditions
            fixed_dofs = []
            for s in self.supports:
                nid = s['id']
                if s['type'] in ['Pin', 'Roller']: fixed_dofs.append(2*nid)
                elif s['type'] == 'Fixed': fixed_dofs.extend([2*nid, 2*nid+1])

            free_dofs = [i for i in range(n_dof) if i not in fixed_dofs]
            d_global = np.zeros(n_dof)
            
            if free_dofs:
                K_free = K_global[np.ix_(free_dofs, free_dofs)]
                F_free = F_global[free_dofs]
                d_free = np.linalg.solve(K_free, F_free)
                d_global[free_dofs] = d_free
            
            # 4. Reactions
            R_global = np.dot(K_global, d_global) - F_global
            reactions = {i: R_global[2*i] for i in range(self.n_nodes)}

            # 5. Generate Smooth Results (Method of Sections)
            x_plot, v_plot, m_plot, d_plot = [], [], [], []
            
            for span_i, L_span in enumerate(self.spans):
                x_start = self.nodes_x[span_i]
                u_ele = d_global[[2*span_i, 2*span_i+1, 2*(span_i+1), 2*(span_i+1)+1]]
                
                for x_loc in np.linspace(0, L_span, 51):
                    x_global = x_start + x_loc
                    
                    # Internal Forces (Statics)
                    V_x, M_x = 0.0, 0.0
                    
                    # Sum Reactions
                    for node_i in range(span_i + 1):
                        if node_i in reactions:
                            r_pos = self.nodes_x[node_i]
                            if r_pos <= x_global + 1e-4:
                                V_x += reactions[node_i]
                                M_x += reactions[node_i] * (x_global - r_pos)
                    
                    # Sum Loads
                    for load in self.loads:
                        l_start = self.nodes_x[load['span_index']]
                        if l_start > x_global: continue
                        
                        if load['type'] == 'P':
                            p_loc = l_start + load['dist']
                            if p_loc <= x_global + 1e-4:
                                V_x -= load['mag']
                                M_x -= load['mag'] * (x_global - p_loc)
                        elif load['type'] == 'U':
                            start = l_start
                            end = start + load['dist']
                            eff_end = min(x_global, end)
                            if eff_end > start:
                                w_len = eff_end - start
                                force = load['mag'] * w_len
                                cent = start + w_len/2
                                V_x -= force
                                M_x -= force * (x_global - cent)
                    
                    # Deflection (Hermitian approx)
                    xi = x_loc / L_span
                    N1 = 1 - 3*xi**2 + 2*xi**3
                    N2 = x_loc * (1 - 2*xi + xi**2)
                    N3 = 3*xi**2 - 2*xi**3
                    N4 = x_loc * (xi**2 - xi)
                    def_val = (N1*u_ele[0] + N2*u_ele[1] + N3*u_ele[2] + N4*u_ele[3]) * 1000

                    x_plot.append(x_global)
                    v_plot.append(V_x)
                    m_plot.append(M_x)
                    d_plot.append(def_val)

            df_res = pd.DataFrame({'x': x_plot, 'shear': v_plot, 'moment': m_plot, 'deflection': d_plot})
            return df_res, reactions, {"status": "success"}

        except Exception as e:
            return None, None, {"error": str(e)}
