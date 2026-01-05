import numpy as np
import pandas as pd
from scipy.linalg import solve

class BeamSolver:
    def __init__(self, spans, supports_df, loads_df, E, I, A=None, G=None):
        self.spans = spans
        self.supports_df = supports_df
        self.loads_df = loads_df
        self.E = E
        self.I = I
        self.A = A if A is not None else 100.0
        self.G = G if G is not None else E / (2*(1+0.3))
        self.use_timoshenko = (A is not None)

    def solve(self):
        # 1. Discretize
        nodes, elements = self._discretize_model()
        num_nodes = len(nodes)
        dof = 2 * num_nodes
        
        K = np.zeros((dof, dof))
        F = np.zeros(dof)
        
        # 2. Assemble Stiffness
        for elem in elements:
            node_i = elem['n1']
            node_j = elem['n2']
            x1 = nodes[node_i]
            x2 = nodes[node_j]
            L = x2 - x1
            
            # ส่งค่า L ที่เป็นตัวเลขเข้าไปคำนวณ
            k_local = self._get_element_stiffness(L)
            
            idx = [2*node_i, 2*node_i+1, 2*node_j, 2*node_j+1]
            for r in range(4):
                for c in range(4):
                    K[idx[r], idx[c]] += k_local[r, c]
                    
        # 3. Assemble Forces
        for _, load in self.loads_df.iterrows():
            # Point/Moment Loads
            node_idx = -1
            for i, x in enumerate(nodes):
                if np.isclose(x, load['x'], atol=1e-5):
                    node_idx = i
                    break
            
            if node_idx != -1:
                if load['type'] == 'P':
                    F[2 * node_idx] -= load['mag'] 
                elif load['type'] == 'M':
                    F[2 * node_idx + 1] -= load['mag'] 

        # Distributed Loads
        for _, load in self.loads_df.iterrows():
            if load['type'] == 'U':
                start = load['x']
                end = start + load['dist']
                mag = load['mag']
                
                for elem in elements:
                    x1, x2 = nodes[elem['n1']], nodes[elem['n2']]
                    L_elem = x2 - x1
                    
                    overlap_start = max(start, x1)
                    overlap_end = min(end, x2)
                    
                    if overlap_end > overlap_start:
                        a = overlap_start - x1
                        b = overlap_end - x1
                        w = -mag
                        
                        load_len = b - a
                        mid = (a + b) / 2
                        gauss_pts = [-0.57735, 0.57735]
                        gauss_w = [1.0, 1.0]
                        
                        fe = np.zeros(4)
                        for gp, gw in zip(gauss_pts, gauss_w):
                            x_in_elem = mid + (load_len/2)*gp
                            s = (x_in_elem - x1) / L_elem
                            
                            n_vec = np.array([
                                1 - 3*s**2 + 2*s**3,
                                (x_in_elem - x1) * (1 - s)**2,
                                3*s**2 - 2*s**3,
                                (x_in_elem - x1) * (s**2 - s)
                            ])
                            fe += n_vec * w * gw * (load_len / 2)

                        idx = [2*elem['n1'], 2*elem['n1']+1, 2*elem['n2'], 2*elem['n2']+1]
                        F[idx] += fe

        # 4. Boundary Conditions
        free_dof = np.full(dof, True)
        for _, sup in self.supports_df.iterrows():
            node_i = sup['id']
            stype = sup['type']
            if stype in ['Pin', 'Roller', 'Fixed']:
                free_dof[2*node_i] = False
            if stype == 'Fixed':
                free_dof[2*node_i+1] = False
        
        # 5. Solve
        U = np.zeros(dof)
        if np.any(free_dof):
            try:
                K_reduced = K[np.ix_(free_dof, free_dof)]
                F_reduced = F[free_dof]
                U_reduced = solve(K_reduced, F_reduced)
                U[free_dof] = U_reduced
            except:
                return pd.DataFrame(), [], None
        
        R = K @ U - F
        
        # 6. Post-Process (Generate Results Table)
        results = []
        for elem in elements:
            node_i, node_j = elem['n1'], elem['n2']
            x1, x2 = nodes[node_i], nodes[node_j]
            L = x2 - x1
            
            u_ele = U[[2*node_i, 2*node_i+1, 2*node_j, 2*node_j+1]]
            
            x_vals = np.linspace(0, L, 50)
            for x_local in x_vals:
                s = x_local / L
                N = np.array([1 - 3*s**2 + 2*s**3, x_local * (1 - s)**2, 3*s**2 - 2*s**3, x_local * (s**2 - s)])
                
                # Derivatives for Moment/Shear
                N_d2 = np.array([-6/L**2 + 12*x_local/L**3, -4/L + 6*x_local/L**2, 6/L**2 - 12*x_local/L**3, -2/L + 6*x_local/L**2])
                N_d3 = np.array([12/L**3, 6/L**2, -12/L**3, 6/L**2])
                
                y = np.dot(N, u_ele)
                m_val = self.E * self.I * np.dot(N_d2, u_ele)
                v_val = self.E * self.I * np.dot(N_d3, u_ele)
                
                # Add Particular Solution for UDL
                for _, load in self.loads_df.iterrows():
                    if load['type'] == 'U':
                        l_start = max(load['x'], x1)
                        l_end = min(load['x'] + load['dist'], x2)
                        if l_end > l_start:
                            a_local = l_start - x1
                            b_local = l_end - x1
                            w = -load['mag']
                            if x_local > a_local:
                                cv_len = min(x_local, b_local) - a_local
                                v_val -= w * cv_len
                                m_val -= w * cv_len * (x_local - (a_local + cv_len/2))
                
                # --- ใช้ชื่อคอลัมน์ตัวใหญ่ (Capitalized) ---
                results.append({'x': x1 + x_local, 'Deflection': y, 'Moment': m_val, 'Shear': v_val})
                
        df_res = pd.DataFrame(results)
        summary = {}
        if not df_res.empty:
            summary['V_max'] = {'value': df_res['Shear'].abs().max(), 'x': df_res.loc[df_res['Shear'].abs().idxmax(), 'x']}
            summary['M_pos'] = {'value': df_res['Moment'].max(), 'x': df_res.loc[df_res['Moment'].idxmax(), 'x']}
            summary['M_neg'] = {'value': df_res['Moment'].min(), 'x': df_res.loc[df_res['Moment'].idxmin(), 'x']}
            summary['D_max'] = {'value': df_res['Deflection'].abs().max(), 'x': df_res.loc[df_res['Deflection'].abs().idxmax(), 'x']}
            
        return df_res, R, summary

    def _discretize_model(self):
        x_points = {0.0}
        current_x = 0.0
        for s in self.spans:
            current_x += s
            x_points.add(round(current_x, 5))
        for _, load in self.loads_df.iterrows():
            x_points.add(round(load['x'], 5))
            if load['type'] == 'U':
                x_points.add(round(load['x'] + load['dist'], 5))
        sorted_x = sorted(list(x_points))
        elements = [{'n1': i, 'n2': i+1} for i in range(len(sorted_x)-1)]
        return sorted_x, elements

    def _get_element_stiffness(self, L):
        E, I = self.E, self.I
        k = np.zeros((4,4))
        factor = E * I / (L**3)
        k[0,0] = 12;  k[0,1] = 6*L;    k[0,2] = -12;  k[0,3] = 6*L
        k[1,0] = 6*L; k[1,1] = 4*L**2; k[1,2] = -6*L; k[1,3] = 2*L**2
        k[2,0] = -12; k[2,1] = -6*L;   k[2,2] = 12;   k[2,3] = -6*L
        k[3,0] = 6*L; k[3,1] = 2*L**2; k[3,2] = -6*L; k[3,3] = 4*L**2
        return k * factor
