import numpy as np
import pandas as pd
from scipy import integrate

class BeamSolver:
    def __init__(self, spans, supports, loads, E, I):
        self.spans = np.array(spans, dtype=float)
        self.supports = supports
        self.loads = loads
        self.E = float(E)
        self.I = float(I)
        self.nodes = np.concatenate(([0], np.cumsum(self.spans)))
        self.n_nodes = len(self.nodes)
        self.total_len = self.nodes[-1]
        
    def _get_fixed_end_reactions(self, L, load):
        fem = np.zeros(4) 
        mag = load['mag']
        
        if load['type'] == 'P':
            a = load['x']
            b = L - a
            fem[0] = (mag * b**2 * (3*a + b)) / L**3  # V1
            fem[1] = (mag * a * b**2) / L**2          # M1
            fem[2] = (mag * a**2 * (a + 3*b)) / L**3  # V2
            fem[3] = -(mag * a**2 * b) / L**2         # M2
            
        elif load['type'] == 'U':
            start = load['x']
            dist = load.get('dist', L - start) 
            end = start + dist
            w = mag
            # Gauss Quadrature for Integration
            gl_x = np.array([-0.774596669, 0, 0.774596669])
            gl_w = np.array([0.555555556, 0.888888889, 0.555555556])
            mid = (start + end) / 2
            jac = (end - start) / 2
            for i in range(3):
                xi_global = mid + jac * gl_x[i] 
                weight = gl_w[i] * jac * w 
                xi = xi_global / L
                n1 = 1 - 3*xi**2 + 2*xi**3
                n2 = L * (xi - 2*xi**2 + xi**3)
                n3 = 3*xi**2 - 2*xi**3
                n4 = L * (-xi**2 + xi**3)
                fem += weight * np.array([n1, n2, n3, n4])

        # [เพิ่มเติม 1] สูตร Fixed End Moment สำหรับ Point Moment (M)
        elif load['type'] == 'M':
            a = load['x']
            b = L - a
            # Moment Load ทำให้เกิดแรงเฉือนที่ Support ด้วย (Reaction)
            # แต่ไม่ทำให้เกิด Shear Jump ตรงกลางคาน
            
            # FEM formulas for concentrated moment M (Clockwise +)
            # Ref: Roark's Formulas / Standard Structural Analysis
            # V_left
            fem[0] = -(6 * mag * a * b) / L**3 
            # M_left
            fem[1] = (mag * b * (2*a - b)) / L**2 
            # V_right
            fem[2] = (6 * mag * a * b) / L**3
            # M_right
            fem[3] = (mag * a * (2*b - a)) / L**2

        return fem

    def solve(self):
        n_dof = 2 * self.n_nodes
        K = np.zeros((n_dof, n_dof))
        F_node = np.zeros(n_dof) 
        
        # Assemble
        for i, L in enumerate(self.spans):
            k = self.E * self.I / L**3
            k_el = k * np.array([
                [12, 6*L, -12, 6*L],
                [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L],
                [6*L, 2*L**2, -6*L, 4*L**2]
            ])
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            for r in range(4):
                for c in range(4):
                    K[idx[r], idx[c]] += k_el[r, c]
                    
            if not self.loads.empty:
                span_loads = self.loads[self.loads['span_idx'] == i]
                for _, l in span_loads.iterrows():
                    reactions = self._get_fixed_end_reactions(L, l)
                    F_node[idx] -= reactions

        # Boundary Conditions
        active_dof = list(range(n_dof))
        for _, s in self.supports.iterrows():
            node_idx = int(s['id'])
            if 2*node_idx in active_dof: active_dof.remove(2*node_idx)
            if s['type'] == 'Fixed' and 2*node_idx+1 in active_dof: active_dof.remove(2*node_idx+1)

        # Solve
        U = np.zeros(n_dof)
        if len(active_dof) > 0:
            try:
                U[active_dof] = np.linalg.solve(K[np.ix_(active_dof, active_dof)], F_node[active_dof])
            except np.linalg.LinAlgError:
                raise Exception("Structure Unstable")

        R = K @ U - F_node
        return self._post_process(U, R)

    def _post_process(self, U, R):
        x_plot, v_plot, m_plot, d_plot = [], [], [], []
        num_points = 500 
        
        if hasattr(integrate, 'cumulative_trapezoid'): cumtrapz = integrate.cumulative_trapezoid
        else: cumtrapz = integrate.cumtrapz

        for i, L in enumerate(self.spans):
            base_points = np.linspace(0, L, num_points)
            
            crit_points = [0.0, L]
            if not self.loads.empty:
                span_loads = self.loads[self.loads['span_idx'] == i]
                for _, l in span_loads.iterrows():
                    crit_points.append(l['x'])
                    crit_points.append(l['x'] + 1e-9) 
                    
                    if l['type'] == 'U':
                        dist = l.get('dist', L - l['x'])
                        end_x = l['x'] + dist
                        crit_points.append(end_x)
                        crit_points.append(end_x + 1e-9)
            
            merged_points = np.concatenate((base_points, crit_points))
            x_local = np.unique(np.round(merged_points, 9))
            x_global = self.nodes[i] + x_local
            
            # --- Calculation ---
            v_seg = []
            m_seg = []
            
            for k, xg in enumerate(x_global):
                V_val, M_val = 0, 0
                for n_idx, nx in enumerate(self.nodes):
                    if nx <= xg + 1e-9: 
                        V_val += R[2*n_idx]
                        M_val += R[2*n_idx] * (xg - nx) + R[2*n_idx+1]
                
                if not self.loads.empty:
                    for _, l in self.loads.iterrows():
                        lx = self.nodes[int(l['span_idx'])] + l['x']
                        
                        if l['type'] == 'P':
                            if lx <= xg - 1e-9:
                                V_val -= l['mag']
                                M_val -= l['mag'] * (xg - lx)       
                        
                        elif l['type'] == 'U':
                            l_start = lx
                            if xg > l_start + 1e-9:
                                dist = l.get('dist', self.spans[int(l['span_idx'])] - l['x'])
                                l_end = l_start + dist
                                eff_end = min(xg, l_end)
                                eff_len = eff_end - l_start
                                if eff_len > 0:
                                    load_mag = l['mag'] * eff_len
                                    centroid = l_start + eff_len / 2
                                    V_val -= load_mag
                                    M_val -= load_mag * (xg - centroid)
                        
                        # [เพิ่มเติม 2] Logic คำนวณ Moment Diagram เมื่อเจอ Moment Load
                        elif l['type'] == 'M':
                            if lx <= xg - 1e-9:
                                # Moment load doesn't change Shear (V), only Moment (M)
                                # Sign convention: Clockwise load creates a step down in Internal Moment diagram
                                # (Or up, depending on specific convention. Here we subtract to be consistent with P load logic)
                                M_val -= l['mag'] 

                v_seg.append(V_val)
                m_seg.append(M_val)
            
            # --- Deflection ---
            theta_start = U[2*i+1]
            y_start = U[2*i]
            M_arr = np.array(m_seg)
            curvature = M_arr / (self.E * self.I)
            theta_change = cumtrapz(curvature, x_local, initial=0)
            theta_arr = theta_start + theta_change
            y_change = cumtrapz(theta_arr, x_local, initial=0)
            y_arr = y_start + y_change
            
            x_plot.extend(x_global)
            v_plot.extend(v_seg)
            m_plot.extend(m_seg)
            d_plot.extend(y_arr)

        return pd.DataFrame({'x': x_plot, 'shear': v_plot, 'moment': m_plot, 'deflection': d_plot}), R
