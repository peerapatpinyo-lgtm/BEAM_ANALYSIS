import numpy as np
import pandas as pd
from scipy import integrate

class BeamSolver:
    def __init__(self, spans, supports, loads, E, I):
        """
        Matrix Stiffness Method Solver (Exact FEM)
        """
        self.spans = np.array(spans, dtype=float)
        self.supports = supports
        self.loads = loads
        self.E = float(E)
        self.I = float(I)
        self.nodes = np.concatenate(([0], np.cumsum(self.spans)))
        self.n_nodes = len(self.nodes)
        self.total_len = self.nodes[-1]
        
    def _get_fixed_end_reactions(self, L, load):
        """
        Calculate Exact Fixed End Reactions (Forces & Moments from Support -> Beam)
        """
        fem = np.zeros(4) # [Fy1, M1, Fy2, M2]
        mag = load['mag']
        
        if load['type'] == 'P':
            # Point Load
            a = load['x']
            b = L - a
            fem[0] = (mag * b**2 * (3*a + b)) / L**3
            fem[1] = (mag * a * b**2) / L**2
            fem[2] = (mag * a**2 * (a + 3*b)) / L**3
            fem[3] = -(mag * a**2 * b) / L**2
            
        elif load['type'] == 'U':
            # Uniform Load (Exact Integration using Gauss Quadrature)
            start = load['x']
            dist = load.get('dist', L - start) 
            end = start + dist
            w = mag
            
            # Gauss-Legendre Quadrature (3 points)
            gl_x = np.array([-0.774596669, 0, 0.774596669])
            gl_w = np.array([0.555555556, 0.888888889, 0.555555556])
            
            mid = (start + end) / 2
            jac = (end - start) / 2
            
            for i in range(3):
                xi_global = mid + jac * gl_x[i] 
                weight = gl_w[i] * jac * w 
                
                # Hermite Shape Functions
                xi = xi_global / L
                n1 = 1 - 3*xi**2 + 2*xi**3
                n2 = L * (xi - 2*xi**2 + xi**3)
                n3 = 3*xi**2 - 2*xi**3
                n4 = L * (-xi**2 + xi**3)
                
                fem += weight * np.array([n1, n2, n3, n4])

        return fem

    def solve(self):
        n_dof = 2 * self.n_nodes
        K = np.zeros((n_dof, n_dof))
        F_node = np.zeros(n_dof) 
        
        # 1. Assemble
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
                    
            # Loads
            if not self.loads.empty:
                span_loads = self.loads[self.loads['span_idx'] == i]
                for _, l in span_loads.iterrows():
                    reactions = self._get_fixed_end_reactions(L, l)
                    F_node[idx] -= reactions # Subtract reactions

        # 2. Boundary Conditions
        active_dof = list(range(n_dof))
        for _, s in self.supports.iterrows():
            node_idx = int(s['id'])
            if 2*node_idx in active_dof: active_dof.remove(2*node_idx)
            if s['type'] == 'Fixed' and 2*node_idx+1 in active_dof: active_dof.remove(2*node_idx+1)

        # 3. Solve Displacements
        U = np.zeros(n_dof)
        if len(active_dof) > 0:
            try:
                U[active_dof] = np.linalg.solve(K[np.ix_(active_dof, active_dof)], F_node[active_dof])
            except np.linalg.LinAlgError:
                raise Exception("Structure Unstable")

        # 4. Reactions
        R = K @ U - F_node
        
        return self._post_process(U, R)

    def _post_process(self, U, R):
        x_plot, v_plot, m_plot, d_plot = [], [], [], []
        num_points = 200 
        
        # Handle integration method depending on scipy version
        if hasattr(integrate, 'cumulative_trapezoid'): cumtrapz = integrate.cumulative_trapezoid
        else: cumtrapz = integrate.cumtrapz

        for i, L in enumerate(self.spans):
            # --- แก้ไข: เพิ่มจุด Critical Points (ตำแหน่ง Load) เข้าไปใน array คำนวณ ---
            # เพื่อให้กราฟแสดงค่า Peak และ Label ได้ตรงตำแหน่งเป๊ะๆ (เช่น @ 2.50m)
            
            # 1. สร้างจุดพื้นฐาน
            base_points = np.linspace(0, L, num_points)
            
            # 2. หาตำแหน่ง Load ใน Span นี้
            crit_points = [0.0, L] # เริ่มและจบ span
            if not self.loads.empty:
                span_loads = self.loads[self.loads['span_idx'] == i]
                for _, l in span_loads.iterrows():
                    crit_points.append(l['x'])
                    if l['type'] == 'U':
                        dist = l.get('dist', L - l['x'])
                        crit_points.append(l['x'] + dist)
            
            # 3. รวมจุดและเรียงลำดับ (Merge & Sort)
            x_local = np.unique(np.concatenate((base_points, crit_points)))
            
            x_global = self.nodes[i] + x_local
            
            # --- จบส่วนแก้ไขการสร้างจุด x ---
            
            # --- 1. Shear & Moment (Exact Statics) ---
            v_seg = []
            m_seg = []
            
            for k, xg in enumerate(x_global):
                V_val, M_val = 0, 0
                
                # Reactions
                for n_idx, nx in enumerate(self.nodes):
                    if nx <= xg + 1e-5: 
                        V_val += R[2*n_idx]
                        M_val += R[2*n_idx] * (xg - nx) + R[2*n_idx+1]
                
                # Loads
                if not self.loads.empty:
                    for _, l in self.loads.iterrows():
                        lx = self.nodes[int(l['span_idx'])] + l['x']
                        
                        if l['type'] == 'P':
                            if lx <= xg - 1e-5:
                                V_val -= l['mag']
                                M_val -= l['mag'] * (xg - lx)
                                
                        elif l['type'] == 'U':
                            l_start = lx
                            if xg > l_start + 1e-5:
                                dist = l.get('dist', self.spans[int(l['span_idx'])] - l['x'])
                                l_end = l_start + dist
                                eff_end = min(xg, l_end)
                                eff_len = eff_end - l_start
                                
                                if eff_len > 0:
                                    load_mag = l['mag'] * eff_len
                                    centroid = l_start + eff_len / 2
                                    V_val -= load_mag
                                    M_val -= load_mag * (xg - centroid)
                
                v_seg.append(V_val)
                m_seg.append(M_val)
            
            # --- 2. Deflection (Double Integration of Moment) ---
            theta_start = U[2*i+1] # Slope at start node
            y_start = U[2*i]       # Deflection at start node
            
            M_arr = np.array(m_seg)
            
            # 1st Integration: Slope (theta)
            curvature = M_arr / (self.E * self.I)
            theta_change = cumtrapz(curvature, x_local, initial=0)
            theta_arr = theta_start + theta_change
            
            # 2nd Integration: Deflection (y)
            y_change = cumtrapz(theta_arr, x_local, initial=0)
            y_arr = y_start + y_change
            
            # Store results
            x_plot.extend(x_global)
            v_plot.extend(v_seg)
            m_plot.extend(m_seg)
            d_plot.extend(y_arr)

        return pd.DataFrame({
            'x': x_plot, 
            'shear': v_plot, 
            'moment': m_plot, 
            'deflection': d_plot
        }), R
