import numpy as np
import pandas as pd

class BeamSolver:
    def __init__(self, spans, supports, loads, E, I):
        self.spans = np.array(spans, dtype=float)
        self.supports = supports
        self.loads = loads
        self.E = float(E)
        self.I = float(I)
        
        # Node positions
        self.nodes = np.concatenate(([0], np.cumsum(self.spans)))
        self.n_nodes = len(self.nodes)
        self.total_len = self.nodes[-1]
        
    def _get_consistent_nodal_loads(self, L, load):
        """
        แปลง Local Load ให้เป็น Equivalent Nodal Loads (Consistent Loads)
        โดยใช้หลักการ Work Equivalent หรือ Fixed End Reactions
        Return: [Fy_left, M_left, Fy_right, M_right] (Vector นี้คือแรงที่ Node รับ)
        """
        # เราคำนวณจาก Fixed End Reaction แล้วกลับเครื่องหมายเพื่อเป็น Nodal Load
        # หรือใช้สูตร Equivalent Nodal Load โดยตรง (ค่าเท่ากันแต่ทิศตรงข้ามกับ Reaction)
        
        # ในที่นี้ผมคำนวณเป็น Fixed End Reactions (FEM) ก่อน 
        # แล้วค่อยไปลบออกจาก F_node ใน loop หลัก (F_node -= FEM) ซึ่งมีค่าเท่ากับ F_node += EquivalentLoad
        
        fem = np.zeros(4) 
        mag = load['mag']
        
        if load['type'] == 'P':
            a = load['x']
            b = L - a
            # Reaction forces caused by load (Fixed Ends)
            fem[0] = (mag * b**2 * (3*a + b)) / L**3
            fem[1] = (mag * a * b**2) / L**2
            fem[2] = (mag * a**2 * (a + 3*b)) / L**3
            fem[3] = -(mag * a**2 * b) / L**2
            
        elif load['type'] == 'U':
            start = load['x']
            dist = load.get('dist', L - start) 
            end = start + dist
            w = mag
            
            # ใช้ Gauss Quadrature เพื่อหา Consistent Nodal Loads ของแรงแผ่
            # Int(N^T * w) dx
            gl_x = np.array([-0.774596669, 0, 0.774596669])
            gl_w = np.array([0.555555556, 0.888888889, 0.555555556])
            mid = (start + end) / 2
            jac = (end - start) / 2
            
            for i in range(3):
                xi_global = mid + jac * gl_x[i] 
                weight = gl_w[i] * jac * w 
                xi = xi_global / L
                
                # Hermite Shape Functions
                n1 = 1 - 3*xi**2 + 2*xi**3
                n2 = L * (xi - 2*xi**2 + xi**3)
                n3 = 3*xi**2 - 2*xi**3
                n4 = L * (-xi**2 + xi**3)
                
                fem += weight * np.array([n1, n2, n3, n4])

        elif load['type'] == 'M':
            a = load['x']
            b = L - a
            fem[0] = -(6 * mag * a * b) / L**3 
            fem[1] = (mag * b * (2*a - b)) / L**2 
            fem[2] = (6 * mag * a * b) / L**3
            fem[3] = (mag * a * (2*b - a)) / L**2

        return fem

    def solve(self):
        """
        System Solving: K*U = F_equivalent
        """
        n_dof = 2 * self.n_nodes
        K = np.zeros((n_dof, n_dof))
        F_node = np.zeros(n_dof) 
        
        # 1. Assemble K and F
        for i, L in enumerate(self.spans):
            # Stiffness Matrix (Bernoulli-Euler Beam)
            k = self.E * self.I / L**3
            k_el = k * np.array([
                [12, 6*L, -12, 6*L],
                [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L],
                [6*L, 2*L**2, -6*L, 4*L**2]
            ])
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            
            # Add to Global K
            for r in range(4):
                for c in range(4):
                    K[idx[r], idx[c]] += k_el[r, c]
                    
            # Add Consistent Nodal Loads from Member Loads
            if not self.loads.empty:
                span_loads = self.loads[self.loads['span_idx'] == i]
                for _, l in span_loads.iterrows():
                    # Get Fixed End Actions
                    fea = self._get_consistent_nodal_loads(L, l)
                    # Subtract FEA from Nodes = Adding Equivalent Nodal Loads
                    F_node[idx] -= fea 

        # 2. Apply Boundary Conditions
        active_dof = list(range(n_dof))
        for _, s in self.supports.iterrows():
            node_idx = int(s['id'])
            if 2*node_idx in active_dof: active_dof.remove(2*node_idx)
            if s['type'] == 'Fixed' and 2*node_idx+1 in active_dof: active_dof.remove(2*node_idx+1)

        # 3. Solve for Nodal Displacements (U)
        U = np.zeros(n_dof)
        if len(active_dof) > 0:
            try:
                U[active_dof] = np.linalg.solve(K[np.ix_(active_dof, active_dof)], F_node[active_dof])
            except np.linalg.LinAlgError:
                raise Exception("Structure Unstable")

        # 4. Calculate Reaction Forces (R)
        # R = K*U - F_external_nodal (Wait, R = K*U - F_eq)
        # Note: F_node here stores Equivalent Nodal Loads.
        R = K @ U - F_node
        
        return self._post_process(U, R)

    def _post_process(self, U, R):
        x_plot, v_plot, m_plot, d_plot = [], [], [], []
        num_points = 100 # Resolution for plotting

        for i, L in enumerate(self.spans):
            # Generate Points
            base_points = np.linspace(0, L, num_points)
            crit_points = [0.0, L]
            
            # Add critical points for Load Discontinuities (Visual Sharpness for Shear/Moment)
            if not self.loads.empty:
                span_loads = self.loads[self.loads['span_idx'] == i]
                for _, l in span_loads.iterrows():
                    crit_points.append(l['x'])
                    crit_points.append(l['x'] + 1e-9) 
                    if l['type'] == 'U':
                        dist = l.get('dist', L - l['x'])
                        crit_points.append(l['x'] + dist)
            
            merged_points = np.concatenate((base_points, crit_points))
            x_local = np.unique(np.round(merged_points, 9))
            x_global = self.nodes[i] + x_local
            
            # --- A. Shear & Moment (Statics Method) ---
            # ยังคงใช้วิธี Statics (ตัด Section) เพราะแม่นยำที่สุดสำหรับการพลอตกราฟแรงภายใน
            v_seg, m_seg = [], []
            for xg in x_global:
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
                                V_val -= l['mag']; M_val -= l['mag'] * (xg - lx)       
                        elif l['type'] == 'U':
                            l_start = lx
                            if xg > l_start + 1e-9:
                                dist = l.get('dist', self.spans[int(l['span_idx'])] - l['x'])
                                l_end = l_start + dist; eff_end = min(xg, l_end)
                                eff_len = eff_end - l_start
                                if eff_len > 0:
                                    load_mag = l['mag'] * eff_len
                                    cent = l_start + eff_len / 2
                                    V_val -= load_mag; M_val -= load_mag * (xg - cent)
                        elif l['type'] == 'M':
                            if lx <= xg - 1e-9: M_val -= l['mag'] 
                v_seg.append(V_val)
                m_seg.append(M_val)
            
            # --- B. Deflection Calculation (PURE FEM SHAPE FUNCTION) ---
            # ใช้เพียง Node Displacement (U) และ Shape Function (N) เท่านั้น
            # v(x) = [N]{u}
            
            u_L = U[2*i]      # y_left
            th_L = U[2*i+1]   # theta_left
            u_R = U[2*i+2]    # y_right
            th_R = U[2*i+3]   # theta_right
            
            xi = x_local / L
            
            # Hermite Shape Functions
            n1 = 1 - 3*xi**2 + 2*xi**3
            n2 = L * (xi - 2*xi**2 + xi**3)
            n3 = 3*xi**2 - 2*xi**3
            n4 = L * (-xi**2 + xi**3)
            
            # Interpolated Deflection
            y_arr = n1*u_L + n2*th_L + n3*u_R + n4*th_R
            
            # Store results
            x_plot.extend(x_global)
            v_plot.extend(v_seg)
            m_plot.extend(m_seg)
            d_plot.extend(y_arr)

        return pd.DataFrame({'x': x_plot, 'shear': v_plot, 'moment': m_plot, 'deflection': d_plot}), R
