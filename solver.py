import numpy as np
import pandas as pd

class AdvancedBeamSolver:
    def __init__(self, spans, supports, loads, properties):
        """
        Professional FEM Beam Solver
        
        Parameters:
        - spans: list of lengths [L1, L2, ...]
        - supports: list of dicts 
          {'id': node_idx, 'type': 'Pin'/'Fixed'/'Roller', 'settlement': val, 'k_spring': val}
          (Note: 'id' here refers to the original span connection points)
        - loads: list of dicts or DataFrame with 'span_idx', 'type', 'mag', 'x'
        - properties: dict {'E': float, 'I': float, 'A': float, 'type': 'Euler'/'Timoshenko', 'nu': 0.3, 'kappa': 5/6}
        """
        self.raw_spans = np.array(spans, dtype=float)
        self.raw_supports = supports
        self.raw_loads = pd.DataFrame(loads) if isinstance(loads, list) else loads
        self.props = properties
        
        # Unpack Properties
        self.E = float(properties.get('E', 2e11))
        self.I = float(properties.get('I', 1e-4))
        self.A = float(properties.get('A', 0.01))
        self.beam_type = properties.get('type', 'Euler')
        self.nu = properties.get('nu', 0.3)
        self.kappa = properties.get('kappa', 5/6.0)
        
        if self.beam_type == 'Timoshenko':
            self.G = self.E / (2 * (1 + self.nu))
            
        # --- 1. PRE-PROCESSING (Auto-Meshing) ---
        self._generate_global_mesh()

    def _generate_global_mesh(self):
        """
        หัวใจสำคัญ: แปลง Input บ้านๆ ให้เป็น Finite Element Mesh ที่ละเอียด
        โดยการตัด Node เพิ่มตรงจุดที่มี Load หรือ Support
        """
        # 1.1 Map Original Nodes to Global X
        original_nodes = np.concatenate(([0], np.cumsum(self.raw_spans)))
        
        # 1.2 Collect all critical X coordinates (Cut points)
        cut_points = set(original_nodes)
        
        # Add Load locations to cut points
        if not self.raw_loads.empty:
            for _, l in self.raw_loads.iterrows():
                # Global X of load
                span_start = original_nodes[int(l['span_idx'])]
                lx_global = span_start + l['x']
                
                cut_points.add(lx_global)
                
                if l['type'] == 'U':
                    # Add end point of UDL
                    dist = l.get('dist', self.raw_spans[int(l['span_idx'])] - l['x'])
                    cut_points.add(lx_global + dist)

        # 1.3 Create Final Mesh Nodes (Sorted and Unique)
        self.nodes = np.array(sorted(list(cut_points)))
        self.n_nodes = len(self.nodes)
        self.dof = 2 * self.n_nodes # Total Degrees of Freedom
        
        # 1.4 Generate Elements (Connectivity)
        # Element เชื่อมระหว่าง Node i และ i+1
        self.elements = []
        for i in range(self.n_nodes - 1):
            x_start = self.nodes[i]
            x_end = self.nodes[i+1]
            length = x_end - x_start
            if length > 1e-9: # Avoid zero length elements
                self.elements.append({
                    'id': i,
                    'n1': i,
                    'n2': i+1,
                    'L': length,
                    'x_start': x_start
                })
    
    def _get_element_stiffness(self, L):
        """ Calculate Stiffness Matrix (k) based on Beam Theory """
        if self.beam_type == 'Euler':
            k_val = (self.E * self.I) / L**3
            return k_val * np.array([
                [12, 6*L, -12, 6*L], [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L], [6*L, 2*L**2, -6*L, 4*L**2]
            ])
        elif self.beam_type == 'Timoshenko':
            As = self.kappa * self.A
            phi = (12 * self.E * self.I) / (self.G * As * L**2)
            k_val = (self.E * self.I) / (L**3 * (1 + phi))
            k11, k12 = 12, 6*L
            k22, k24 = (4+phi)*L**2, (2-phi)*L**2
            return k_val * np.array([
                [k11, k12, -k11, k12], [k12, k22, -k12, k24],
                [-k11, -k12, k11, -k12], [k12, k24, -k12, k22]
            ])

    def _get_consistent_nodal_loads(self, el, load_type, mag, local_x, dist=0):
        """
        แปลง Load บน Element ให้เป็น Equivalent Nodal Loads
        el: element dict
        local_x: ตำแหน่งเทียบกับจุดเริ่ม Element
        """
        L = el['L']
        fem = np.zeros(4) # [Fy1, M1, Fy2, M2]
        
        # --- Handle Loads based on Type ---
        if load_type == 'P':
            # Point load MUST be at node for Pure FEM? 
            # With auto-meshing, Point loads are mostly AT nodes, 
            # but we keep this logic for robustness.
            a = local_x
            b = L - a
            fem = np.array([
                (mag*b**2*(3*a+b))/L**3, (mag*a*b**2)/L**2,
                (mag*a**2*(a+3*b))/L**3, -(mag*a**2*b)/L**2
            ])
            
        elif load_type == 'M':
            a = local_x; b = L - a
            fem = np.array([
                -(6*mag*a*b)/L**3, (mag*b*(2*a-b))/L**2,
                (6*mag*a*b)/L**3, (mag*a*(2*b-a))/L**2
            ])

        elif load_type == 'U':
            # Gauss Quadrature for Partial/Full UDL inside this element
            # Note: Because we auto-mesh, complex partial UDLs are split 
            # into simpler full/partial UDLs on each small element.
            start = local_x
            end = min(start + dist, L) # Clip at element end
            if end > start:
                w = mag
                gl_x = np.array([-0.774596669, 0, 0.774596669])
                gl_w = np.array([0.555555556, 0.888888889, 0.555555556])
                mid = (start + end)/2; jac = (end - start)/2
                for i in range(3):
                    xi = (mid + jac*gl_x[i])/L
                    n1 = 1 - 3*xi**2 + 2*xi**3; n2 = L*(xi - 2*xi**2 + xi**3)
                    n3 = 3*xi**2 - 2*xi**3; n4 = L*(-xi**2 + xi**3)
                    fem += gl_w[i] * jac * w * np.array([n1, n2, n3, n4])

        return fem # This is Reaction. Load Vector = -Reaction

    def solve(self):
        # 1. Initialize Global System
        K = np.zeros((self.dof, self.dof))
        F = np.zeros(self.dof)
        
        # 2. Assemble Stiffness & Loads
        # เราต้อง Map Original Loads ใส่ New Elements
        original_nodes_x = np.concatenate(([0], np.cumsum(self.raw_spans)))

        for el in self.elements:
            # 2.1 Stiffness
            k_el = self._get_element_stiffness(el['L'])
            idx = [2*el['n1'], 2*el['n1']+1, 2*el['n2'], 2*el['n2']+1]
            
            for r in range(4):
                for c in range(4):
                    K[idx[r], idx[c]] += k_el[r, c]
            
            # 2.2 Process Loads (Mapping Global Load -> Local Element)
            if not self.raw_loads.empty:
                for _, l in self.raw_loads.iterrows():
                    # Check if this load affects this element
                    l_span_start = original_nodes_x[int(l['span_idx'])]
                    l_global_x = l_span_start + l['x']
                    
                    # Transform to local coordinates of this element
                    local_x_start = l_global_x - el['x_start']
                    
                    affects = False
                    dist = 0
                    
                    if l['type'] in ['P', 'M']:
                        # Point load falls strictly inside or on boundary of this element
                        if 0 <= local_x_start <= el['L'] + 1e-9:
                            affects = True
                    elif l['type'] == 'U':
                        # UDL overlaps with this element
                        l_dist = l.get('dist', self.raw_spans[int(l['span_idx'])] - l['x'])
                        l_global_end = l_global_x + l_dist
                        
                        el_global_end = el['x_start'] + el['L']
                        
                        # Check overlap logic
                        overlap_start = max(l_global_x, el['x_start'])
                        overlap_end = min(l_global_end, el_global_end)
                        
                        if overlap_end > overlap_start + 1e-9:
                            affects = True
                            local_x_start = overlap_start - el['x_start']
                            dist = overlap_end - overlap_start

                    if affects:
                        # Calculate Fixed End Reactions
                        fea = self._get_consistent_nodal_loads(el, l['type'], l['mag'], local_x_start, dist)
                        # F_node = - Reaction
                        F[idx] -= fea

        # 3. Apply Boundary Conditions (Supports, Settlements, Springs)
        # Using Partition Method / Penalty Method approach
        
        # Map original support IDs to new Node IDs
        # (Original support at index i corresponds to global X = original_nodes_x[i])
        support_map = {} # {global_node_idx: support_data}
        for s in self.raw_supports:
            orig_id = int(s['id'])
            s_loc = original_nodes_x[orig_id]
            # Find closest node in new mesh
            node_idx = np.argmin(np.abs(self.nodes - s_loc))
            support_map[node_idx] = s

        # Identify DOFs to fix (Constrained) vs Free
        fixed_dofs = {} # {dof_index: prescribed_value}
        
        for n_idx, s in support_map.items():
            # Vertical Support
            if s.get('k_spring', 0) > 0:
                # Spring Support: Add k to Diagonal of K
                K[2*n_idx, 2*n_idx] += s['k_spring']
            else:
                # Rigid Support: Prescribe Displacement
                # Default settlement = 0 if not specified
                settlement = s.get('settlement', 0.0)
                fixed_dofs[2*n_idx] = settlement
                
            # Rotation Support
            if s['type'] == 'Fixed':
                fixed_dofs[2*n_idx+1] = 0.0 # Rotation = 0

        # 4. Solving (Partition Method for Prescribed Displacements)
        # Separate Free DOFs and Constrained DOFs
        free_dofs = [i for i in range(self.dof) if i not in fixed_dofs]
        cons_dofs = list(fixed_dofs.keys())
        U_cons = np.array(list(fixed_dofs.values()))
        
        U = np.zeros(self.dof)
        U[cons_dofs] = U_cons
        
        if len(free_dofs) > 0:
            # F_free_effective = F_free - K_fc * U_c
            K_ff = K[np.ix_(free_dofs, free_dofs)]
            K_fc = K[np.ix_(free_dofs, cons_dofs)]
            F_f = F[free_dofs]
            
            F_eff = F_f - K_fc @ U_cons
            
            try:
                U_f = np.linalg.solve(K_ff, F_eff)
                U[free_dofs] = U_f
            except np.linalg.LinAlgError:
                raise Exception("Unstable Structure")

        # 5. Reactions
        R = K @ U - F
        
        return self._post_process(U, R)

    def _post_process(self, U, R):
        """ 
        Pure FEM Interpolation Post-Processing 
        ตอนนี้เราใช้ Mesh ละเอียดแล้ว สามารถใช้ Shape Function ได้อย่างมั่นใจ
        """
        x_plot, v_plot, m_plot, d_plot = [], [], [], []
        
        # Plot แต่ละ Element (ที่มีขนาดเล็กๆ)
        for el in self.elements:
            # Sub-divide for smooth curve plotting
            xi_arr = np.linspace(0, 1, 20) 
            x_local_arr = xi_arr * el['L']
            x_global_arr = el['x_start'] + x_local_arr
            
            # Nodal Displacements for this element
            u1 = U[2*el['n1']]; th1 = U[2*el['n1']+1]
            u2 = U[2*el['n2']]; th2 = U[2*el['n2']+1]
            
            # --- Deflection (Pure Interpolation) ---
            # v(x) = N * U
            L = el['L']
            n1 = 1 - 3*xi_arr**2 + 2*xi_arr**3
            n2 = L * (xi_arr - 2*xi_arr**2 + xi_arr**3)
            n3 = 3*xi_arr**2 - 2*xi_arr**3
            n4 = L * (-xi_arr**2 + xi_arr**3)
            
            y_curve = n1*u1 + n2*th1 + n3*u2 + n4*th2
            
            # --- Shear & Moment (From Equilibrium/Statics) ---
            # คำนวณแบบแม่นยำด้วยการตัด Section จากซ้ายสุด (Global Statics)
            # เพื่อความต่อเนื่องของกราฟ
            v_curve = []
            m_curve = []
            
            for xg in x_global_arr:
                V_val, M_val = 0, 0
                
                # Sum Reactions (Nodes < xg)
                # Note: R vector index maps to new Mesh Nodes
                for n_idx in range(self.n_nodes):
                    nx = self.nodes[n_idx]
                    if nx <= xg + 1e-9:
                        V_val += R[2*n_idx]
                        M_val += R[2*n_idx] * (xg - nx) + R[2*n_idx+1]
                
                # Sum Loads (Global Loads < xg)
                original_nodes_x = np.concatenate(([0], np.cumsum(self.raw_spans)))
                if not self.raw_loads.empty:
                    for _, l in self.raw_loads.iterrows():
                        l_start = original_nodes_x[int(l['span_idx'])] + l['x']
                        
                        if l['type'] == 'P':
                            if l_start <= xg - 1e-9:
                                V_val -= l['mag']
                                M_val -= l['mag'] * (xg - l_start)
                        elif l['type'] == 'M':
                            if l_start <= xg - 1e-9:
                                M_val -= l['mag']
                        elif l['type'] == 'U':
                            # Global UDL logic
                            l_dist = l.get('dist', self.raw_spans[int(l['span_idx'])] - l['x'])
                            l_end = l_start + l_dist
                            
                            # Active portion
                            eff_start = l_start
                            eff_end = min(xg, l_end)
                            
                            if eff_end > eff_start + 1e-9:
                                len_act = eff_end - eff_start
                                load_act = l['mag'] * len_act
                                cent_act = eff_start + len_act/2
                                V_val -= load_act
                                M_val -= load_act * (xg - cent_act)
                                
                v_curve.append(V_val)
                m_curve.append(M_val)

            x_plot.extend(x_global_arr)
            d_plot.extend(y_curve)
            v_plot.extend(v_curve)
            m_plot.extend(m_curve)

        return pd.DataFrame({'x': x_plot, 'shear': v_plot, 'moment': m_plot, 'deflection': d_plot}), R
