import numpy as np
import pandas as pd

class BeamSolver:
    def __init__(self, spans, supports, loads, E, I, A=None, beam_type='Euler', nu=0.3, kappa=5/6):
        """
        Solver ที่เลือกโหมดทฤษฎีคานได้
        
        Parameters:
        - beam_type: 'Euler' (default) หรือ 'Timoshenko'
        - A: พื้นที่หน้าตัด (จำเป็นสำหรับ Timoshenko)
        - nu: Poisson's ratio (default 0.3)
        - kappa: Shear correction factor (default 5/6 สำหรับสี่เหลี่ยม)
        """
        self.spans = np.array(spans, dtype=float)
        self.supports = supports
        self.loads = loads
        self.E = float(E)
        self.I = float(I)
        
        # Parameters for Timoshenko
        self.beam_type = beam_type
        self.A = float(A) if A is not None else 0
        self.nu = float(nu)
        self.kappa = float(kappa)
        
        # Calculate Shear Modulus (G) if needed
        if self.beam_type == 'Timoshenko':
            if self.A <= 0:
                raise ValueError("Area (A) is required for Timoshenko beam analysis.")
            self.G = self.E / (2 * (1 + self.nu))
        
        # Node setup
        self.nodes = np.concatenate(([0], np.cumsum(self.spans)))
        self.n_nodes = len(self.nodes)
        
    def _get_element_stiffness(self, L):
        """
        สร้าง Stiffness Matrix (4x4) ตามทฤษฎีที่เลือก
        """
        # 1. Euler-Bernoulli Stiffness (Classic)
        # ไม่คิด Shear deformation (Phi = 0)
        if self.beam_type == 'Euler':
            k_val = (self.E * self.I) / L**3
            return k_val * np.array([
                [12,      6*L,    -12,     6*L],
                [6*L,     4*L**2, -6*L,    2*L**2],
                [-12,     -6*L,    12,    -6*L],
                [6*L,     2*L**2, -6*L,    4*L**2]
            ])
            
        # 2. Timoshenko Stiffness (Advanced)
        # คิด Shear deformation โดยใช้ตัวแปร Phi
        elif self.beam_type == 'Timoshenko':
            # Shear Area
            As = self.kappa * self.A
            
            # Phi (Shear Deformation Parameter)
            # Phi = 12*EI / (G*As*L^2)
            phi = (12 * self.E * self.I) / (self.G * As * L**2)
            
            # Factor common term
            k_val = (self.E * self.I) / (L**3 * (1 + phi))
            
            # Timoshenko Matrix Elements
            k11 = 12
            k12 = 6 * L
            k22 = (4 + phi) * L**2
            k24 = (2 - phi) * L**2
            
            return k_val * np.array([
                [k11,   k12,   -k11,   k12],
                [k12,   k22,   -k12,   k24],
                [-k11,  -k12,   k11,  -k12],
                [k12,   k24,   -k12,   k22]
            ])
            
    def _get_consistent_nodal_loads(self, L, load):
        # ... (ใช้ฟังก์ชันเดิมจากโพสต์ก่อนหน้าได้เลยครับ ไม่ต้องแก้) ...
        # หมายเหตุ: ในทางทฤษฎี Timoshenko ก็มีผลต่อ Fixed End Moment เล็กน้อย
        # แต่ในทางปฏิบัติมักอนุโลมให้ใช้สูตรเดิมได้สำหรับ Load ทั่วไป
        return self._get_fixed_end_reactions_classic(L, load)

    # (Helper function เดิม แปะมาให้เพื่อความสมบูรณ์)
    def _get_fixed_end_reactions_classic(self, L, load):
        fem = np.zeros(4) 
        mag = load['mag']
        if load['type'] == 'P':
            a = load['x']; b = L - a
            fem[0] = (mag * b**2 * (3*a + b)) / L**3
            fem[1] = (mag * a * b**2) / L**2
            fem[2] = (mag * a**2 * (a + 3*b)) / L**3
            fem[3] = -(mag * a**2 * b) / L**2
        elif load['type'] == 'U':
            # Simplified for UDL full span or partial (Integrate shape func)
            # เพื่อความกระชับ ขอใช้ Gauss method แบบเดิม
            start = load['x']; dist = load.get('dist', L - start); end = start + dist; w = mag
            gl_x = np.array([-0.774596669, 0, 0.774596669])
            gl_w = np.array([0.555555556, 0.888888889, 0.555555556])
            mid = (start + end)/2; jac = (end - start)/2
            for i in range(3):
                xi = (mid + jac*gl_x[i])/L
                n1 = 1 - 3*xi**2 + 2*xi**3; n2 = L*(xi - 2*xi**2 + xi**3)
                n3 = 3*xi**2 - 2*xi**3; n4 = L*(-xi**2 + xi**3)
                fem += gl_w[i] * jac * w * np.array([n1, n2, n3, n4])
        elif load['type'] == 'M':
            a = load['x']; b = L - a
            fem[0] = -(6*mag*a*b)/L**3; fem[1] = (mag*b*(2*a-b))/L**2
            fem[2] = (6*mag*a*b)/L**3; fem[3] = (mag*a*(2*b-a))/L**2
        return fem

    def solve(self):
        n_dof = 2 * self.n_nodes
        K = np.zeros((n_dof, n_dof))
        F_node = np.zeros(n_dof) 
        
        for i, L in enumerate(self.spans):
            # เรียกใช้ฟังก์ชัน Stiffness ตามประเภทคานที่เลือก
            k_el = self._get_element_stiffness(L)
            
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            for r in range(4):
                for c in range(4):
                    K[idx[r], idx[c]] += k_el[r, c]
                    
            if not self.loads.empty:
                span_loads = self.loads[self.loads['span_idx'] == i]
                for _, l in span_loads.iterrows():
                    fea = self._get_consistent_nodal_loads(L, l)
                    F_node[idx] -= fea 

        # Boundary Conditions & Solving
        active_dof = list(range(n_dof))
        for _, s in self.supports.iterrows():
            node_idx = int(s['id'])
            if 2*node_idx in active_dof: active_dof.remove(2*node_idx)
            if s['type'] == 'Fixed' and 2*node_idx+1 in active_dof: active_dof.remove(2*node_idx+1)

        U = np.zeros(n_dof)
        if len(active_dof) > 0:
            U[active_dof] = np.linalg.solve(K[np.ix_(active_dof, active_dof)], F_node[active_dof])

        R = K @ U - F_node
        return self._post_process(U, R)

    def _post_process(self, U, R):
        # ... (ส่วนนี้ใช้โค้ด Pure FEM Interpolation จากโพสต์ก่อนหน้าได้เลย) ...
        # หมายเหตุ: แม้เป็น Timoshenko แต่การใช้ Hermite Cubic Interpolation ในการ Plot กราฟ
        # ยังคงยอมรับได้ในเชิง Visualization ถ้าระยะห่าง Node ไม่กว้างจนเกินไป
        
        # เพื่อความสมบูรณ์ ให้ Copy โค้ด _post_process จาก "Pure FEM" มาวางที่นี่ครับ
        # (ผมละไว้เพื่อประหยัดพื้นที่ ถ้าต้องการให้แปะซ้ำ บอกได้เลยครับ)
        pass
