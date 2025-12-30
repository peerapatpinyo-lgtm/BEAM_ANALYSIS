import numpy as np
import pandas as pd

class BeamSolver:
    def __init__(self, spans, supports, loads):
        self.spans = spans
        self.supports = supports
        self.loads = loads
        self.n_nodes = len(spans) + 1
        self.total_length = sum(spans)
        self.nodes = [0] + list(np.cumsum(spans))
        
    def solve(self):
        # --- 1. FEM Process (หา Reaction ที่จุดรองรับ) ---
        n_dof = 2 * self.n_nodes
        K_global = np.zeros((n_dof, n_dof))
        F_global = np.zeros(n_dof)
        
        # Element Stiffness & Fixed End Forces
        for i, L in enumerate(self.spans):
            k = self._element_stiffness(L)
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            
            # Assembly K
            for r in range(4):
                for c in range(4):
                    K_global[idx[r], idx[c]] += k[r, c]
            
            # Load Vector (FEM Load)
            fem = np.zeros(4)
            span_loads = [l for l in self.loads if int(l['span_idx']) == i]
            
            for load in span_loads:
                if load['type'] == 'P':
                    a = load['x']
                    b = L - a
                    P = load['P']
                    L2, L3 = L**2, L**3
                    fem[0] += P * b**2 * (3*a + b) / L3
                    fem[1] += P * a * b**2 / L2
                    fem[2] += P * a**2 * (a + 3*b) / L3
                    fem[3] -= P * a**2 * b / L2
                elif load['type'] == 'U':
                    w = load['w']
                    fem[0] += w * L / 2
                    fem[1] += w * L**2 / 12
                    fem[2] += w * L / 2
                    fem[3] -= w * L**2 / 12
            
            F_global[idx] += fem

        # Boundary Conditions
        free_dofs = list(range(n_dof))
        fixed_dofs = []
        
        for _, sup in self.supports.iterrows():
            node_idx = int(sup['id'])
            fixed_dofs.append(2*node_idx) # Fix Y
            if sup['type'] == 'Fixed':
                fixed_dofs.append(2*node_idx + 1) # Fix Moment
        
        fixed_dofs = sorted(list(set(fixed_dofs)))
        for d in sorted(fixed_dofs, reverse=True):
            if d in free_dofs: free_dofs.remove(d)
            
        # Solve Displacements
        K_ff = K_global[np.ix_(free_dofs, free_dofs)]
        F_f = F_global[free_dofs]
        
        try:
            U_f = np.linalg.solve(K_ff, F_f)
        except:
            U_f = np.zeros(len(free_dofs))

        U_global = np.zeros(n_dof)
        U_global[free_dofs] = U_f
        
        # Calculate Reactions
        Reactions = K_global @ U_global - F_global
        
        # --- 2. Post-Processing (สร้างกราฟแบบ Textbook) ---
        results = []
        
        # รวบรวมตำแหน่งสำคัญ (Critical Points) เพื่อสร้างกราฟ
        # จุดสำคัญคือ: หัวคาน, ท้ายคาน, จุดรองรับ, จุดที่มี Load
        critical_x = set()
        critical_x.add(0)
        critical_x.add(self.total_length)
        
        # Add Node locations
        for n in self.nodes: critical_x.add(n)
        
        # Add Load locations
        for l in self.loads:
            global_x_start = self.nodes[int(l['span_idx'])]
            if l['type'] == 'P':
                critical_x.add(global_x_start + l['x'])
        
        # Convert set to sorted list
        sorted_x = sorted(list(critical_x))
        
        # สร้าง Mesh ถี่ๆ สำหรับวาดกราฟ (Evaluation Points)
        # เทคนิค: ที่จุดวิกฤต x เราจะคำนวณที่ x-epsilon และ x+epsilon
        # เพื่อให้เกิด "เส้นดิ่ง" ในกราฟ Shear
        eval_points = []
        epsilon = 1e-5
        
        for i in range(len(sorted_x) - 1):
            x_start = sorted_x[i]
            x_end = sorted_x[i+1]
            
            # จุดซ้ายสุดของช่วง
            eval_points.append(x_start)
            if x_start != 0: eval_points.append(x_start + epsilon)
            
            # จุดระหว่างกลาง (ละเอียดเพื่อให้กราฟ Moment โค้งสวย)
            sub_div = np.linspace(x_start, x_end, 20) # 20 จุดต่อช่วงย่อย
            for val in sub_div[1:-1]:
                eval_points.append(val)
                
            # จุดขวาสุดของช่วง
            if x_end != self.total_length: eval_points.append(x_end - epsilon)
            eval_points.append(x_end)

        # คำนวณ V, M ที่ทุกจุด
        # วิธีการ: ตัด Section จากซ้ายไปขวา (Method of Sections)
        for x in eval_points:
            V_x = 0.0
            M_x = 0.0
            
            # 1. ผลจาก Reaction ด้านซ้ายของ x
            for i in range(self.n_nodes):
                node_x = self.nodes[i]
                if node_x <= x: # Reaction อยู่ซ้ายมือ
                    # Force reaction (Up is positive)
                    Ry = Reactions[2*i]
                    V_x += Ry
                    M_x += Ry * (x - node_x)
                    
                    # Moment reaction (CCW is positive)
                    Mz = Reactions[2*i+1]
                    M_x += Mz # Moment couple directly adds to M
            
            # 2. ผลจาก Loads ด้านซ้ายของ x
            for l in self.loads:
                lx_start = self.nodes[int(l['span_idx'])]
                
                if l['type'] == 'P':
                    load_x = lx_start + l['x']
                    if load_x <= x: # Load อยู่ซ้ายมือ
                        P = l['P']
                        V_x -= P # Load ลงเป็นลบ
                        M_x -= P * (x - load_x)
                        
                elif l['type'] == 'U':
                    # Load แผ่ เริ่มต้นและสิ้นสุดของ Span นี้
                    x_span_start = lx_start
                    x_span_end = self.nodes[int(l['span_idx'])+1]
                    
                    # ช่วงที่ Load กระทำจริงๆ (Intersection)
                    # เราสนใจเฉพาะส่วนที่อยู่ "ซ้ายมือ" ของ x
                    effective_end = min(x, x_span_end)
                    
                    if effective_end > x_span_start:
                        len_eff = effective_end - x_span_start
                        w = l['w']
                        total_load = w * len_eff
                        # Centroid ของ load ส่วนนี้
                        centroid_dist = (effective_end - x_span_start) / 2
                        moment_arm = x - (x_span_start + centroid_dist)
                        
                        V_x -= total_load
                        M_x -= total_load * moment_arm
            
            # ปัดเศษเล็กน้อยป้องกัน Error ทาง Math (-0.00000 -> 0)
            if abs(V_x) < 1e-4: V_x = 0
            if abs(M_x) < 1e-4: M_x = 0
            
            results.append({'x': x, 'shear': V_x, 'moment': M_x})
            
        return pd.DataFrame(results), Reactions

    def _element_stiffness(self, L, E=1, I=1):
        k = (E * I / L**3) * np.array([
            [12, 6*L, -12, 6*L],
            [6*L, 4*L**2, -6*L, 2*L**2],
            [-12, -6*L, 12, -6*L],
            [6*L, 2*L**2, -6*L, 4*L**2]
        ])
        return k
