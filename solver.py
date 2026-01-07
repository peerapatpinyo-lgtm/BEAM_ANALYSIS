import numpy as np
import pandas as pd
from scipy.linalg import solve

class BeamSolver:
    def __init__(self, spans, supports_input, loads_input, E, b=0.3, h=0.5, I_custom=None):
        self.spans = [float(s) for s in spans]
        self.E = float(E)
        self.b = b
        self.h = h
        # I (Moment of Inertia)
        self.I = float(I_custom) if I_custom else (b * h**3) / 12
        
        # Shear Modulus (G) for Concrete (approx Poisson ratio = 0.2)
        self.G = self.E / (2 * (1 + 0.2)) 
        
        # Shear Area (As) for rectangular section
        self.As = (5/6) * (b * h)        
        
        # Node setup
        self.cum_spans = [round(x, 4) for x in ([0.0] + list(np.cumsum(self.spans)))]
        self.loads_df = pd.DataFrame(loads_input)
        self.supports_df = pd.DataFrame(supports_input)

    def _get_phi(self, L):
        """คำนวณค่า Phi สำหรับ Timoshenko Beam Element"""
        if L == 0: return 0
        EI = self.E * self.I
        return (12 * EI) / (L**2 * self.G * self.As)

    def _get_k_timoshenko(self, L):
        """สร้าง Element Stiffness Matrix (Timoshenko)"""
        EI = self.E * self.I
        Phi = self._get_phi(L)
        
        if L == 0: return np.zeros((4,4))

        coeff = EI / (L**3 * (1 + Phi))
        
        # Stiffness Matrix 4x4
        return coeff * np.array([
            [12, 6*L, -12, 6*L],
            [6*L, (4+Phi)*L**2, -6*L, (2-Phi)*L**2],
            [-12, -6*L, 12, -6*L],
            [6*L, (2-Phi)*L**2, -6*L, (4+Phi)*L**2]
        ])

    def solve(self):
        try:
            # --- 1. Generate Nodes (Recheck UDL Start/End) ---
            # สร้างจุด Nodes จาก Support และ Load locations
            pts = self.cum_spans.copy()
            
            if not self.loads_df.empty:
                for _, l in self.loads_df.iterrows():
                    # Global X Start
                    gx_start = self.cum_spans[int(l['span_index'])] + float(l['x'])
                    pts.append(round(gx_start, 4))
                    
                    # Global X End (เฉพาะ Uniform Load)
                    if l['type'] == 'U': 
                        gx_end = gx_start + float(l['dist'])
                        pts.append(round(gx_end, 4)) # <--- Add End Point Explicitly
            
            # Sort และลบจุดซ้ำ (Unique)
            nodes = sorted(list(set([round(p, 4) for p in pts])))
            num_nodes = len(nodes)
            dof = 2 * num_nodes # 2 DOF per node (Y, Theta)
            
            K = np.zeros((dof, dof))
            F = np.zeros(dof)

            # --- 2. Assemble Global Stiffness Matrix (K) ---
            for i in range(num_nodes - 1):
                L = nodes[i+1] - nodes[i]
                if L > 1e-5:
                    idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
                    k_local = self._get_k_timoshenko(L)
                    K[np.ix_(idx, idx)] += k_local

            # --- 3. Assemble Load Vector (F) ---
            if not self.loads_df.empty:
                for _, l in self.loads_df.iterrows():
                    gx = self.cum_spans[int(l['span_index'])] + float(l['x'])
                    mag = float(l['mag']) 
                    
                    if l['type'] == 'P': # Point Load
                        # หา Node ที่ตรงกับตำแหน่ง Load
                        nid = np.argmin([abs(n - gx) for n in nodes])
                        F[2*nid] -= mag
                        
                    elif l['type'] == 'M': # Moment Load
                        nid = np.argmin([abs(n - gx) for n in nodes])
                        F[2*nid+1] += mag
                        
                    elif l['type'] == 'U': # UDL
                        dist = float(l['dist'])
                        w = mag # N/m
                        
                        # Loop หาทุก Element ที่อยู่ใต้ UDL
                        # (เพราะ UDL อาจจะคลุมหลาย Node ที่เราซอยไว้)
                        for i in range(num_nodes - 1):
                            x_i = nodes[i]
                            x_j = nodes[i+1]
                            
                            # เช็คช่วงซ้อนทับ (Overlap) ระหว่าง Element กับ Load
                            load_start = gx
                            load_end = gx + dist
                            
                            start_overlap = max(x_i, load_start)
                            end_overlap = min(x_j, load_end)
                            
                            overlap_len = end_overlap - start_overlap
                            
                            if overlap_len > 1e-5:
                                # กรณีนี้ Load ทับ Element เต็มๆ หรือบางส่วน
                                # เพื่อความง่ายและแม่นยำใน FEA แบบ 1D 
                                # เราจะกระจายแรงเข้า Node (Work Equivalent Load)
                                # *สมมติว่า Node ตรงกับช่วง Load พอดีจากการ generate node*
                                
                                Le = x_j - x_i
                                # Fixed End Forces สำหรับ UDL เต็มช่วง Element
                                # Fy = wL/2, M = wL^2/12
                                
                                F[2*i] -= (w * Le / 2)
                                F[2*i+1] -= (w * Le**2 / 12)
                                F[2*(i+1)] -= (w * Le / 2)
                                F[2*(i+1)+1] += (w * Le**2 / 12)

            # --- 4. Apply Boundary Conditions ---
            free_d = np.full(dof, True)
            if not self.supports_df.empty:
                for _, s in self.supports_df.iterrows():
                    if s['type'] == "None": continue
                    # Find closest node
                    nid = np.argmin([abs(n - self.cum_spans[int(s['id'])]) for n in nodes])
                    
                    # Lock Vertical (Y)
                    if s['type'] in ['Pin', 'Roller', 'Fixed']: 
                        free_d[2*nid] = False
                    
                    # Lock Rotation (Theta)
                    if s['type'] == 'Fixed': 
                        free_d[2*nid+1] = False

            # --- 5. Solve ---
            U = np.zeros(dof)
            if np.any(free_d):
                U[free_d] = solve(K[np.ix_(free_d, free_d)], F[free_d])
            
            # --- 6. Calculate Reactions ---
            R = K @ U - F

            reac_res = {}
            if not self.supports_df.empty:
                for _, s in self.supports_df.iterrows():
                    if s['type'] == "None": continue
                    nid = np.argmin([abs(n - self.cum_spans[int(s['id'])]) for n in nodes])
                    reac_res[int(s['id'])] = R[2*nid] 

            # --- 7. Post-Processing (High-Res Diagrams) ---
            # ใช้ Method of Sections เพื่อให้ได้กราฟที่ถูกต้องตาม Theory เป๊ะๆ
            res = []
            
            # สร้างจุด Plot จำนวนมาก + จุด Critical ทั้งหมด
            num_plot = 500
            base_points = np.linspace(0, nodes[-1], num_plot)
            # รวมจุด Nodes เข้าไปเพื่อให้กราฟหักมุมตรงจุดต่อพอดี
            all_points = sorted(list(set(np.concatenate((base_points, nodes)))))
            
            for x in all_points:
                v_sh, m_bm = 0.0, 0.0
                d_defl = 0.0
                
                # A. Sum Reactions (Left of x)
                for i, np_x in enumerate(nodes):
                    if np_x <= x + 1e-5:
                        v_sh += R[2*i]
                        m_bm += R[2*i]*(x - np_x) + R[2*i+1]
                
                # B. Sum Loads (Left of x)
                if not self.loads_df.empty:
                    for _, l in self.loads_df.iterrows():
                        gx = self.cum_spans[int(l['span_index'])] + float(l['x'])
                        
                        if l['type'] == 'P' and gx <= x + 1e-5:
                            v_sh -= l['mag']
                            m_bm -= l['mag']*(x - gx)
                            
                        elif l['type'] == 'M' and gx <= x + 1e-5:
                            m_bm -= l['mag'] # Clockwise Moment defined as positive input
                            
                        elif l['type'] == 'U':
                            # Start: gx, End: gx+dist
                            l_start = gx
                            l_end = gx + float(l['dist'])
                            
                            # ส่วนของ Load ที่อยู่ทางซ้ายของ x
                            if l_start < x:
                                overlap_end = min(x, l_end)
                                overlap_len = overlap_end - l_start
                                
                                if overlap_len > 0:
                                    force = l['mag'] * overlap_len
                                    # ระยะจาก centroid ของแรง ถึง x
                                    centroid_dist = x - (l_start + overlap_len/2)
                                    
                                    v_sh -= force
                                    m_bm -= force * centroid_dist
                
                # C. Deflection Interpolation
                for i in range(num_nodes - 1):
                    if nodes[i] <= x <= nodes[i+1] + 1e-5:
                        Le = nodes[i+1] - nodes[i]
                        if Le > 1e-6:
                            xi = (x - nodes[i]) / Le
                            Phi = self._get_phi(Le)
                            
                            # Shape Functions
                            N1 = (1/(1+Phi))*(1 - 3*xi**2 + 2*xi**3 + Phi*(1-xi))
                            N2 = (Le/(1+Phi))*(xi - 2*xi**2 + xi**3 + 0.5*Phi*(xi-xi**2))
                            N3 = (1/(1+Phi))*(3*xi**2 - 2*xi**3 + Phi*xi)
                            N4 = (Le/(1+Phi))*(-xi**2 + xi**3 - 0.5*Phi*(xi-xi**2))
                            
                            d_defl = N1*U[2*i] + N2*U[2*i+1] + N3*U[2*i+2] + N4*U[2*i+3]
                        break
                
                res.append({'x': x, 'shear': v_sh, 'moment': m_bm, 'deflection': d_defl})

            return pd.DataFrame(res), reac_res, {'status': 'OK'}
            
        except Exception as e:
            print(f"Solver Error: {e}")
            return pd.DataFrame(), {}, {"error": str(e)}
