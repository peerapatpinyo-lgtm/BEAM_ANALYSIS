import numpy as np
import pandas as pd
from scipy.linalg import solve

class BeamSolver:
    def __init__(self, spans, supports_input, loads_input, E, b=0.3, h=0.5, I_custom=None):
        self.spans = [float(s) for s in spans]
        self.E = float(E)
        self.b = b
        self.h = h
        # ใช้ I_custom ถ้ามีค่าส่งมา ถ้าไม่มีให้คำนวณจาก b*h^3/12
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
        EI = self.E * self.I
        return (12 * EI) / (L**2 * self.G * self.As)

    def _get_k_timoshenko(self, L):
        """สร้าง Element Stiffness Matrix (Timoshenko)"""
        EI = self.E * self.I
        Phi = self._get_phi(L)
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
            # --- 1. Generate Nodes ---
            # สร้างจุด Nodes จาก Support และ Load locations
            pts = self.cum_spans.copy()
            if not self.loads_df.empty:
                for _, l in self.loads_df.iterrows():
                    gx = self.cum_spans[int(l['span_index'])] + float(l['x'])
                    pts.append(gx)
                    if l['type'] == 'U': 
                        pts.append(round(gx + float(l['dist']), 4))
            
            nodes = sorted(list(set([round(p, 4) for p in pts])))
            num_nodes = len(nodes)
            dof = 2 * num_nodes # 2 DOF per node (Vertical Y, Rotation Theta)
            
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
                    mag = float(l['mag']) # Value in Newtons or Nm
                    
                    if l['type'] == 'P': # Point Load
                        nid = np.argmin([abs(n - gx) for n in nodes])
                        F[2*nid] -= mag
                        
                    elif l['type'] == 'M': # Moment Load
                        nid = np.argmin([abs(n - gx) for n in nodes])
                        F[2*nid+1] += mag
                        
                    elif l['type'] == 'U': # UDL
                        dist = float(l['dist'])
                        for i in range(num_nodes - 1):
                            # หาช่วงที่ UDL ทับซ้อนกับ Element
                            overlap = min(gx+dist, nodes[i+1]) - max(gx, nodes[i])
                            if overlap > 1e-5:
                                w = mag # N/m
                                Le = nodes[i+1] - nodes[i]
                                # Equivalent Nodal Forces (Euler-Bernoulli approx for UDL)
                                F[2*i] -= (w * Le / 2)
                                F[2*i+1] -= (w * Le**2 / 12)
                                F[2*(i+1)] -= (w * Le / 2)
                                F[2*(i+1)+1] += (w * Le**2 / 12)

            # --- 4. Apply Boundary Conditions ---
            free_d = np.full(dof, True)
            if not self.supports_df.empty:
                for _, s in self.supports_df.iterrows():
                    if s['type'] == "None": continue
                    nid = np.argmin([abs(n - self.cum_spans[int(s['id'])]) for n in nodes])
                    
                    # Lock Vertical (Y)
                    if s['type'] in ['Pin', 'Roller', 'Fixed']: 
                        free_d[2*nid] = False
                    
                    # Lock Rotation (Theta)
                    if s['type'] == 'Fixed': 
                        free_d[2*nid+1] = False

            # --- 5. Solve for Displacements (U) ---
            U = np.zeros(dof)
            if np.any(free_d):
                U[free_d] = solve(K[np.ix_(free_d, free_d)], F[free_d])
            
            # --- 6. Calculate Reactions (R = K*U - F) ---
            R = K @ U - F

            # Format Reactions for app.py
            # Return as Dict {Support_Index: Force_Value_Newtons}
            reac_res = {}
            if not self.supports_df.empty:
                for _, s in self.supports_df.iterrows():
                    if s['type'] == "None": continue
                    nid = np.argmin([abs(n - self.cum_spans[int(s['id'])]) for n in nodes])
                    # ส่งค่าดิบ (Newtons) กลับไป ให้ app.py จัดการหาร 1000 เอง
                    reac_res[int(s['id'])] = R[2*nid] 

            # --- 7. Post-Processing (Shear / Moment Diagrams) ---
            # Using Method of Sections (Summing forces from left) for exact diagrams
            res = []
            plot_points = np.linspace(0, nodes[-1], 500)
            
            # รวมจุดสำคัญเข้าไปใน plot_points เพื่อกราฟที่คมชัด
            critical_points = nodes.copy()
            plot_points = sorted(list(set(np.concatenate((plot_points, critical_points)))))

            for x in plot_points:
                v_sh, m_bm, d_defl = 0, 0, 0
                
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
                            m_bm -= l['mag']
                        elif l['type'] == 'U' and gx < x:
                            d_overlap = min(x, gx + l['dist']) - gx
                            if d_overlap > 0:
                                force = l['mag'] * d_overlap
                                centroid_dist = x - (gx + d_overlap/2)
                                v_sh -= force
                                m_bm -= force * centroid_dist
                
                # C. Calculate Deflection (Interpolation using Shape Functions)
                for i in range(num_nodes - 1):
                    if nodes[i] <= x <= nodes[i+1] + 1e-5:
                        Le = nodes[i+1] - nodes[i]
                        xi = (x - nodes[i]) / Le
                        Phi = self._get_phi(Le)
                        
                        # Timoshenko Shape Functions
                        N1 = (1/(1+Phi))*(1 - 3*xi**2 + 2*xi**3 + Phi*(1-xi))
                        N2 = (Le/(1+Phi))*(xi - 2*xi**2 + xi**3 + 0.5*Phi*(xi-xi**2))
                        N3 = (1/(1+Phi))*(3*xi**2 - 2*xi**3 + Phi*xi)
                        N4 = (Le/(1+Phi))*(-xi**2 + xi**3 - 0.5*Phi*(xi-xi**2))
                        
                        d_defl = N1*U[2*i] + N2*U[2*i+1] + N3*U[2*i+2] + N4*U[2*i+3]
                        break
                
                res.append({'x': x, 'shear': v_sh, 'moment': m_bm, 'deflection': d_defl})

            # Check Equilibrium (Optional Info)
            eq_check = {'status': 'OK'} # Placeholder

            return pd.DataFrame(res), reac_res, eq_check
            
        except Exception as e:
            # Return empty if failed
            print(f"Solver Error: {e}")
            return pd.DataFrame(), {}, {"error": str(e)}
