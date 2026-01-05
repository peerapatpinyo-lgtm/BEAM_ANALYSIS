# solver.py (Fixed Equilibrium & Node Mapping)
import numpy as np
import pandas as pd
from scipy.linalg import solve

class BeamSolver:
    def __init__(self, spans, supports_df, loads_input, E, I, A=None, G=None):
        self.spans = [float(s) for s in spans]
        self.E = float(E)
        self.I = float(I)
        self.cum_spans = [round(x, 6) for x in ([0.0] + list(np.cumsum(self.spans)))] # ปัดเศษเพื่อความแม่นยำ
        
        self.loads_df = self._sanitize_loads(loads_input)
        self.supports_df = self._sanitize_supports(supports_df)

    # ... [_sanitize_loads และ _sanitize_supports คงเดิมตามข้อตกลง] ...

    def solve(self):
        # 1. Critical Points (เพิ่ม Tolerance ในการสร้าง Nodes)
        points = set([round(x, 6) for x in self.cum_spans])
        for _, l in self.loads_df.iterrows():
            points.add(round(l['x'], 6))
            if l['type'] == 'U': points.add(round(l['x'] + l['dist'], 6))
        for _, s in self.supports_df.iterrows():
            points.add(round(s['x'], 6))
            
        nodes = sorted(list(points))
        num_nodes = len(nodes)
        dof = 2 * num_nodes
        
        # 2. Stiffness K
        K = np.zeros((dof, dof))
        elements = []
        for i in range(num_nodes - 1):
            x1, x2 = nodes[i], nodes[i+1]
            L = x2 - x1
            elements.append({'n1': i, 'n2': i+1, 'L': L})
            if L > 1e-7: # ปรับ Tolerance เล็กน้อย
                k_el = self._get_k(L)
                idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
                for r in range(4):
                    for c in range(4):
                        K[idx[r], idx[c]] += k_el[r, c]

        # 3. Force F
        F = np.zeros(dof)
        for _, load in self.loads_df.iterrows():
            nid = self._find_nearest_node(nodes, load['x']) # แก้ไขฟังก์ชันนี้ด้านล่าง
            
            if load['type'] == 'P' and nid != -1:
                F[2*nid] -= load['mag']
            elif load['type'] == 'M' and nid != -1:
                F[2*nid+1] += load['mag']
            elif load['type'] == 'U':
                # ... [ส่วน Gauss Quadrature สำหรับ UDL คงเดิม] ...
                start, dist, mag = load['x'], load['dist'], load['mag']
                end = start + dist
                for elem in elements:
                    ex1, ex2 = nodes[elem['n1']], nodes[elem['n2']]
                    if ex2 <= start + 1e-7 or ex1 >= end - 1e-7: continue
                    ov_s = max(start, ex1)
                    ov_e = min(end, ex2)
                    len_load = ov_e - ov_s
                    if len_load <= 0: continue
                    mid = (ov_s + ov_e)/2
                    for gp in [-0.57735, 0.57735]:
                        xi = mid + (len_load/2)*gp
                        s = (xi - ex1) / elem['L']
                        N_vec = np.array([1-3*s**2+2*s**3, (xi-ex1)*(1-s)**2, 3*s**2-2*s**3, (xi-ex1)*(s**2-s)])
                        idx_el = [2*elem['n1'], 2*elem['n1']+1, 2*elem['n2'], 2*elem['n2']+1]
                        F[idx_el] -= N_vec * mag * (len_load/2)

        # 4. Supports
        free_dof = np.full(dof, True)
        for _, sup in self.supports_df.iterrows():
            nid = self._find_nearest_node(nodes, sup['x'])
            if nid != -1:
                stype = sup.get('type', 'Pin')
                if stype in ['Pin', 'Roller', 'Fixed']: free_dof[2*nid] = False
                if stype == 'Fixed': free_dof[2*nid+1] = False

        # 5. Solve
        U = np.zeros(dof)
        if np.sum(free_dof) < dof:
            try:
                U[free_dof] = solve(K[np.ix_(free_dof, free_dof)], F[free_dof])
            except: 
                return pd.DataFrame(), np.zeros(dof), {'error': 'Singular Matrix'}

        # 6. Reactions
        R = K @ U - F

        # 7. Post-Processing (จุดที่แก้เพื่อให้ Equilibrium ครบ)
        base_x = np.linspace(0, nodes[-1], 400)
        critical_x = []
        for n in nodes: critical_x.extend([n - 1e-7, n, n + 1e-7]) # ลดระยะ Offset ให้เล็กลง
        all_x = np.unique(np.sort(np.concatenate([base_x, critical_x])))
        all_x = all_x[(all_x >= 0) & (all_x <= nodes[-1])]
        
        results = []
        for x in all_x:
            defl, V, M = 0.0, 0.0, 0.0
            # Deflection calculation ... [คงเดิม]
            for elem in elements:
                x1, x2 = nodes[elem['n1']], nodes[elem['n2']]
                if x1 <= x <= x2 + 1e-7:
                    s = (x - x1) / elem['L']
                    idx = [2*elem['n1'], 2*elem['n1']+1, 2*elem['n2'], 2*elem['n2']+1]
                    H = np.array([1-3*s**2+2*s**3, (x-x1)*(1-s)**2, 3*s**2-2*s**3, (x-x1)*(s**2-s)])
                    defl = np.dot(H, U[idx])
                    break
            
            # --- FIXED STATICS INTEGRATION ---
            # ใช้ค่า 1e-6 เพื่อให้ครอบคลุมแรงปฏิกิริยาที่ตำแหน่ง x พอดี
            for i, nx in enumerate(nodes):
                if nx <= x + 1e-6: 
                    V += R[2*i]
                    M += R[2*i]*(x-nx) + R[2*i+1]
            
            for _, l in self.loads_df.iterrows():
                lx, mag = l['x'], l['mag']
                if l['type'] == 'P':
                    if lx <= x + 1e-6: 
                        V -= mag
                        M -= mag * (x - lx)
                elif l['type'] == 'M':
                    if lx <= x + 1e-6: M -= mag 
                elif l['type'] == 'U':
                    start, end = lx, lx + l['dist']
                    if start < x + 1e-6:
                        cov = min(x, end) - start
                        if cov > 0:
                            force = mag * cov
                            arm = x - (start + cov/2)
                            V -= force
                            M -= force * arm
            results.append({'x': x, 'deflection': defl, 'shear': V, 'moment': M})
            
        return pd.DataFrame(results), R, self._create_summary(pd.DataFrame(results))

    def _find_nearest_node(self, nodes, val):
        arr = np.array(nodes)
        diff = np.abs(arr - val)
        idx = diff.argmin()
        if diff[idx] < 1e-5: # เพิ่มความกว้างของ Tolerance เล็กน้อย
            return idx
        return -1

    def _create_summary(self, df_res):
        # แยกออกมาเป็นฟังก์ชันเพื่อให้ code หลักสะอาดขึ้น
        summary = {}
        if not df_res.empty:
            summary['V_max'] = {'value': df_res['shear'].abs().max(), 'x': df_res.loc[df_res['shear'].abs().idxmax(), 'x']}
            summary['M_pos'] = {'value': df_res['moment'].max(), 'x': df_res.loc[df_res['moment'].idxmax(), 'x']}
            summary['M_neg'] = {'value': df_res['moment'].min(), 'x': df_res.loc[df_res['moment'].idxmin(), 'x']}
            summary['D_max'] = {'value': df_res['deflection'].abs().max(), 'x': df_res.loc[df_res['deflection'].abs().idxmax(), 'x']}
        return summary

    def _get_k(self, L):
        # ... [คงเดิม] ...
        c = self.E * self.I / L**3
        return c * np.array([[12, 6*L, -12, 6*L], [6*L, 4*L**2, -6*L, 2*L**2], [-12, -6*L, 12, -6*L], [6*L, 2*L**2, -6*L, 4*L**2]])
