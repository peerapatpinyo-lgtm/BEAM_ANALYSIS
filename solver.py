import numpy as np
import pandas as pd
from scipy.linalg import solve

class BeamSolver:
    def __init__(self, spans, supports_input, loads_input, E, I, A=None, G=None):
        self.spans = [float(s) for s in spans]
        self.E = float(E)
        self.I = float(I)
        # ใช้ 4 ตำแหน่งเพื่อความแม่นยำสูงสุดในการเปรียบเทียบ
        self.cum_spans = [round(x, 4) for x in ([0.0] + list(np.cumsum(self.spans)))]
        
        # กฎเหล็ก: จัดการ Load ก่อนเพื่อหาตำแหน่ง Node ทั้งหมดที่อาจเกิดขึ้น
        self.loads_df = self._sanitize_loads(loads_input)
        # จัดการ Support โดยใช้ตำแหน่งจริงของคานเป็นที่ตั้ง
        self.supports_df = self._sanitize_supports(supports_input)

    def _sanitize_supports(self, supports_input):
        sanitized = []
        # แปลง input ไม่ว่าจะเป็น list หรือ dataframe ให้เป็น list ของ dict
        data = supports_input.to_dict('records') if hasattr(supports_input, 'to_dict') else supports_input
        
        for s in data:
            # ดึงประเภท Support
            stype = str(s.get('type', s.get('Support Type', 'None')))
            if stype == "None": continue

            # วิธีใหม่: พยายามหาตำแหน่ง X จาก ID ก่อน ถ้าไม่ได้ให้ใช้ลำดับใน cum_spans
            raw_id = s.get('id', s.get('Node ID'))
            try:
                idx = int(raw_id)
                # ถ้ามาจาก Node ID (1, 2, 3) ปรับเป็น 0-based
                if 'Node ID' in s: idx -= 1
                
                if 0 <= idx < len(self.cum_spans):
                    sanitized.append({
                        'x': self.cum_spans[idx],
                        'type': stype
                    })
            except (TypeError, ValueError):
                continue
                
        return pd.DataFrame(sanitized)

    def _sanitize_loads(self, loads_input):
        if not loads_input: return pd.DataFrame(columns=['span_idx', 'type', 'mag', 'x', 'dist', 'case'])
        df = pd.DataFrame(loads_input)
        def get_global_x(row):
            s_idx = int(row.get('span_index', row.get('span_idx', 0)))
            local_x = float(row.get('x', 0))
            return round(self.cum_spans[s_idx] + local_x, 4)
        df['x'] = df.apply(get_global_x, axis=1)
        return df

    def solve(self):
        # 1. สร้าง Unique Nodes จาก Spans และ Loads
        pts = self.cum_spans.copy()
        for _, l in self.loads_df.iterrows():
            pts.append(l['x'])
            if l['type'] == 'U': pts.append(round(l['x'] + l['dist'], 4))
        
        # กำจัดจุดที่ซ้อนกันด้วยรัศมี 1 มิลลิเมตร
        nodes = []
        for p in sorted(pts):
            if not any(abs(p - n) < 1e-3 for n in nodes):
                nodes.append(p)
        
        num_nodes = len(nodes)
        dof = 2 * num_nodes
        K = np.zeros((dof, dof))
        F = np.zeros(dof)

        # 2. Global Stiffness Matrix K
        for i in range(num_nodes - 1):
            L = nodes[i+1] - nodes[i]
            if L > 1e-5:
                k_el = self._get_k(L)
                idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
                K[np.ix_(idx, idx)] += k_el

        # 3. Force Vector F (Point Loads)
        for _, l in self.loads_df.iterrows():
            nid = np.argmin([abs(n - l['x']) for n in nodes])
            if l['type'] == 'P': F[2*nid] -= l['mag']
            elif l['type'] == 'M': F[2*nid+1] += l['mag']
            elif l['type'] == 'U':
                s_glob, e_glob = l['x'], round(l['x'] + l['dist'], 4)
                for i in range(num_nodes - 1):
                    n1, n2 = nodes[i], nodes[i+1]
                    overlap = min(e_glob, n2) - max(s_glob, n1)
                    if overlap > 1e-5:
                        F[2*i] -= l['mag'] * overlap * 0.5
                        F[2*(i+1)] -= l['mag'] * overlap * 0.5

        # 4. Boundary Conditions (แก้ปัญหารูรั่วที่ทำให้ Reaction เป็น 0)
        free_dof = np.full(dof, True)
        for _, sup in self.supports_df.iterrows():
            # ค้นหาโหนดที่ตำแหน่ง X ตรงกันกับที่ระบุใน Support
            nid = np.argmin([abs(n - sup['x']) for n in nodes])
            stype = sup['type']
            # เช็คระยะห่างว่าต้องอยู่บนโหนดจริงๆ (รัศมี 1 มม.)
            if abs(nodes[nid] - sup['x']) < 1e-3:
                if stype in ['Pin', 'Roller', 'Fixed']:
                    free_dof[2*nid] = False
                if stype == 'Fixed':
                    free_dof[2*nid+1] = False

        # 5. แก้สมการ Displacement
        U = np.zeros(dof)
        if not np.all(free_dof):
            try:
                K_sub = K[np.ix_(free_dof, free_dof)]
                F_sub = F[free_dof]
                U[free_dof] = solve(K_sub, F_sub)
            except np.linalg.LinAlgError:
                # กรณี Matrix มีปัญหา (คานไม่เสถียร)
                return pd.DataFrame(), np.zeros(dof), {}

        # 6. หาค่า Reactions (R = K*U - F)
        R = K @ U - F

        # 7. Post-Processing (Internal Forces)
        results = []
        # ใช้จุดตรวจสอบที่ละเอียดขึ้นเพื่อความแม่นยำของ Equilibrium
        plot_x = np.unique(np.sort(np.concatenate([np.linspace(0, nodes[-1], 300), nodes])))
        
        for x in plot_x:
            V, M, defl = 0.0, 0.0, 0.0
            # ใช้สมการสมดุล (Method of Sections) ตัดจากซ้ายไปขวา
            for i, n_pos in enumerate(nodes):
                if n_pos <= x + 1e-5:
                    V += R[2*i]
                    M += R[2*i]*(x - n_pos) + R[2*i+1]
            
            for _, l in self.loads_df.iterrows():
                lx, lmag = l['x'], l['mag']
                if l['type'] == 'P' and lx <= x + 1e-5:
                    V -= lmag
                    M -= lmag*(x - lx)
                elif l['type'] == 'M' and lx <= x + 1e-5:
                    M -= lmag
                elif l['type'] == 'U' and lx < x:
                    dist = min(x, lx + l['dist']) - lx
                    if dist > 0:
                        V -= lmag * dist
                        M -= lmag * dist * (x - (lx + dist/2))
            
            # คำนวณ Deflection จากโหนดที่ครอบคลุมจุด x
            for i in range(num_nodes - 1):
                if nodes[i] <= x <= nodes[i+1] + 1e-5:
                    L_el = nodes[i+1] - nodes[i]
                    s = (x - nodes[i]) / L_el
                    idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
                    H = np.array([1-3*s**2+2*s**3, (x-nodes[i])*(1-s)**2, 3*s**2-2*s**3, (x-nodes[i])*(s**2-s)])
                    defl = np.dot(H, U[idx])
                    break
            
            results.append({'x': x, 'deflection': defl, 'shear': V, 'moment': M})

        df_res = pd.DataFrame(results)
        return df_res, R, self._create_summary(df_res)

    def _get_k(self, L):
        EI = self.E * self.I
        return (EI / L**3) * np.array([
            [12, 6*L, -12, 6*L], [6*L, 4*L**2, -6*L, 2*L**2],
            [-12, -6*L, 12, -6*L], [6*L, 2*L**2, -6*L, 4*L**2]
        ])

    def _create_summary(self, df):
        if df.empty: return {}
        return {
            'V_max': {'value': df['shear'].abs().max(), 'x': df.iloc[df['shear'].abs().idxmax()]['x']},
            'M_pos': {'value': df['moment'].max(), 'x': df.iloc[df['moment'].idxmax()]['x']},
            'M_neg': {'value': df['moment'].min(), 'x': df.iloc[df['moment'].idxmin()]['x']},
            'D_max': {'value': df['deflection'].abs().max(), 'x': df.iloc[df['deflection'].abs().idxmax()]['x']}
        }
