import numpy as np
import pandas as pd

class BeamAnalysisEngine:
    def __init__(self, spans, supports, loads, E=2e6, I=1e-3):
        self.spans = spans
        self.supports = supports
        self.loads = loads
        self.E = E
        self.I = I
        self.nodes = [0] + list(np.cumsum(spans))
        self.n_nodes = len(self.nodes)
        self.dof = 2 * self.n_nodes 
        
    def solve(self):
        # 1. Global Stiffness Matrix
        K_global = np.zeros((self.dof, self.dof))
        
        for i, L in enumerate(self.spans):
            k = self.E * self.I / L**3
            K_elem = np.array([
                [12*k,      6*k*L,    -12*k,     6*k*L],
                [6*k*L,     4*k*L**2, -6*k*L,    2*k*L**2],
                [-12*k,    -6*k*L,     12*k,    -6*k*L],
                [6*k*L,     2*k*L**2, -6*k*L,    4*k*L**2]
            ])
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            for r in range(4):
                for c in range(4):
                    K_global[idx[r], idx[c]] += K_elem[r, c]

        # 2. Force Vector (FEM)
        F_global = np.zeros(self.dof)
        
        for l in self.loads:
            span_idx = l['span_idx']
            L = self.spans[span_idx]
            idx_base = 2*span_idx
            
            if l['type'] == 'U':
                w = l['w']
                fem = np.array([-w*L/2, -w*L**2/12, -w*L/2, w*L**2/12])
                F_global[idx_base:idx_base+4] += fem
                
            elif l['type'] == 'P':
                P = l['P']; a = l['x']; b = L - a
                r1 = (P*b**2*(3*a+b))/L**3
                m1 = (P*a*b**2)/L**2
                r2 = (P*a**2*(a+3*b))/L**3
                m2 = -(P*a**2*b)/L**2
                F_global[idx_base:idx_base+4] += np.array([-r1, -m1, -r2, -m2])

        # 3. Apply Boundary Conditions
        constrained_dof = []
        for i, row in self.supports.iterrows():
            stype = row['type']
            node_idx = i
            if stype in ["Pin", "Roller"]:
                constrained_dof.append(2*node_idx) 
            elif stype == "Fixed":
                constrained_dof.append(2*node_idx)
                constrained_dof.append(2*node_idx+1)
        
        # 4. Solve
        free_dof = [i for i in range(self.dof) if i not in constrained_dof]
        K_reduced = K_global[np.ix_(free_dof, free_dof)]
        F_reduced = F_global[free_dof]
        
        try:
            D_free = np.linalg.solve(K_reduced, F_reduced)
        except np.linalg.LinAlgError:
            return None, None

        D_total = np.zeros(self.dof)
        D_total[free_dof] = D_free
        
        # 5. Post-Processing & Reactions
        R_calc = np.dot(K_global, D_total) 
        reactions = R_calc - F_global
        
        # Internal Forces Integration
        curr_x = 0
        all_x, all_v, all_m = [], [], []
        
        for i, L in enumerate(self.spans):
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            d_local = D_total[idx]
            
            # Stiffness Forces
            k = self.E * self.I / L**3
            K_el = np.array([
                [12*k,      6*k*L,    -12*k,     6*k*L],
                [6*k*L,     4*k*L**2, -6*k*L,    2*k*L**2],
                [-12*k,    -6*k*L,     12*k,    -6*k*L],
                [6*k*L,     2*k*L**2, -6*k*L,    4*k*L**2]
            ])
            f_member = np.dot(K_el, d_local)
            
            # FEM Back-calculation
            fem_vec = np.zeros(4)
            span_loads = [l for l in self.loads if l['span_idx'] == i]
            
            # Critical Points for this span (Start, End, Point Loads)
            critical_pts = [0, L]
            
            for l in span_loads:
                if l['type'] == 'U':
                    w = l['w']
                    fem_vec += np.array([w*L/2, w*L**2/12, w*L/2, -w*L**2/12])
                elif l['type'] == 'P':
                    P = l['P']; a = l['x']; b = L - a
                    r1 = (P*b**2*(3*a+b))/L**3
                    m1 = (P*a*b**2)/L**2
                    r2 = (P*a**2*(a+3*b))/L**3
                    m2 = -(P*a**2*b)/L**2
                    fem_vec += np.array([r1, m1, r2, m2])
                    # Add point load location to critical points
                    critical_pts.append(l['x'])
                    critical_pts.append(l['x'] - 0.001) 
                    critical_pts.append(l['x'] + 0.001) 
            
            # Sort and create evaluation points
            critical_pts = sorted(list(set(critical_pts)))
            fine_pts = np.linspace(0, L, 50)
            x_span = sorted(list(set(critical_pts + list(fine_pts))))
            
            f_final = f_member + fem_vec
            V_start, M_start = f_final[0], f_final[1]
            
            for x in x_span:
                if x < 0 or x > L: continue
                v_x = V_start
                m_x = -M_start + V_start * x 
                
                for l in span_loads:
                    if l['type'] == 'U':
                        if x > 0:
                            w = l['w']
                            v_x -= w * x
                            m_x -= w * x**2 / 2
                    elif l['type'] == 'P':
                        if x >= l['x']: 
                            v_x -= l['P']
                            m_x -= l['P'] * (x - l['x'])
                
                all_x.append(curr_x + x)
                all_v.append(v_x)
                all_m.append(m_x)
                
            curr_x += L
            
        return pd.DataFrame({'x': all_x, 'shear': all_v, 'moment': all_m}), reactions
