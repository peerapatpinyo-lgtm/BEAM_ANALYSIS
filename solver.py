import numpy as np

class TimoshenkoBeamSolver:
    def __init__(self, spans, supports, loads, b_mm, h_mm, fc):
        self.spans = spans
        self.supports = supports
        self.loads = loads
        self.b = b_mm / 1000 # m
        self.h = h_mm / 1000 # m
        # Material Properties
        self.E = 4700 * np.sqrt(fc) * 1e6 # Pa
        self.G = self.E / (2 * (1 + 0.2)) # Shear Modulus (nu=0.2)
        self.I = (self.b * self.h**3) / 12
        self.As = self.b * self.h
        self.kappa = 5/6 # Shear correction factor for rectangular
        self.sw_unit = self.As * 2400 * 9.81 # N/m

    def solve(self):
        total_l = sum(self.spans)
        # Mesh: แบ่ง 100 ช่วงเพื่อความละเอียดของ Timoshenko
        nodes = np.linspace(0, total_l, 101)
        # แทรก Node ตรงจุดลงแรง
        cum_spans = [0] + list(np.cumsum(self.spans))
        for ld in self.loads:
            gx = cum_spans[ld['span_index']] + ld['x']
            nodes = np.append(nodes, gx)
        nodes = np.unique(np.sort(nodes))
        
        num_nodes = len(nodes)
        dof = 2 * num_nodes
        K = np.zeros((dof, dof))
        F = np.zeros(dof)

        for i in range(num_nodes - 1):
            le = nodes[i+1] - nodes[i]
            idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
            EI = self.E * self.I
            GAk = self.G * self.As * self.kappa
            
            # --- Timoshenko Phi (หัวใจสำคัญของคานสั้น) ---
            Phi = (12 * EI) / (GAk * le**2)
            
            const = EI / (le**3 * (1 + Phi))
            k_local = const * np.array([
                [12, 6*le, -12, 6*le],
                [6*le, (4+Phi)*le**2, -6*le, (2-Phi)*le**2],
                [-12, -6*le, 12, -6*le],
                [6*le, (2-Phi)*le**2, -6*le, (4+Phi)*le**2]
            ])
            K[np.ix_(idx, idx)] += k_local
            
            # Self-weight
            F[2*i] -= self.sw_unit * le / 2
            F[2*(i+1)] -= self.sw_unit * le / 2

        # External Loads
        for ld in self.loads:
            gx = cum_spans[ld['span_index']] + ld['x']
            nid = np.argmin(np.abs(nodes - gx))
            F[2*nid] -= ld['mag'] * 1000

        # Boundary Conditions (Simple Supports)
        active_dof = np.ones(dof, dtype=bool)
        for s_pos in self.supports:
            nid = np.argmin(np.abs(nodes - s_pos))
            active_dof[2*nid] = False 

        u = np.zeros(dof)
        u[active_dof] = np.linalg.solve(K[np.ix_(active_dof, active_dof)], F[active_dof])

        # Internal Forces Recovery
        max_m = 0
        for i in range(num_nodes - 1):
            le = nodes[i+1] - nodes[i]
            u_el = u[[2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]]
            Phi = (12 * self.E * self.I) / (self.G * self.As * self.kappa * le**2)
            # Timoshenko Moment Formula
            m_el = (self.E * self.I / (le * (1+Phi))) * (-(u_el[1]*(4+Phi) + u_el[3]*(2-Phi)) + (6/le)*(u_el[2]-u_el[0]))
            max_m = max(max_m, abs(m_el))

        return max_m / 1000 # kNm
