import numpy as np

class TimoshenkoBeamSolver:
    def __init__(self, spans, supports, loads, b_mm, h_mm, fc):
        self.spans = spans
        self.supports = supports
        self.loads = loads
        self.b = b_mm / 1000  # m
        self.h = h_mm / 1000  # m
        self.fc = fc
        
        # Material Properties
        self.E = 4700 * np.sqrt(fc) * 1e6  # Pa (N/m2)
        self.nu = 0.2  # Poisson's ratio for concrete
        self.G = self.E / (2 * (1 + self.nu))
        self.I = (self.b * self.h**3) / 12
        self.A = self.b * self.h
        self.kappa = 5/6  # Shear correction factor for rectangular section
        self.sw_unit = self.A * 2400 * 9.81  # N/m (Self-weight)

    def solve(self):
        # 1. Mesh Generation (100 elements + load points)
        total_l = sum(self.spans)
        nodes = np.linspace(0, total_l, 101)
        cum_spans = [0] + list(np.cumsum(self.spans))
        for ld in self.loads:
            gx = cum_spans[ld['span_index']] + ld['x']
            nodes = np.append(nodes, gx)
        nodes = np.unique(np.sort(nodes))
        
        num_nodes = len(nodes)
        dof = 2 * num_nodes
        K = np.zeros((dof, dof))
        F = np.zeros(dof)

        # 2. Assembly with Timoshenko Stiffness Matrix
        for i in range(num_nodes - 1):
            le = nodes[i+1] - nodes[i]
            idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
            EI = self.E * self.I
            GAk = self.G * self.A * self.kappa
            
            # Timoshenko Shear Deformation Parameter (Phi)
            phi = (12 * EI) / (GAk * le**2)
            
            const = EI / (le**3 * (1 + phi))
            k_local = const * np.array([
                [12, 6*le, -12, 6*le],
                [6*le, (4+phi)*le**2, -6*le, (2-phi)*le**2],
                [-12, -6*le, 12, -6*le],
                [6*le, (2-phi)*le**2, -6*le, (4+phi)*le**2]
            ])
            K[np.ix_(idx, idx)] += k_local
            
            # Load Vector (Self-weight distributed to nodes)
            F[2*i] -= self.sw_unit * le / 2
            F[2*(i+1)] -= self.sw_unit * le / 2

        # 3. Apply External Point Loads
        for ld in self.loads:
            gx = cum_spans[ld['span_index']] + ld['x']
            nid = np.argmin(np.abs(nodes - gx))
            F[2*nid] -= ld['mag'] * 1000  # kN to N

        # 4. Boundary Conditions (Support vertical displacement = 0)
        active_dof = np.ones(dof, dtype=bool)
        for s_pos in self.supports:
            nid = np.argmin(np.abs(nodes - s_pos))
            active_dof[2*nid] = False 

        # 5. Solve for Nodal Displacements (u)
        u = np.zeros(dof)
        u[active_dof] = np.linalg.solve(K[np.ix_(active_dof, active_dof)], F[active_dof])

        # 6. Force Recovery (Max Moment & Shear)
        max_m = 0
        max_v = 0
        for i in range(num_nodes - 1):
            le = nodes[i+1] - nodes[i]
            idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
            u_el = u[idx]
            phi = (12 * self.E * self.I) / (self.G * self.A * self.kappa * le**2)
            
            # Moment from Timoshenko displacement field
            m_start = (self.E * self.I / (le * (1+phi))) * (-(u_el[1]*(4+phi) + u_el[3]*(2-phi)) + (6/le)*(u_el[2]-u_el[0]))
            v_el = (12 * self.E * self.I / (le**3 * (1+phi))) * (u_el[0] - u_el[2] + (le/2)*(u_el[1] + u_el[3]))
            
            max_m = max(max_m, abs(m_start))
            max_v = max(max_v, abs(v_el))

        return max_m / 1000, max_v / 1000  # kNm, kN
