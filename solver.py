import numpy as np

class BeamSolver:
    def __init__(self, spans, supports, loads, b_m, h_m, fc):
        self.spans = spans  # List [5.0]
        self.supports = supports  # List [0, 5.0]
        self.loads = loads  # List of dicts
        self.b = b_m
        self.h = h_m
        self.E = 4700 * np.sqrt(fc) * 1e6  # Pa
        self.I = (b_m * h_m**3) / 12
        self.sw_unit = b_m * h_m * 2400 * 9.81  # N/m

    def solve(self):
        # 1. Mesh Generation
        cum_spans = [0] + list(np.cumsum(self.spans))
        total_l = sum(self.spans)
        nodes = np.linspace(0, total_l, 51)
        for ld in self.loads:
            gx = cum_spans[ld['span_index']] + ld['x']
            nodes = np.append(nodes, gx)
        nodes = np.unique(np.sort(nodes))
        
        num_nodes = len(nodes)
        dof = 2 * num_nodes
        K = np.zeros((dof, dof))
        F = np.zeros(dof)

        # 2. Stiffness & Self-weight
        for i in range(num_nodes - 1):
            L = nodes[i+1] - nodes[i]
            idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
            EI = self.E * self.I
            k_local = (EI / L**3) * np.array([
                [12, 6*L, -12, 6*L], [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L], [6*L, 2*L**2, -6*L, 4*L**2]
            ])
            K[np.ix_(idx, idx)] += k_local
            # Force from Self-weight
            F[2*i] -= self.sw_unit * L / 2
            F[2*i+1] -= self.sw_unit * L**2 / 12
            F[2*(i+1)] -= self.sw_unit * L / 2
            F[2*(i+1)+1] += self.sw_unit * L**2 / 12

        # 3. Add External Point Loads
        for ld in self.loads:
            gx = cum_spans[ld['span_index']] + ld['x']
            nid = np.argmin(np.abs(nodes - gx))
            F[2*nid] -= ld['mag'] * 1000

        # 4. Boundary Conditions
        active_dof = np.ones(dof, dtype=bool)
        for s_pos in self.supports:
            nid = np.argmin(np.abs(nodes - s_pos))
            active_dof[2*nid] = False 

        u = np.zeros(dof)
        u[active_dof] = np.linalg.solve(K[np.ix_(active_dof, active_dof)], F[active_dof])

        # 5. Get Internal Forces (Max Moment & Shear)
        moments = []
        shears = []
        for i in range(num_nodes - 1):
            L = nodes[i+1] - nodes[i]
            idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
            u_el = u[idx]
            # Moment calculation from nodal displacements
            m_start = (EI / L**2) * np.array([-6, -4*L, 6, -2*L]) @ u_el
            m_end = (EI / L**2) * np.array([6, 2*L, -6, 4*L]) @ u_el
            # Shear force calculation
            v_el = (EI / L**3) * np.array([12, 6*L, -12, 6*L]) @ u_el
            moments.extend([abs(m_start), abs(m_end)])
            shears.append(abs(v_el))

        return max(moments)/1000, max(shears)/1000 # Return kNm, kN
