
import numpy as np

class TimoshenkoBeamSolver:
    def __init__(self, L, b, h, fc, P_kN, pos_p):
        self.L = L
        self.b = b / 1000 # m
        self.h = h / 1000 # m
        self.fc = fc
        self.fy = 400 # MPa
        self.P = P_kN * 1000 # N
        self.pos_p = pos_p
        
        # คุณสมบัติวัสดุ
        self.E = 4700 * np.sqrt(fc) * 1e6 # Pa
        self.G = self.E / (2 * (1 + 0.2)) # Shear Modulus (Poisson's ratio approx 0.2)
        self.I = (self.b * self.h**3) / 12
        self.As = self.b * self.h # Area
        self.kappa = 5/6 # Shear correction factor for rectangular section
        self.sw_unit = self.As * 2400 * 9.81 # N/m

    def solve(self):
        # Mesh: 100 elements เพื่อความละเอียด
        nodes = np.linspace(0, self.L, 101)
        if not any(np.isclose(nodes, self.pos_p)):
            nodes = np.sort(np.append(nodes, self.pos_p))
        
        num_nodes = len(nodes)
        dof = 2 * num_nodes
        K = np.zeros((dof, dof))
        F = np.zeros(dof)

        for i in range(num_nodes - 1):
            le = nodes[i+1] - nodes[i]
            idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
            
            EI = self.E * self.I
            GAk = self.G * self.As * self.kappa
            Phi = (12 * EI) / (GAk * le**2) # Timoshenko's Phi (Shear deformation parameter)

            # Stiffness Matrix for Timoshenko Beam Element
            const = EI / (le**3 * (1 + Phi))
            k_local = const * np.array([
                [12, 6*le, -12, 6*le],
                [6*le, (4+Phi)*le**2, -6*le, (2-Phi)*le**2],
                [-12, -6*le, 12, -6*le],
                [6*le, (2-Phi)*le**2, -6*le, (4+Phi)*le**2]
            ])
            K[np.ix_(idx, idx)] += k_local

            # Force Vector (Self-weight)
            F[2*i] -= self.sw_unit * le / 2
            F[2*(i+1)] -= self.sw_unit * le / 2

        # Point Load
        p_node = np.argmin(np.abs(nodes - self.pos_p))
        F[2*p_node] -= self.P

        # Boundary Conditions
        active_dof = np.ones(dof, dtype=bool)
        active_dof[0] = False # Vertical at 0
        active_dof[2*(num_nodes-1)] = False # Vertical at L
        
        u = np.zeros(dof)
        u[active_dof] = np.linalg.solve(K[np.ix_(active_dof, active_dof)], F[active_dof])

        # คำนวณ Moment และ Shear
        moments = []
        for i in range(num_nodes - 1):
            le = nodes[i+1] - nodes[i]
            u_el = u[[2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]]
            # Moment in Timoshenko element
            m_start = (EI / (le * (1 + Phi))) * (-(u_el[1]*(4+Phi) + u_el[3]*(2-Phi)) + (6/le)*(u_el[2]-u_el[0]))
            moments.append(abs(m_start))

        return max(moments) / 1000 # kNm
