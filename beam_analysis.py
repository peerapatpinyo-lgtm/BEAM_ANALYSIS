import numpy as np
import pandas as pd

class BeamFiniteElement:
    """ 
    Simple 1D Beam Finite Element Analysis (Matrix Stiffness Method)
    รองรับ Point Load และ Uniform Load
    """
    def __init__(self, spans, supports, loads):
        self.spans = spans
        self.supports = supports
        self.loads = loads
        self.E = 2e10  # Modulus of Elasticity (Est. Concrete)
        self.I = 0.005 # Moment of Inertia (Est.)

    def solve(self):
        # 1. Mesh Generation (Nodes & Elements)
        nodes = [0] + list(np.cumsum(self.spans))
        n_nodes = len(nodes)
        total_dof = n_nodes * 2 # 2 DOF per node (Vertical, Rotation)
        
        # 2. Global Stiffness Matrix (K) & Force Vector (F)
        K = np.zeros((total_dof, total_dof))
        F = np.zeros(total_dof)
        
        for i, L in enumerate(self.spans):
            # Element Stiffness Matrix
            k = (self.E * self.I / L**3) * np.array([
                [12, 6*L, -12, 6*L],
                [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L],
                [6*L, 2*L**2, -6*L, 4*L**2]
            ])
            # Assemble to Global K
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            for r in range(4):
                for c in range(4):
                    K[idx[r], idx[c]] += k[r, c]
            
            # Equivalent Nodal Forces from Loads (Fixed End Moments)
            # เฉพาะ Uniform Load ใน Span นี้
            span_loads = [l for l in self.loads if l['span_idx'] == i and l['type'] == 'Uniform']
            for load in span_loads:
                w = -load['total_w'] # ทิศลงเป็นลบ
                fem = np.array([w*L/2, w*L**2/12, w*L/2, -w*L**2/12])
                F[idx] += fem

        # 3. Apply Point Loads directly to Nodes (Simplified)
        # (Note: In strict FEM, point loads within spans need node splitting or equivalent forces. 
        # Here we approximate by adding equivalent FEM for point loads is too complex for this snippet.
        # So we use a hybrid approach: Solve reactions for supports, then standard statics for internal forces.)
        
        # --- Boundary Conditions ---
        fixed_dofs = []
        for i, sup in enumerate(self.supports):
            dof_y = 2*i
            dof_m = 2*i + 1
            if sup in ["Pin", "Roller", "Fix"]:
                fixed_dofs.append(dof_y) # Fix vertical
            if sup == "Fix":
                fixed_dofs.append(dof_m) # Fix rotation

        # Reduce Matrix
        free_dofs = [i for i in range(total_dof) if i not in fixed_dofs]
        if not free_dofs: return False, "Over-constrained"
        
        K_red = K[np.ix_(free_dofs, free_dofs)]
        F_red = F[free_dofs]
        
        # Solve Displacements (u = K^-1 * F)
        try:
            u_red = np.linalg.solve(K_red, F_red)
        except np.linalg.LinAlgError:
            return False, "Unstable Structure"

        u_full = np.zeros(total_dof)
        u_full[free_dofs] = u_red
        
        # Reaction Forces (R = K*u - F_applied)
        self.R = K @ u_full - F # Total nodal forces (Reactions)
        self.displacements = u_full
        return True, None

    def get_internal_forces(self, n_points=200):
        """ ใช้วิธี Statics ไล่ตัด Section จากซ้ายไปขวา โดยใช้ Reaction ที่คำนวณได้ """
        total_len = sum(self.spans)
        cum_spans = [0] + list(np.cumsum(self.spans))
        x_eval = np.linspace(0, total_len, n_points)
        
        shears = []
        moments = []
        
        for x in x_eval:
            V = 0
            M = 0
            
            # 1. ผลจาก Reactions (ด้านซ้ายของ x)
            for i, r_val in enumerate(self.R[::2]): # เอาเฉพาะแรงแนวดิ่ง (indices 0, 2, 4...)
                pos = cum_spans[i]
                if pos <= x + 1e-5: # อยู่ซ้ายมือ
                    V += r_val
                    M += r_val * (x - pos)
            
            # 2. ผลจาก Loads (ด้านซ้ายของ x)
            for load in self.loads:
                # แปลงตำแหน่ง Load ให้อยู่ใน Global Global Coordinate
                start_x = cum_spans[load['span_idx']]
                
                if load['type'] == 'Uniform':
                    w = load['total_w'] # Positive value entered
                    end_x = start_x + self.spans[load['span_idx']]
                    
                    # ช่วงที่ Load กระทำที่อยู่ซ้ายมือของ x
                    lx_start = start_x
                    lx_end = min(x, end_x)
                    
                    if lx_end > lx_start:
                        load_len = lx_end - lx_start
                        force = w * load_len
                        V -= force # Load ลง ทิศลบ
                        # Centroid ของ Load ก้อนนี้ถึง x
                        dist = x - (lx_start + load_len/2)
                        M -= force * dist
                        
                elif load['type'] == 'Point':
                    px = start_x + load['pos']
                    P = load['total_w']
                    if px <= x + 1e-5:
                        V -= P
                        M -= P * (x - px)
            
            shears.append(V)
            moments.append(M)
            
        return pd.DataFrame({'x': x_eval, 'shear': shears, 'moment': moments})
