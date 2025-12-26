import numpy as np
import pandas as pd

class BeamFiniteElement:
    def __init__(self, spans, supports, loads):
        self.spans = spans
        self.supports = supports
        self.loads = loads
        self.E = 200e9  # Young's Modulus (Dummy for rigid support analysis)
        self.I = 0.0001 # Moment of Inertia
        self.nodes = [0] + list(np.cumsum(spans))
        self.n_nodes = len(self.nodes)
        self.n_elements = len(spans)
        
    def solve(self):
        try:
            # 1. Mesh Generation (Simple 1 element per span)
            # To get better diagrams, we calculate internal forces at multiple points later
            # This FEM solves for Node Displacements first
            
            K_global = np.zeros((2 * self.n_nodes, 2 * self.n_nodes))
            F_global = np.zeros(2 * self.n_nodes)
            
            # 2. Build Stiffness Matrix
            for i in range(self.n_elements):
                L = self.spans[i]
                k = (self.E * self.I / L**3) * np.array([
                    [12, 6*L, -12, 6*L],
                    [6*L, 4*L**2, -6*L, 2*L**2],
                    [-12, -6*L, 12, -6*L],
                    [6*L, 2*L**2, -6*L, 4*L**2]
                ])
                
                # DOF mapping
                idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
                for r in range(4):
                    for c in range(4):
                        K_global[idx[r], idx[c]] += k[r, c]
                        
            # 3. Apply Loads (Fixed End Forces)
            for load in self.loads:
                span_idx = load['span_idx']
                w = load['w'] # Load magnitude
                L = self.spans[span_idx]
                
                # FEM Fixed End Actions for Uniform Load
                # Node i
                fem_v1 = w * L / 2
                fem_m1 = w * L**2 / 12
                # Node j
                fem_v2 = w * L / 2
                fem_m2 = -w * L**2 / 12
                
                # Add to global force vector (Substract FEM from external force)
                # DOFs
                idx = [2*span_idx, 2*span_idx+1, 2*span_idx+2, 2*span_idx+3]
                
                F_global[idx[0]] -= fem_v1 # Fy
                F_global[idx[1]] -= fem_m1 # Mz
                F_global[idx[2]] -= fem_v2
                F_global[idx[3]] -= fem_m2

            # 4. Apply Boundary Conditions
            freedofs = np.ones(2 * self.n_nodes, dtype=bool)
            
            for i, sup in enumerate(self.supports):
                dof_y = 2*i
                dof_m = 2*i + 1
                
                if sup == "Pin":
                    freedofs[dof_y] = False # Fix Y
                elif sup == "Roller":
                    freedofs[dof_y] = False # Fix Y
                elif sup == "Fixed":
                    freedofs[dof_y] = False # Fix Y
                    freedofs[dof_m] = False # Fix Rotation
            
            # 5. Solve
            K_reduced = K_global[freedofs, :][:, freedofs]
            F_reduced = F_global[freedofs]
            
            d_reduced = np.linalg.solve(K_reduced, F_reduced)
            
            displacements = np.zeros(2 * self.n_nodes)
            displacements[freedofs] = d_reduced
            
            self.displacements = displacements
            return True, "Analysis Complete"
            
        except Exception as e:
            return False, str(e)

    def get_internal_forces(self, num_points=50):
        # Calculate Shear and Moment diagrams using the solved displacements
        results = []
        
        for i in range(self.n_elements):
            L = self.spans[i]
            x_local = np.linspace(0, L, num_points)
            
            # Nodal Displacements for this element
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            u = self.displacements[idx] # [v1, theta1, v2, theta2]
            
            # Element Stiffness
            k = (self.E * self.I / L**3) * np.array([
                [12, 6*L, -12, 6*L],
                [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L],
                [6*L, 2*L**2, -6*L, 4*L**2]
            ])
            
            # Nodal Forces (f = k * u)
            f_local = k @ u
            
            # Superimpose with External Loads
            # Find load on this span
            w = 0
            for load in self.loads:
                if load['span_idx'] == i:
                    w = load['w']
            
            shear = []
            moment = []
            
            # Reaction at left node from stiffness
            Ry_start = f_local[0]
            M_start = f_local[1]
            
            # Calculate V(x) and M(x) along span
            # V(x) = Ry_start - w*x
            # M(x) = -M_start + Ry_start*x - w*x^2/2 (Sign convention: Tension bottom positive)
            
            # Adjusting sign convention for engineering plots
            # Standard: Shear Up +, Moment Sagging +
            
            for x in x_local:
                # Forces from Boundary Displacements
                # Using Shape Functions derivatives would be cleaner, but simple equilibrium works for uniform loads
                
                # Initial forces from nodes (End forces)
                # Correct approach: Calculate sections
                
                vx = Ry_start - w * x
                mx = -M_start + Ry_start * x - (w * x**2) / 2
                
                shear.append(vx)
                moment.append(mx)
                
            # Global X coordinates
            x_global = x_local + self.nodes[i]
            
            df_span = pd.DataFrame({
                'x': x_global,
                'shear': shear,
                'moment': moment
            })
            results.append(df_span)
            
        return pd.concat(results, ignore_index=True)
