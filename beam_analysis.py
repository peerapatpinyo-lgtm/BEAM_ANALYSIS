import numpy as np
import pandas as pd

def run_beam_analysis(spans, supports, loads):
    """
    Main entry point function to connect with Streamlit App.
    Returns exactly what the app expects: (vis_spans, vis_supports)
    """
    model = BeamFiniteElement(spans, supports, loads)
    success, message = model.solve()
    
    if not success:
        raise ValueError(f"Analysis Failed: {message}")

    # Prepare DataFrames for Visualization
    vis_spans = model.get_internal_forces()
    vis_supports = model.get_reaction_forces()
    
    return vis_spans, vis_supports

class BeamFiniteElement:
    def __init__(self, spans, supports, loads):
        self.spans = spans
        self.supports = supports
        self.loads = loads
        self.E = 200e9  # Young's Modulus
        self.I = 0.0001 # Moment of Inertia
        self.nodes = [0] + list(np.cumsum(spans))
        self.n_nodes = len(self.nodes)
        self.n_elements = len(spans)
        self.displacements = None
        self.reactions = None

    def solve(self):
        try:
            # --- 1. Setup Global Matrices ---
            n_dof = 2 * self.n_nodes
            K_global = np.zeros((n_dof, n_dof))
            F_global = np.zeros(n_dof) # External Nodal Forces - FEM

            # --- 2. Build Stiffness Matrix ---
            for i in range(self.n_elements):
                L = self.spans[i]
                k_ele = self._get_element_stiffness(L)
                
                # DOF mapping
                idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
                for r in range(4):
                    for c in range(4):
                        K_global[idx[r], idx[c]] += k_ele[r, c]

            # --- 3. Apply Loads (Fixed End Actions) ---
            # We subtract FEM from the global force vector to solve for displacements
            for load in self.loads:
                span_idx = load['span_idx']
                w = load['w']
                L = self.spans[span_idx]

                # Calculate FEM (Fixed End Moments/Forces)
                # Convention: Up = +, Counter-Clockwise = +
                fem_v1 = w * L / 2
                fem_m1 = w * L**2 / 12
                fem_v2 = w * L / 2
                fem_m2 = -w * L**2 / 12
                
                # Global DOFs for this element
                idx = [2*span_idx, 2*span_idx+1, 2*span_idx+2, 2*span_idx+3]
                
                # Subtract FEM from External Forces (F_node = F_ext - F_fem)
                F_global[idx[0]] -= fem_v1
                F_global[idx[1]] -= fem_m1
                F_global[idx[2]] -= fem_v2
                F_global[idx[3]] -= fem_m2

            # --- 4. Apply Boundary Conditions ---
            freedofs = np.ones(n_dof, dtype=bool)
            
            for i, sup in enumerate(self.supports):
                dof_y = 2*i
                dof_m = 2*i + 1
                
                if sup == "Pin":
                    freedofs[dof_y] = False
                elif sup == "Roller":
                    freedofs[dof_y] = False
                elif sup == "Fixed":
                    freedofs[dof_y] = False
                    freedofs[dof_m] = False

            # --- 5. Solve for Displacements ---
            K_reduced = K_global[freedofs, :][:, freedofs]
            F_reduced = F_global[freedofs]
            
            d_reduced = np.linalg.solve(K_reduced, F_reduced)
            
            self.displacements = np.zeros(n_dof)
            self.displacements[freedofs] = d_reduced
            
            # --- 6. Calculate Global Reactions (F = K*d - F_ext_actual) ---
            # Re-calculate full F = K*d. The difference between this and applied loads is the reaction.
            # Simplified: For this app, we will calculate reactions element-by-element later.
            
            return True, "Analysis Complete"

        except Exception as e:
            return False, str(e)

    def _get_element_stiffness(self, L):
        """Helper to get element stiffness matrix"""
        return (self.E * self.I / L**3) * np.array([
            [12, 6*L, -12, 6*L],
            [6*L, 4*L**2, -6*L, 2*L**2],
            [-12, -6*L, 12, -6*L],
            [6*L, 2*L**2, -6*L, 4*L**2]
        ])

    def get_internal_forces(self, num_points=100):
        """Calculates V(x), M(x), Deflection(x) for plotting"""
        results = []
        
        for i in range(self.n_elements):
            L = self.spans[i]
            x_local = np.linspace(0, L, num_points)
            
            # Nodal Displacements
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            u = self.displacements[idx] # [v1, theta1, v2, theta2]
            
            # 1. Forces from Displacements (Stiffness * u)
            k = self._get_element_stiffness(L)
            f_disp = k @ u # [Fy1, M1, Fy2, M2] due to deformation
            
            # 2. Forces from Fixed End Actions (FEM)
            # Find load on this span
            w = 0
            for load in self.loads:
                if load['span_idx'] == i:
                    w = load['w']
            
            # FEM vector for this element
            fem_vector = np.array([
                w * L / 2,       # Fy1
                w * L**2 / 12,   # M1
                w * L / 2,       # Fy2
                -w * L**2 / 12   # M2
            ])
            
            # Total Nodal Forces = Stiffness_Forces + FEM_Forces
            # This gives us the correct starting shear and moment at the left node
            f_total = f_disp + fem_vector
            
            # Extract start forces (Left Node)
            # Note: FEM standard convention (CCW positive), Beam convention (Sagging positive)
            V_start = f_total[0]  # Shear
            M_start = f_total[1]  # Moment (Counter-Clockwise is positive)
            
            # Calculate arrays along the span
            # V(x) = V_start - w*x
            # M(x) = -M_start + V_start*x - w*x^2/2  (Note: -M_start because M_start is reaction on beam)
            
            shear = V_start - w * x_local
            moment = -M_start + V_start * x_local - (w * x_local**2) / 2
            
            # Deflection (Shape function method)
            # N1 = 1 - 3x^2/L^2 + 2x^3/L^3 ... etc
            xi = x_local / L
            N1 = 1 - 3*xi**2 + 2*xi**3
            N2 = L * (xi - 2*xi**2 + xi**3)
            N3 = 3*xi**2 - 2*xi**3
            N4 = L * (-xi**2 + xi**3)
            
            # Deflection due to nodal displacements
            v_disp = N1*u[0] + N2*u[1] + N3*u[2] + N4*u[3]
            
            # Add simple deflection due to load (Superposition for simply supported beam approx)
            # For exact viz, usually Shape Function interpolation is enough if mesh is fine,
            # but for 1 element per span, we visualize just the nodal interpolation.
            deflection = v_disp 

            # Create DataFrame
            x_global = x_local + self.nodes[i]
            df_span = pd.DataFrame({
                'x': x_global,
                'shear': shear,
                'moment': moment,
                'deflection': deflection,
                'span_id': i
            })
            results.append(df_span)
            
        return pd.concat(results, ignore_index=True)

    def get_reaction_forces(self):
        """Calculates reactions at supported nodes"""
        reactions = []
        
        # We need to sum up forces from all elements connected to a node
        # Initialize global reaction vector
        R_global = np.zeros(2 * self.n_nodes)
        
        for i in range(self.n_elements):
            L = self.spans[i]
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            u = self.displacements[idx]
            
            # Stiffness part
            k = self._get_element_stiffness(L)
            f_ele = k @ u
            
            # Load part (FEM)
            w = 0
            for load in self.loads:
                if load['span_idx'] == i:
                    w = load['w']
            
            fem_vector = np.array([
                w*L/2, w*L**2/12, w*L/2, -w*L**2/12
            ])
            
            f_total = f_ele + fem_vector
            
            # Add to global nodes
            R_global[idx[0]] += f_total[0] # Fy node i
            R_global[idx[1]] += f_total[1] # Mz node i
            R_global[idx[2]] += f_total[2] # Fy node j
            R_global[idx[3]] += f_total[3] # Mz node j

        # Create Output Data
        for i, sup in enumerate(self.supports):
            if sup != "None":
                reactions.append({
                    'node_id': i,
                    'x': self.nodes[i],
                    'type': sup,
                    'Ry': R_global[2*i],
                    'Mz': R_global[2*i+1]
                })
                
        return pd.DataFrame(reactions)
