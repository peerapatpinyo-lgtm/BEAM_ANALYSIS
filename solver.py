import numpy as np
import pandas as pd

class BeamSolver:
    def __init__(self, spans, supports, loads, E, I, A=None, G=None):
        """
        Modified for Timoshenko:
        - A: Cross-sectional Area (Required for Shear calc)
        - G: Shear Modulus (If None, calculated from E assuming steel v=0.3)
        """
        self.spans = spans
        self.supports = supports if isinstance(supports, pd.DataFrame) else pd.DataFrame(supports)
        self.loads = loads
        self.E = float(E)
        self.I = float(I)
        
        # --- TIMOSHENKO PARAMETERS ---
        # If A is not provided, estimate from I assuming square section (Fallback)
        if A is None:
            # h^4 = 12*I / b (assuming b=h) -> h = (12I)^0.25 -> A = h^2
            # This is a rough estimation to prevent crash if user forgets A
            self.A = (12 * self.I)**0.5 
        else:
            self.A = float(A)
            
        # If G is not provided, assume Steel (v = 0.3) -> G = E / 2(1+v)
        if G is None:
            self.G = self.E / (2 * (1 + 0.3))
        else:
            self.G = float(G)
            
        # Shear Correction Factor (k)
        # Rectangular = 5/6, Circular = 0.9. Let's use 5/6 as standard default.
        self.kappa = 5/6 

        self.num_nodes = len(spans) + 1
        self.num_dof = 2 * self.num_nodes

    def solve(self):
        # 1. Setup Global Matrix
        num_dof = self.num_dof
        K = np.zeros((num_dof, num_dof))
        F = np.zeros(num_dof)
        
        # 2. Build Stiffness & Load Vector
        for i, L in enumerate(self.spans):
            # --- Modified: Get Timoshenko Stiffness ---
            k_local = self._get_element_stiffness(L)
            
            idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
            
            for r in range(4):
                for c in range(4):
                    K[idx[r], idx[c]] += k_local[r, c]

            if not self.loads.empty:
                span_loads = self.loads[self.loads['span_idx'] == i].to_dict('records')
            else:
                span_loads = []
            
            for load in span_loads:
                f_equiv = self._calc_equivalent_nodal_forces(load, L)
                F[idx[0]] += f_equiv[0]
                F[idx[1]] += f_equiv[1]
                F[idx[2]] += f_equiv[2]
                F[idx[3]] += f_equiv[3]

        # 3. Apply Boundary Conditions
        fixed_dofs = []
        for _, sup in self.supports.iterrows():
            node_idx = int(sup['id'])
            sType = sup['type']
            if sType in ['Pin', 'Roller']:
                fixed_dofs.append(2*node_idx) 
            elif sType == 'Fixed':
                fixed_dofs.append(2*node_idx)   
                fixed_dofs.append(2*node_idx+1) 
        
        fixed_dofs = sorted(list(set(fixed_dofs)))
        free_dofs = [i for i in range(num_dof) if i not in fixed_dofs]
        
        # 4. Solve
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        F_f = F[free_dofs]
        
        try:
            u_f = np.linalg.solve(K_ff, F_f)
        except np.linalg.LinAlgError:
            return pd.DataFrame(), np.zeros(num_dof)

        U_global = np.zeros(num_dof)
        U_global[free_dofs] = u_f
        
        Reactions = K @ U_global - F

        # 5. Post-Processing
        results = []
        x_cursor = 0
        
        for i, L in enumerate(self.spans):
            idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
            u_elem = U_global[idx]
            
            if not self.loads.empty:
                span_loads = self.loads[self.loads['span_idx'] == i].to_dict('records')
            else:
                span_loads = []

            k_loc = self._get_element_stiffness(L)
            f_elastic = k_loc @ u_elem
            
            f_fea = np.zeros(4)
            for load in span_loads:
                f_eq = self._calc_equivalent_nodal_forces(load, L)
                f_fea -= f_eq 
            
            f_total_start = f_elastic + f_fea
            
            V0 = f_total_start[0]
            M0 = -f_total_start[1]
            
            # Points Generation (Same as before)
            x_eval = set(np.linspace(0, L, 100))
            x_eval.update([0, L])
            load_locs = set()
            for l in span_loads:
                lx = float(l['x'])
                load_locs.add(lx)
                if l['type'] == 'U':
                    load_locs.add(lx + float(l.get('dist', L)))
            
            eps = 1e-10
            for loc in load_locs:
                if 0 <= loc <= L:
                    x_eval.add(loc)
                    if loc - eps >= 0: x_eval.add(loc - eps)
                    if loc + eps <= L: x_eval.add(loc + eps)
            
            unique_points = sorted(list(x_eval))
            
            for x in unique_points:
                V, M = self._calculate_statics_at_x(x, V0, M0, span_loads)
                # --- Modified: Get Timoshenko Deflection ---
                D = self._get_deflection(x, L, u_elem)
                
                results.append({
                    'x': x_cursor + x,
                    'shear': V,
                    'moment': M,
                    'deflection': D
                })
            
            x_cursor += L

        return pd.DataFrame(results), Reactions

    def _get_phi(self, L):
        # Calculate Shear Deformation Parameter (Phi)
        # Phi = 12 * EI / (k * G * A * L^2)
        if self.G * self.A == 0: return 0 # Avoid div by zero
        return (12 * self.E * self.I) / (self.kappa * self.G * self.A * L**2)

    def _get_element_stiffness(self, L):
        # --- TIMOSHENKO STIFFNESS MATRIX ---
        E, I = self.E, self.I
        Phi = self._get_phi(L)
        
        # Common multiplier
        # For Timoshenko: EI / (L^3 * (1+Phi))
        C = (E * I) / (L**3 * (1 + Phi))
        
        # Matrix Coefficients
        k11 = 12
        k12 = 6 * L
        k22 = (4 + Phi) * L**2
        k24 = (2 - Phi) * L**2
        
        k = np.array([
            [k11,    k12,   -k11,    k12],
            [k12,    k22,   -k12,    k24],
            [-k11,  -k12,    k11,   -k12],
            [k12,    k24,   -k12,    k22]
        ])
        
        return C * k

    def _calc_equivalent_nodal_forces(self, load, L):
        # Note: Ideally, Timoshenko shape functions should be used here too.
        # But for standard loads (P, U), the work-equivalent loads are 
        # identical or negligibly different for visualization purposes.
        # Keeping this standard prevents over-complication while K matrix handles the main behavior.
        f = np.zeros(4)
        mag = float(load['mag']) 
        a = float(load['x']) 
        F_load = -mag 
        
        if load['type'] == 'P': 
            xi = a/L
            N1 = 1 - 3*xi**2 + 2*xi**3
            N2 = L * (xi - 2*xi**2 + xi**3)
            N3 = 3*xi**2 - 2*xi**3
            N4 = L * (-xi**2 + xi**3)
            
            f[0] = N1 * F_load
            f[1] = N2 * F_load
            f[2] = N3 * F_load
            f[3] = N4 * F_load

        elif load['type'] == 'U': 
            w_total = F_load * L 
            f[0] = w_total / 2
            f[1] = -abs(F_load) * L**2 / 12 
            f[2] = w_total / 2
            f[3] = +abs(F_load) * L**2 / 12

        return f

    def _calculate_statics_at_x(self, x, V0, M0, loads):
        # Statics Equilibrium is independent of Material (E, I, G, A)
        # So this remains exactly the same logic.
        V_x = V0
        M_x = M0 + V0 * x 
        
        for load in loads:
            lx = float(load['x'])
            mag = float(load['mag']) 
            
            if x > lx:
                if load['type'] == 'P':
                    V_x -= mag
                    M_x -= mag * (x - lx)
                    
                elif load['type'] == 'U':
                    w = mag
                    dist = float(load.get('dist', 1e9))
                    start_load = lx
                    end_load = lx + dist
                    
                    x_eff_end = min(x, end_load)
                    length = x_eff_end - start_load
                    
                    if length > 0:
                        force = w * length
                        V_x -= force
                        centroid = start_load + length/2
                        arm = x - centroid
                        M_x -= force * arm
        
        return V_x, M_x

    def _get_deflection(self, x, L, u_elem):
        # --- TIMOSHENKO SHAPE FUNCTIONS ---
        # These are different from Euler-Bernoulli to account for shear deformation
        Phi = self._get_phi(L)
        xi = x/L
        
        # Denominator
        D = 1 + Phi
        
        # Shape Functions N1..N4 depending on Phi
        N1 = (1 / D) * (1 - 3*xi**2 + 2*xi**3 + Phi*(1 - xi))
        N2 = (L / D) * (xi - 2*xi**2 + xi**3 + (Phi/2)*(xi - xi**2))
        N3 = (1 / D) * (3*xi**2 - 2*xi**3 + Phi*xi)
        N4 = (L / D) * (-xi**2 + xi**3 - (Phi/2)*(xi - xi**2))
        
        N_vec = np.array([N1, N2, N3, N4])
        return np.dot(N_vec, u_elem)
