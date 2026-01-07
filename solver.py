import numpy as np
import pandas as pd

def solve_beam(spans, sup_df, loads_df, params):
    """
    Solves the continuous beam using Direct Stiffness Method (FEM).
    INCLUDES: Timoshenko Beam Theory (Shear Deformation).
    INCLUDES: Correct Sign Convention for Gravity Loads.
    """
    # --- 0. Safety Check for Empty Loads ---
    if loads_df.empty or 'span_index' not in loads_df.columns:
        loads_df = pd.DataFrame(columns=['span_index', 'type', 'mag', 'dist'])

    E = params['E'] # Pa (N/m2)
    I = params['I'] # m4
    b = params['b'] # m
    h = params['h'] # m
    
    # --- Timoshenko Parameters ---
    # Poisson's ratio for concrete approx 0.2
    nu = 0.2 
    G = E / (2 * (1 + nu))  # Shear Modulus
    k = 5.0 / 6.0           # Shear Correction Factor for Rectangle
    As = k * b * h          # Shear Area
    
    # 1. Setup Nodes & Elements
    n_spans = len(spans)
    n_nodes = n_spans + 1
    node_coords = [0] + list(np.cumsum(spans))
    
    n_dof = 2 * n_nodes
    K_global = np.zeros((n_dof, n_dof))
    F_global = np.zeros(n_dof)
    
    # 2. Build Stiffness Matrix (K) with Timoshenko Factor (Phi)
    for i in range(n_spans):
        L = spans[i]
        
        # Phi (Shear Deformation Parameter)
        # If Phi = 0, it behaves like Euler-Bernoulli
        Phi = (12 * E * I) / (G * As * L**2)
        
        # Common Multiplier
        const = (E * I) / ((1 + Phi) * L**3)
        
        # Timoshenko Stiffness Matrix Elements
        k11 = 12
        k12 = 6 * L
        k22 = (4 + Phi) * L**2
        k24 = (2 - Phi) * L**2  # Carry-over factor differs in Timoshenko
        
        k_ele = const * np.array([
            [k11,   k12, -k11,   k12],
            [k12,   k22, -k12,   k24],
            [-k11, -k12,  k11,  -k12],
            [k12,   k24, -k12,   k22]
        ])
        
        idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
        for r in range(4):
            for c in range(4):
                K_global[idx[r], idx[c]] += k_ele[r, c]

    # 3. Process Loads (Fixed End Actions - FEA)
    # Note: Using Standard FEA is generally acceptable for building frames 
    # even with Timoshenko stiffness, as global displacement governs.
    
    fea_local = [] 
    for _ in range(n_spans):
        fea_local.append(np.zeros(4)) 

    if not loads_df.empty:
        for _, load in loads_df.iterrows():
            span_idx = int(load['span_index'])
            L = spans[span_idx]
            mag = load['mag'] 
            
            # FEA Reactions (Upward +, CCW +)
            idx = [2*span_idx, 2*span_idx+1, 2*(span_idx+1), 2*(span_idx+1)+1]
            fea = np.zeros(4)
            
            if load['type'] == 'P':
                P = mag
                a = load['dist']
                b_dist = L - a
                
                fea[0] = (P * b_dist**2 * (3*a + b_dist)) / L**3
                fea[1] = (P * a * b_dist**2) / L**2
                fea[2] = (P * a**2 * (a + 3*b_dist)) / L**3
                fea[3] = -(P * a**2 * b_dist) / L**2
                
            elif load['type'] == 'U':
                w = mag
                fea[0] = w * L / 2
                fea[1] = w * L**2 / 12
                fea[2] = w * L / 2
                fea[3] = -w * L**2 / 12

            fea_local[span_idx] += fea
            
            # Global Load Vector F = F_ext - FEA
            F_global[idx[0]] -= fea[0]
            F_global[idx[1]] -= fea[1]
            F_global[idx[2]] -= fea[2]
            F_global[idx[3]] -= fea[3]

    # 4. Apply Boundary Conditions
    fixed_dofs = []
    for _, row in sup_df.iterrows():
        node_idx = int(row['id'])
        stype = row['type']
        
        fixed_dofs.append(2*node_idx) 
        if stype == 'Fixed':
            fixed_dofs.append(2*node_idx + 1)
            
    free_dofs = [i for i in range(n_dof) if i not in fixed_dofs]
    
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    F_ff = F_global[free_dofs]
    
    try:
        d_free = np.linalg.solve(K_ff, F_ff)
    except np.linalg.LinAlgError:
        return np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), {}
    
    d_all = np.zeros(n_dof)
    d_all[free_dofs] = d_free
    
    # 5. Post-Processing
    plot_points_per_span = 50
    x_total = []
    def_total = []
    moment_total = []
    shear_total = []
    
    for i in range(n_spans):
        L = spans[i]
        x0 = node_coords[i]
        u_ele = d_all[[2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]]
        x_local = np.linspace(0, L, plot_points_per_span)
        
        # Recalculate Phi for Shape Function (optional/refined) or reuse
        Phi = (12 * E * I) / (G * As * L**2)
        
        # 5.1 Deflection
        # Note: Ideally, Timoshenko shape functions are complex. 
        # Using Cubic Hermite is standard for visualization unless beam is extremely deep.
        xi = x_local / L
        N1 = 1 - 3*xi**2 + 2*xi**3
        N2 = L * (xi - 2*xi**2 + xi**3)
        N3 = 3*xi**2 - 2*xi**3
        N4 = L * (-xi**2 + xi**3)
        v_x = N1*u_ele[0] + N2*u_ele[1] + N3*u_ele[2] + N4*u_ele[3]
        
        # 5.2 Internal Forces (Back-calculation from K*d + FEA)
        # Must use the SAME Timoshenko K matrix here
        const = (E * I) / ((1 + Phi) * L**3)
        k11 = 12
        k12 = 6 * L
        k22 = (4 + Phi) * L**2
        k24 = (2 - Phi) * L**2
        
        k_ele = const * np.array([
            [k11,   k12, -k11,   k12],
            [k12,   k22, -k12,   k24],
            [-k11, -k12,  k11,  -k12],
            [k12,   k24, -k12,   k22]
        ])
        
        fea_vec = fea_local[i]
        
        # Force at Start Node
        f_int = np.dot(k_ele, u_ele) + fea_vec 
        
        Fy_start = f_int[0]
        M_start = f_int[1] # CCW +
        
        m_x = []
        v_x_static = []
        
        span_loads = loads_df[loads_df['span_index'] == i]
        
        for x in x_local:
            # Equilibrium Method
            V_curr = Fy_start
            M_curr = M_start + Fy_start * x
            
            if not span_loads.empty:
                for _, load in span_loads.iterrows():
                    mag = load['mag']
                    if load['type'] == 'P':
                        if x > load['dist']:
                            V_curr -= mag
                            M_curr -= mag * (x - load['dist'])
                    elif load['type'] == 'U':
                        udl_end = load['dist']
                        if x > 0:
                            len_cov = min(x, udl_end)
                            if len_cov > 0:
                                V_curr -= mag * len_cov
                                M_curr -= (mag * len_cov) * (x - len_cov/2)

            m_x.append(M_curr)
            v_x_static.append(V_curr)

        x_total.extend(x0 + x_local)
        moment_total.extend(m_x)
        shear_total.extend(v_x_static)
        def_total.extend(v_x) 

    # 6. Reactions
    R_vec = np.dot(K_global, d_all)
    FEA_R = np.zeros(n_dof)
    for i in range(n_spans):
        f = fea_local[i]
        idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
        FEA_R[idx[0]] += f[0]
        FEA_R[idx[1]] += f[1]
        FEA_R[idx[2]] += f[2]
        FEA_R[idx[3]] += f[3]
        
    R_final = R_vec + FEA_R
    reactions = {f"R{row['id']}": R_final[2*int(row['id'])] for _, row in sup_df.iterrows()}

    # Return def_total as is (Positive = Down in Plotly Y-axis flip, but in engineering graph negative is usually down)
    # The previous fix removed the *-1, assuming Plotly handles it or we want signed value.
    # Standard: Deflection Down is Negative.
    return np.array(x_total), np.array(moment_total), np.array(shear_total), np.array(def_total), reactions
