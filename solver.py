import numpy as np
import pandas as pd

def solve_beam(spans, sup_df, loads_df, params):
    """
    Solves the continuous beam using Direct Stiffness Method (FEM).
    Pure NumPy implementation.
    """
    # --- 0. Safety Check for Empty Loads ---
    # ถ้า loads_df ว่าง ให้สร้างตารางเปล่าที่มีชื่อคอลัมน์ครบ เพื่อกัน KeyError
    if loads_df.empty or 'span_index' not in loads_df.columns:
        loads_df = pd.DataFrame(columns=['span_index', 'type', 'mag', 'dist'])

    E = params['E']
    I = params['I']
    
    # 1. Setup Nodes & Elements
    n_spans = len(spans)
    n_nodes = n_spans + 1
    node_coords = [0] + list(np.cumsum(spans))
    
    # DOFs: 2 per node (Vertical Y, Rotation Theta) -> Total 2*n_nodes
    n_dof = 2 * n_nodes
    K_global = np.zeros((n_dof, n_dof))
    F_global = np.zeros(n_dof)
    
    # 2. Build Stiffness Matrix (K)
    for i in range(n_spans):
        L = spans[i]
        # Element Stiffness Matrix
        k = (E * I / L**3) * np.array([
            [12, 6*L, -12, 6*L],
            [6*L, 4*L**2, -6*L, 2*L**2],
            [-12, -6*L, 12, -6*L],
            [6*L, 2*L**2, -6*L, 4*L**2]
        ])
        
        # Map to Global Indices
        idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
        for r in range(4):
            for c in range(4):
                K_global[idx[r], idx[c]] += k[r, c]

    # 3. Process Loads (Fixed End Actions - FEA)
    fea_local = [] 
    for _ in range(n_spans):
        fea_local.append(np.zeros(4)) 

    if not loads_df.empty:
        for _, load in loads_df.iterrows():
            span_idx = int(load['span_index'])
            L = spans[span_idx]
            mag = load['mag'] 
            
            # Load direction: Input mag is positive. Load Down is usually -Y in FEM.
            P = -mag 
            
            idx = [2*span_idx, 2*span_idx+1, 2*(span_idx+1), 2*(span_idx+1)+1]
            fea = np.zeros(4)
            
            if load['type'] == 'P':
                a = load['dist']
                b = L - a
                # FEM standard: P down at distance a
                fea[0] = (P * b**2 * (3*a + b)) / L**3
                fea[1] = (P * a * b**2) / L**2
                fea[2] = (P * a**2 * (a + 3*b)) / L**3
                fea[3] = -(P * a**2 * b) / L**2
                
            elif load['type'] == 'U':
                w = P # N/m (negative)
                # Full span formulas (Current logic assumes full span UDL)
                fea[0] = w * L / 2
                fea[1] = w * L**2 / 12
                fea[2] = w * L / 2
                fea[3] = -w * L**2 / 12

            fea_local[span_idx] += fea
            
            # Add to Global F (Subtract FEA because F = K*d + FEA -> K*d = -FEA)
            F_global[idx[0]] -= fea[0]
            F_global[idx[1]] -= fea[1]
            F_global[idx[2]] -= fea[2]
            F_global[idx[3]] -= fea[3]

    # 4. Apply Boundary Conditions
    fixed_dofs = []
    for _, row in sup_df.iterrows():
        node_idx = int(row['id'])
        stype = row['type']
        
        fixed_dofs.append(2*node_idx) # Fix Vertical
        if stype == 'Fixed':
            fixed_dofs.append(2*node_idx + 1) # Fix Rotation
            
    free_dofs = [i for i in range(n_dof) if i not in fixed_dofs]
    
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    F_ff = F_global[free_dofs]
    
    # Solve
    try:
        d_free = np.linalg.solve(K_ff, F_ff)
    except np.linalg.LinAlgError:
        # Fallback for unstable
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
        
        # Nodal displacements
        u_ele = d_all[[2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]]
        
        # Local coordinate
        x_local = np.linspace(0, L, plot_points_per_span)
        
        # 5.1 Deflection (Hermite Shape Functions)
        xi = x_local / L
        N1 = 1 - 3*xi**2 + 2*xi**3
        N2 = L * (xi - 2*xi**2 + xi**3)
        N3 = 3*xi**2 - 2*xi**3
        N4 = L * (-xi**2 + xi**3)
        
        v_x = N1*u_ele[0] + N2*u_ele[1] + N3*u_ele[2] + N4*u_ele[3]
        
        # 5.2 Internal Forces (Statics Method)
        # Calculate forces at left end of span from Stiffness
        k_ele = (E * I / L**3) * np.array([
            [12, 6*L, -12, 6*L],
            [6*L, 4*L**2, -6*L, 2*L**2],
            [-12, -6*L, 12, -6*L],
            [6*L, 2*L**2, -6*L, 4*L**2]
        ])
        
        fea_vec = fea_local[i]
        f_int = np.dot(k_ele, u_ele) + fea_vec 
        
        Fy_start = f_int[0]
        M_start = f_int[1] # Counter-Clockwise +
        
        m_x = []
        v_x_static = []
        
        # Filter loads for THIS span only
        span_loads = loads_df[loads_df['span_index'] == i]
        
        for x in x_local:
            # Equilibrium at section x
            # V = Sum(Forces Left) = Fy_start - Sum(Loads)
            # M = Sum(Moments Left) = -M_start + Fy_start*x - Sum(Load * arm)
            
            V_curr = Fy_start
            M_curr = -M_start + Fy_start * x
            
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
                                # Moment arm from centroid of load part to x
                                # Centroid is at len_cov/2 from start. Arm = x - len_cov/2
                                M_curr -= (mag * len_cov) * (x - len_cov/2)

            m_x.append(M_curr)
            v_x_static.append(V_curr)

        x_total.extend(x0 + x_local)
        moment_total.extend(m_x)
        shear_total.extend(v_x_static)
        def_total.extend(v_x) 

    # 6. Reactions
    # R = K*d + FEA_global (Force needed to maintain d)
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
    
    reactions = {}
    for _, row in sup_df.iterrows():
        nid = row['id']
        reactions[f"R{nid}"] = R_final[2*nid]

    return np.array(x_total), np.array(moment_total), np.array(shear_total), np.array(def_total)*-1, reactions
