import numpy as np
import pandas as pd

def run_beam_analysis(spans, supports, loads):
    """
    Direct Stiffness Method for Continuous Beam Analysis (1D).
    Supports: Point Loads (P) and Uniform Loads (UDL).
    """
    n_spans = len(spans)
    n_nodes = n_spans + 1
    dof_per_node = 2  # Vertical Displacement (v), Rotation (theta)
    total_dof = n_nodes * dof_per_node

    # 1. Global Stiffness Matrix (K) & Force Vector (F)
    K_global = np.zeros((total_dof, total_dof))
    F_global = np.zeros(total_dof)

    # Elastic Modulus (E) and Inertia (I) - Dummy constant values for proportion
    # (Since we solve for forces, E and I cancel out for determinate/indeterminate ratios if constant)
    E = 2e6  
    I = 0.001 

    # --- ASSEMBLE MATRICES ---
    for i in range(n_spans):
        L = spans[i]
        L2 = L * L
        L3 = L2 * L
        
        # Element Stiffness Matrix (Local)
        k_local = (E * I / L3) * np.array([
            [12,      6*L,    -12,     6*L],
            [6*L,     4*L2,   -6*L,    2*L2],
            [-12,    -6*L,     12,    -6*L],
            [6*L,     2*L2,   -6*L,    4*L2]
        ])

        # Map to Global Indices
        # Node i (2*i, 2*i+1) -> Node i+1 (2*(i+1), 2*(i+1)+1)
        idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
        
        for r in range(4):
            for c in range(4):
                K_global[idx[r], idx[c]] += k_local[r, c]

        # --- CALCULATE FIXED END ACTIONS (FEM) ---
        # Initialize FEM for this span [Fy_L, M_L, Fy_R, M_R]
        fem_vec = np.zeros(4) 
        
        span_loads = [l for l in loads if l.get('span_idx') == i]
        
        for load in span_loads:
            # === UNIFORM LOAD (UDL) ===
            if load['type'] == 'U' and abs(load['w']) > 0:
                w = -load['w'] # Downward is negative in math calc (usually)
                # FEM Formulas:
                # M_L = -wL^2/12, M_R = +wL^2/12
                # Fy = wL/2
                
                # Careful with signs: 
                # Vector convention: Up = +, Counter-Clockwise = +
                # Load w is gravity (down).
                
                # Fixed End Reactions (Forces exerted BY support ON beam)
                # Left Node
                fem_vec[0] += w * L / 2       # Fy
                fem_vec[1] += w * L2 / 12     # M (CCW +)
                # Right Node
                fem_vec[2] += w * L / 2       # Fy
                fem_vec[3] += -w * L2 / 12    # M (CW -)

            # === POINT LOAD (P) ===
            elif load['type'] == 'P' and abs(load['P']) > 0:
                P = -load['P'] # Downward P
                a = load['x']  # Distance from left
                b = L - a
                
                if 0 < a < L: # Ensure it's inside span
                    # FEM Formulas for Point Load:
                    # M_L = -Pab^2/L^2
                    # M_R = +Pa^2b/L^2
                    # Fy_L = Pb^2(3a+b)/L^3
                    # Fy_R = Pa^2(a+3b)/L^3
                    
                    # Left Node
                    fem_vec[0] += P * (b**2 * (3*a + b)) / L3  # Fy
                    fem_vec[1] += P * a * (b**2) / L2          # M (CCW +)
                    # Right Node
                    fem_vec[2] += P * (a**2 * (a + 3*b)) / L3  # Fy
                    fem_vec[3] += -P * (a**2) * b / L2         # M (CW -)

        # Subtract Fixed End Actions from Global Force Vector
        # (F_equiv = F_nodal - F_fixed)
        for r in range(4):
            F_global[idx[r]] -= fem_vec[r]

    # 2. Apply Boundary Conditions
    # Create mask for free DOFs
    free_dof = np.ones(total_dof, dtype=bool)
    
    vis_supports = [] # Data for visualization later

    for i in range(n_nodes):
        # Determine support type from input dataframe or logic
        sup_type = "None"
        if i < len(supports):
            sup_type = supports.iloc[i]['type']
        
        vis_supports.append({'x': sum(spans[:i]), 'type': sup_type})

        dof_y = 2*i
        dof_m = 2*i + 1
        
        if sup_type == "Pin":
            free_dof[dof_y] = False # Fix Y
            # Moment free
        elif sup_type == "Roller":
            free_dof[dof_y] = False # Fix Y
            # Moment free
        elif sup_type == "Fixed":
            free_dof[dof_y] = False # Fix Y
            free_dof[dof_m] = False # Fix Rotation

    # 3. Solve for Displacements
    K_reduced = K_global[np.ix_(free_dof, free_dof)]
    F_reduced = F_global[free_dof]
    
    try:
        D_reduced = np.linalg.solve(K_reduced, F_reduced)
    except np.linalg.LinAlgError:
        # Unstable structure (mechanism)
        D_reduced = np.zeros_like(F_reduced) 

    # Reconstruct Full Displacement Vector
    D_full = np.zeros(total_dof)
    D_full[free_dof] = D_reduced

    # 4. Post-Process: Internal Forces (SFD/BMD)
    # We will compute forces by "Cutting Sections" (Statics)
    # First, get Reactions (R = K*D - F_equivalent_nodal? No, easier: R = K*D + FEM_reaction)
    
    # Or cleaner: Calculate Element End Forces using slope-deflection
    # Then integrate along the beam.
    
    x_coords = []
    shear_vals = []
    moment_vals = []
    span_ids = []

    global_x = 0.0
    
    for i in range(n_spans):
        L = spans[i]
        idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
        u = D_full[idx] # [v1, theta1, v2, theta2]
        
        # Get Span Loads
        span_loads = [l for l in loads if l.get('span_idx') == i]

        # Calculate Member End Forces (Local Stiffness * u + FEM)
        # k_local defined earlier
        L2 = L*L; L3 = L2*L
        k_local = (E * I / L3) * np.array([
            [12,      6*L,    -12,     6*L],
            [6*L,     4*L2,   -6*L,    2*L2],
            [-12,    -6*L,     12,    -6*L],
            [6*L,     2*L2,   -6*L,    4*L2]
        ])
        
        # Re-calculate FEM for this span to add back
        fem_vec = np.zeros(4)
        for load in span_loads:
            if load['type'] == 'U':
                w = -load['w']
                fem_vec[0] += w * L / 2
                fem_vec[1] += w * L2 / 12
                fem_vec[2] += w * L / 2
                fem_vec[3] += -w * L2 / 12
            elif load['type'] == 'P':
                P = -load['P']
                a = load['x']
                b = L - a
                fem_vec[0] += P * (b**2 * (3*a + b)) / L3
                fem_vec[1] += P * a * (b**2) / L2
                fem_vec[2] += P * (a**2 * (a + 3*b)) / L3
                fem_vec[3] += -P * (a**2) * b / L2

        f_end = np.dot(k_local, u) + fem_vec
        # f_end = [Fy_L, M_L, Fy_R, M_R]
        # Sign convention:
        # Fy Up +, M CCW +
        
        # Start of span values
        V_start = f_end[0]
        M_start = f_end[1] # CCW is + for matrix. 
        # For Beam Theory Plotting: Sagging Moment is + (Smile). 
        # If M_start is CCW (Reaction moment), it creates sagging? 
        # M(x) = M_start_reaction + V*x - ...
        # Actually M_start from matrix is the "Action on the node". 
        # Action on beam = Action on node. 
        # Let's use Statics from Left End.
        
        num_pts = 100
        x_span = np.linspace(0, L, num_pts)
        
        # Add critical points (Load locations)
        for load in span_loads:
            if load['type'] == 'P':
                # Add point just before and just after to capture shear jump
                x_span = np.append(x_span, [load['x'] - 1e-5, load['x'] + 1e-5])
        
        x_span = np.sort(np.unique(x_span))
        x_span = x_span[(x_span >= 0) & (x_span <= L)]

        for x in x_span:
            # V(x) = V_start + sum(loads_left)
            # M(x) = -M_start + V_start*x + sum(moment_of_loads_left)
            # Matrix M1 is CCW. Beam sign convention: Hogging (Frown) is usually negative.
            # If M1 is + (CCW) at left end, it tries to lift the beam -> Sagging (Positive).
            # Wait, Standard Structural Mechanics: M_internal = M_reaction + ...
            # Actually: M_internal(at x) = M_start(CCW) + V_start*x - Loads...
            
            V_x = V_start
            M_x = M_start + V_start * x
            
            for load in span_loads:
                # UDL Contribution
                if load['type'] == 'U':
                    w_mag = load['w'] # Magnitude (+ down)
                    # Force: -w * x
                    V_x -= w_mag * x
                    # Moment: -w * x * (x/2)
                    M_x -= w_mag * x**2 / 2
                
                # Point Load Contribution
                elif load['type'] == 'P':
                    P_mag = load['P']
                    px = load['x']
                    if x > px:
                        V_x -= P_mag
                        M_x -= P_mag * (x - px)
            
            x_coords.append(global_x + x)
            shear_vals.append(V_x)
            moment_vals.append(M_x) # Plot positive for Sagging
            span_ids.append(i)
            
        global_x += L

    res_df = pd.DataFrame({
        'x': x_coords,
        'shear': shear_vals,
        'moment': moment_vals,
        'span_id': span_ids
    })
    
    vis_supports_df = pd.DataFrame(vis_supports)
    
    return res_df, vis_supports_df
