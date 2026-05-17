import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

def simple_benson_style_solver(P, B, b, num_points=50):
    """
    A simplified 2D multi-objective Linear Programming solver.
    
    Problem:
        Minimize P @ x  (where P is a 2 x d matrix for 2 objectives)
        Subject to: B @ x >= b (which we convert to -B @ x <= -b for scipy)
        
    Args:
        P: Objective matrix (q x d), here q=2.
        B: Constraint matrix (m x d).
        b: Constraint RHS vector (m,).
        num_points: Number of scalarized LPs to solve to build the frontier.
        
    Returns:
        pareto_front: A list of (y1, y2) coordinates mapping the boundary.
        vertices: The corresponding optimal x portfolios.
    """
    q, d = P.shape
    if q != 2:
        raise ValueError("This simplified solver is designed for 2 objectives (q=2) for visualization.")

    # Scipy linprog uses A_ub @ x <= b_ub, so we negate the Bx >= b formulation
    A_ub = -B
    b_ub = -b
    
    pareto_front = []
    optimal_portfolios = []

    # In Benson's algorithm, we iteratively explore the dual space.
    # Here, we systematically sweep the dual weights (w1, w2) where w1 + w2 = 1.
    # These represent the normal vectors to the supporting hyperplanes of the set.
    weights = np.linspace(0.001, 0.999, num_points)
    
    for w1 in weights:
        w2 = 1.0 - w1
        weight_vector = np.array([w1, w2])
        
        # Scalarize the objective: c^T = w^T P
        c = weight_vector @ P
        
        # Solve the single-objective LP
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(None, None), method='highs')
        
        if res.success:
            x_opt = res.x
            # Calculate the outcome in the objective space: y = Px
            y_opt = P @ x_opt
            
            # Avoid duplicate vertices (due to numerical flat edges)
            if not pareto_front or not np.allclose(pareto_front[-1], y_opt, atol=1e-5):
                pareto_front.append(y_opt)
                optimal_portfolios.append(x_opt)
        else:
            print(f"LP failed for weights {weight_vector}. Region might be unbounded or infeasible.")

    return np.array(pareto_front), np.array(optimal_portfolios)

# ==========================================
# Example Usage: Testing the Solver
# ==========================================
if __name__ == "__main__":
    # Let's set up a mock portfolio optimization problem.
    # d = 2 assets. We want to evaluate the trade-off (Liquidation Map).
    
    # P: The objective/liquidation map (e.g., valuing the portfolio in 2 different scenarios)
    P = np.array([
        [1.0, 0.0],  # Objective 1: Maximize Asset 1 (we minimize negative)
        [0.0, 1.0]   # Objective 2: Maximize Asset 2
    ])
    
    # Bx >= b: The constraints (e.g., initial capital, risk limits, transaction cost boundaries)
    # 1. x_1 + 2*x_2 >= 10
    # 2. 3*x_1 + x_2 >= 15
    # 3. x_1 >= 0
    # 4. x_2 >= 0
    B = np.array([
        [ 1.0,  2.0],
        [ 3.0,  1.0],
        [ 1.0,  0.0],
        [ 0.0,  1.0]
    ])
    b = np.array([10.0, 15.0, 0.0, 0.0])
    
    print("Running Benson-style multi-objective solver...")
    frontier_y, portfolios_x = simple_benson_style_solver(P, B, b, num_points=100)
    
    print(f"Found {len(frontier_y)} unique vertices on the Pareto frontier.")
    
    # Optional: Plot the result if you are running this locally
    plt.figure(figsize=(8, 5))
    plt.plot(frontier_y[:, 0], frontier_y[:, 1], marker='o', linestyle='-', color='b')
    plt.fill_between(frontier_y[:, 0], frontier_y[:, 1], np.max(frontier_y[:, 1]) + 5, color='gray', alpha=0.3)
    plt.title("Upper Image (Set of Superhedging Portfolios)")
    plt.xlabel("Asset 1")
    plt.ylabel("Asset 2")
    plt.grid(True)
    plt.show()