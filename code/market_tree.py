import numpy as np
from scipy.optimize import linprog
import itertools

class MarketNode:
    """
    Represents a node in the event tree at time t, state omega.
    """
    def __init__(self, t, omega_id, bid_ask_matrix):
        self.t = t
        self.omega_id = omega_id
        self.Pi = bid_ask_matrix  # d x d bid-ask matrix Pi_t(omega)
        self.d = bid_ask_matrix.shape[0]
        self.successors = []      # List of successor MarketNodes
        self.payoff = None        # X(omega) if t == T

    def get_dual_cone_constraints(self):
        """
        Generates the constraint matrix representing the positive dual cone K_t^+.
        According to the paper, K_t is spanned by (pi^{ij} * e^i - e^j) and e^i.
        Therefore, K_t^+ = {v | v^T u >= 0 for all u in K_t}.
        Returns A_ub, b_ub such that A_ub @ v <= b_ub enforces v in K_t^+.
        """
        constraints = []
        # v_j <= pi^{ij} v_i  =>  -pi^{ij} v_i + v_j <= 0
        for i in range(self.d):
            for j in range(self.d):
                if i != j:
                    row = np.zeros(self.d)
                    row[i] = -self.Pi[i, j]
                    row[j] = 1.0
                    constraints.append(row)
        
        # v_i >= 0 => -v_i <= 0 (Dual cone positivity)
        for i in range(self.d):
            row = np.zeros(self.d)
            row[i] = -1.0
            constraints.append(row)
            
        return np.array(constraints), np.zeros(len(constraints))

    def add_successor(self, node):
        self.successors.append(node)