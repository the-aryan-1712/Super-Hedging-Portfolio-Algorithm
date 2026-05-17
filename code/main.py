import numpy as np

# Assuming you saved the previous snippets into these files:
from market_tree import MarketNode
from scalar_pricing import compute_scalar_superhedging_price

def create_bid_ask_matrix(bid, ask):
    """
    Creates the d x d bid-ask matrix Pi_t.
    Asset 0 is Cash, Asset 1 is Stock.
    Pi[i, j] denotes the number of units of asset i to buy one unit of asset j.
    """
    Pi = np.eye(2)
    Pi[0, 1] = ask        # Cash needed to buy 1 Stock
    Pi[1, 0] = 1.0 / bid  # Stock needed to buy 1 Cash (equivalent to selling stock)
    return Pi

def run_example_3_7():
    """
    Recreates Example 3.7 from the paper: A one-period binomial model 
    with a digital option and transaction costs.
    """
    print("--- Setting up Market Model (Example 3.7) ---")
    
    # 1. Define Bid-Ask Matrices based on the paper's parameters
    # t=0: Bid 18, Ask 25
    pi_0 = create_bid_ask_matrix(bid=18.0, ask=25.0)
    
    # t=1 (Up Node): Bid 20, Ask 26
    pi_1_up = create_bid_ask_matrix(bid=20.0, ask=26.0)
    
    # t=1 (Down Node): Bid 16, Ask 23
    pi_1_down = create_bid_ask_matrix(bid=16.0, ask=23.0)

    # 2. Build the Event Tree
    root = MarketNode(t=0, omega_id="root", bid_ask_matrix=pi_0)
    
    node_up = MarketNode(t=1, omega_id="up", bid_ask_matrix=pi_1_up)
    node_down = MarketNode(t=1, omega_id="down", bid_ask_matrix=pi_1_down)
    
    root.add_successor(node_up)
    root.add_successor(node_down)

    # 3. Set Terminal Payoffs X(omega) = (Cash, Stock)
    # The option is an asset-or-nothing call with physical delivery, strike K=24.
    # Payoff in the up-node is (0, 1)^T and down-node is (0, 0)^T.
    node_up.payoff = np.array([0.0, 1.0])
    node_down.payoff = np.array([0.0, 0.0])

    print("Market tree successfully constructed.")
    print(f"Root Matrix:\n{root.Pi}")

    # 4. Compute the Scalar Superhedging Price
    print("\n--- Computing Scalar Superhedging Price ---")
    try:
        # We calculate the price in terms of numeraire 0 (Cash)
        cash_price = compute_scalar_superhedging_price(root, numeraire_index=0)
        
        print(f"Algorithm Output: {cash_price:.4f} units of Cash")
        print("Paper Expected Output: 25.0000 units of cash corresponding to a buy-and-hold strategy.")
        
        if abs(cash_price - 25.0) < 1e-4:
            print("SUCCESS: The algorithm successfully matched the paper's result!")
            
    except Exception as e:
        print(f"An error occurred during calculation: {e}")

if __name__ == "__main__":
    run_example_3_7()