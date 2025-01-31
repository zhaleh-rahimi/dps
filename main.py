# main.py
# Entry point to run the demand prioritization system
from models.gnn_clustering import CustomerClustering
from agents.rl_agent import InventoryRLAgent
from models.gen_ai_simulation import DemandSimulator
from utils.data_loader import load_customer_data, load_inventory_data, load_demand_data


def main():
    # Load data
    customers = load_customer_data()
    inventory = load_inventory_data()
    demand = load_demand_data()

    # Step 1: Cluster customers using GNN
    clustering_model = CustomerClustering()
    customer_clusters = clustering_model.cluster_customers(customers)

    # Step 2: Train RL model for inventory prioritization
    rl_agent = InventoryRLAgent()
    rl_agent.train(customer_clusters, inventory, demand)

    # Step 3: Simulate demand with Generative AI
    demand_simulator = DemandSimulator()
    simulated_demand = demand_simulator.generate_scenarios(demand)

    print("Pipeline execution completed!")


if __name__ == "__main__":
    main()
