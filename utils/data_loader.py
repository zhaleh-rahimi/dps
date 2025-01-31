import pandas as pd


def load_customer_data():
    return pd.read_csv("data/customer_data.csv")


def load_inventory_data():
    return pd.read_csv("data/inventory_data.csv")


def load_demand_data():
    return pd.read_csv("data/demand_data.csv")
