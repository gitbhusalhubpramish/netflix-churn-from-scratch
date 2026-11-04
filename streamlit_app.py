import kagglehub
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

path = kagglehub.dataset_download("abdulwadood11220/netflix-customer-churn-dataset", "netflix_customer_churn.csv")

df = pd.read_csv(path)

df["subscription_type"] = df["subscription_type"].map({"Basic":1, "Premium":2, "Standard": 3})

df["device"] = df["device"].map({"Mobile":1, "Tablet":2, "Laptop":3, "Desktop": 4, "TV": 5})

x,y = df[["age", "subscription_type", "watch_hours", "monthly_fee"]], df["churned"]

xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.2, random_state=42, stratify=y)


