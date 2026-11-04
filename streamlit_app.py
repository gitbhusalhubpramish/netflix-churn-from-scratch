import kagglehub
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import graphviz

st.write('''## Data(set)''')
path = kagglehub.dataset_download("abdulwadood11220/netflix-customer-churn-dataset", "netflix_customer_churn.csv")

df = pd.read_csv(path)

df["subscription_type"] = df["subscription_type"].map({"Basic":1, "Premium":2, "Standard": 3})

df["device"] = df["device"].map({"Mobile":1, "Tablet":2, "Laptop":3, "Desktop": 4, "TV": 5})

x,y = df[["age", "subscription_type", "watch_hours", "monthly_fee"]], df["churned"]

xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.2, random_state=42, stratify=y)

def gini(y):
	cls, cnt = np.unique(y, return_counts=True)
	impr = 1
	for count in cnt:
		impr -= (count/len(y))**2
	return impr

def split(x,y,feature, trashold):
	left = x[feature]<=trashold
	right = x[feature]>trashold
	return (x[left], y[left], x[right], y[right])

def split_score(ly,ry):
	n = len(ly)+len(ry)
	return (len(ly)/n)*gini(ly) + (len(ry)/n)*gini(ry)

def best_split(x,y):
	best = None
	bestscr = float("inf")
	for feature in x:
		values = np.unique(x[feature])
		for trashold in values:
			xl,yl,xr,yr = split(x,y,feature,trashold)
			if len(yl) == 0 or len(yr) == 0:
				continue
			score=split_score(yl,yr)
			if score<bestscr:
				bestscr = score
				best = {"feature": feature, "trashold": trashold, "left":(xl,yl), "right": (xr,yr)}

	return best

class Node:
	def __init__(self, feature=None, trashold=None, left = None, right=None, value=None):
		self.feature = feature
		self.trashold = trashold
		self.left = left
		self.right = right
		self.value = value

def build_tree(X,Y, depth, max_depth):
	if depth == max_depth or len(set(Y)) == 1:
		return Node(value=max(set(Y), key=list(Y).count))

	best = best_split(X,Y)
	left = build_tree(best["left"][0], best["left"][1], depth+1, max_depth)
	right = build_tree(best["right"][0], best["right"][1], depth+1, max_depth)
	return Node(feature=best["feature"], trashold=best["trashold"], left=left, right=right)

def run_tree(node, x):
	if node.value is not None:
		return node.value
	if x[node.feature]<=node.trashold:
		return run_tree(node.left, x)
	else:
		return run_tree(node.right, x)

def build_graph(root):
	graph = []
	def traverse(node, node_id=0):
		nonlocal graph
		if node is None:
			return 
		if node.value is not None:
			graph.append(f'{node_id} [label="predict: {node.value}"]')
			return

		graph.append(f'{node_id} [label="X[{node.feature}]>={node.trashold:.3f}"];')
		
		for child, child_id, label in [(node.left, 2*node_id+1, "True"), (node.right, 2*node_id+2, "False")]:
			if child:
				graph.append(f'{node_id}-> {child_id} [label="{label}"];')
				traverse(child, child_id)

	traverse(root)
	return """digraph Tree{\n graph [rankdir=TB, nodesep=0.08, ranksep=0.3 spline=line, fontsize=5, pad = 0.1]; node [fontsize=2, width=0.2, height=0.1, margin=0.1, fixedsize=false] edge [fontsize=2, arrowsixe = 0.1];""" + "\n".join(graph)+"}"




st.write('''## Models''')
st.write('''It trains 9 decision tree with same algoritm but with different depth.''')

charts = [st.empty() for _ in range(1,10*2)]

acc = {"train":[], "test":[]}

st.write('''## Accuracy''')
st.write('''The accuracy of 9 model given above is:''')

accchat = st.empty()

for i in range(1,10):
	tree = build_tree(xtrain,ytrain, 0,i)
	graph = graphviz.Source(build_graph(tree))
	svg=graph.pipe(format="svg", engine="dot").decode("utf-8")
	charts[i*2].graphviz(graph)
	charts[i*2+1].download_button(label="Download Tree", data = svg, file_name=f"decisiontree({i}).svg", mime="image/svg+xml")
	prediction = xtest.apply(lambda x: run_tree(tree,x), axis =1)
	predictiontrain = xtrain.apply(lambda x:run_tree(tree, x), axis=1)
	acc["test"].append(accuracy_score(prediction, ytest))
	acc["train"].append(accuracy_score(predictiontrain, ytrain))
	accchat.line_chart(acc)
	
