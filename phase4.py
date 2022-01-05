from py2neo import Graph
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import Pipeline as imbpipeline
import io
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc
from collections import Counter
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline



def na(dis):
    
    
    graph = Graph("bolt://15.207.24.149:7687", auth=("neo4j", "dilpreet"))
    #query for creating graph for new node like crohns disease
    graph.run("""Load CSV with headers from "https://docs.google.com/spreadsheets/d/e/2PACX-1vSab3yrUmdt0ov77T3h555Ow6YdtncsfUZzyllLKAkgOOH6iL3n-2C0JT8qUODvnqnZDzFGAcfctQBR/pub?gid=0&single=true&output=csv" as line merge(n:disease{Name:line.disease}) merge (m:diet{Name:line.diet}) merge (n)-[r:linked_to{cooccurrence:toFloat(line.link),relation:line.relation}]->(m)
""").to_data_frame()
    
    results = graph.run("""MATCH (m:disease)-[r:linked_to]->(n:diet) RETURN m.Name as disease, collect(a.Name) as diet""")
    
                                                         
    nodes = []
    rels = []
    i = 0
    for record in results:
        nodes.append({"Name": record["disease"], "label": "disease"})
        target = i
        i += 1
        for Name in record['diet']:
            diet1 = {"Name": Name, "label": "diet"}
            try:
                source = nodes.index(diet1)
            except ValueError:
                nodes.append(diet1)
                source = i
                i += 1
            rels.append({"source": source, "target": target})
   
    return Response(dumps({"nodes": nodes, "links": rels}),mimetype="application/json")

   
