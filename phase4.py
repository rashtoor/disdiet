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
    
    query1 = graph.run("""MATCH (m:disease)-[r:linked_to]->(n:diet) RETURN m,n,r""").to_data_frame()
    
    



    


    
    
    



   

    # Formatting in html
    #q2="perfect"
    #file = io.open("/home/ubuntu/disdiet/templates/rnn_index.html", "r", encoding='utf-8')
    #q=file.read()
    #html = '{% extends' + q + '%} {% block content %}'
    html=''
    html = addContent(html, header(
        'Harmful Diets for '+dis, color='black'))
    html = addContent(html, box(dr_harm[col].to_html()))
    html = addContent(html, header(
        'Helpful Diets for '+dis, color='black'))
    html = addContent(html, box(dr_help[col].to_html()))
   
    return f'<div>{html}</div>'


def header(text, color='black'):
    """Create an HTML header"""

    raw_html = f'<h1 style="margin-top:12px;color: {color};font-size:54px"><center>' + str(
            text) + '</center></h1>'
    return raw_html


def box(text):
    """Create an HTML box of text"""
    raw_html = '<div style="border-bottom:1px inset black;border-top:1px inset black;padding:8px;font-size: 28px;">' + str(
            text) + '</div>'
    return raw_html


def addContent(old_html, raw_html):
    """Add html content together"""

    old_html += raw_html
    return old_html
