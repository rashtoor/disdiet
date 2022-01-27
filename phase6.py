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
import matplotlib.pyplot as plt
from os import getcwd
import io
import base64


def network_analysis(dis):
    
    #graph = Graph("https://15.207.24.149:7473", auth=("neo4j", "dilpreet"))
    graph = Graph("bolt://15.207.24.149:7687", auth=("neo4j", "dilpreet"))
    #query for creating graph for new node like crohns disease
    graph.run("""Load CSV with headers from "https://docs.google.com/spreadsheets/d/e/2PACX-1vSab3yrUmdt0ov77T3h555Ow6YdtncsfUZzyllLKAkgOOH6iL3n-2C0JT8qUODvnqnZDzFGAcfctQBR/pub?gid=0&single=true&output=csv" as line merge(n:disease{Name:line.disease}) merge (m:diet{Name:line.diet}) merge (n)-[r:linked_to{cooccurrence:toFloat(line.link),relation:line.relation}]->(m)
""").to_data_frame()
    #graph.run("""MATCH (n:disease{Name:$dis})-[r:linked_to]-()
     #         SET r.cooccurrence = toFloat(r.cooccurrence)""",dis=dis)
    existing_links1 = graph.run("""MATCH (m:disease)-[r:linked_to]->(n:diet) 
                                      where m.Name='IBD'
                                      RETURN m.Name as node1, n.Name as node2
                                      """).to_data_frame()
    existing_links2 = graph.run("""MATCH (m:disease)-[r:linked_to]->(n:diet) 
                                      where m.Name='UC' 
                                      RETURN m.Name as node1, n.Name as node2
                                      """).to_data_frame()
    pred_existing_links = graph.run("""MATCH (m:disease)-[r:linked_to]->(n:diet) 
                                    where m.Name=$dis
                                    RETURN m.Name as node1, n.Name as node2
                                    """,dis=dis).to_data_frame()
    

    pdList1 = [existing_links1,existing_links2] 
    pdList2 = [pred_existing_links] 
    df = pd.concat(pdList1) #train_test
    df1= pd.concat(pdList2) 
    



    


    
    
    print(df.shape)
    print(df1.shape)



    query1 = graph.run("""match p=(m:disease)-[r1]->()<-[r2]-()-[r3]->(a:diet) 
                       with a.Name as diet_related, r1.cooccurrence+r2.cooccurrence+r3.cooccurrence as cc, m.Name as m 
                       with distinct diet_related as diets, collect(cc) as cooccurrence_all, m as m 
                       return diets as node2,m as node1, 
                   reduce(acc=0,v IN cooccurrence_all|acc+v) as cooccurrence_sum order by diets
                   """).to_data_frame()
    
  
    df=pd.merge(df, query1, how="left", on=['node1', 'node2'])
    df1=pd.merge(df1, query1, how="left", on=['node1', 'node2'])
    
    df=df.fillna(0)
    df1=df1.fillna(0)
    
   
    query2 = graph.run("""
                       CALL gds.triangleCount.write({
                           nodeProjection: '*',
                           relationshipProjection: {
                               relType: {
                                   type: '*',
                                   orientation: 'UNDIRECTED',
                                   properties: {}
                                   }
                               },
                           writeProperty: 'trianglesCount'
                           }); """)
   
    query3= graph.run(""" MATCH (node)
                      WHERE exists(node.`trianglesCount`)
                      RETURN node.Name, node.`trianglesCount` AS triangles
                      ORDER BY triangles DESC;
                      """).to_data_frame()

    #print(df.shape[0])

    v_min=[]
    v_max=[]
    for i in range(df.shape[0]):
        a=df.iloc[i,0]
        b=df.iloc[i,1]
        a1=query3.loc[query3['node.Name']==a]
        a2=a1.iat[0,1]
        b1=query3.loc[query3['node.Name']==b]
        b2=b1.iat[0,1]
   
        if a2>b2:
            v_min.insert(i,b2)
            v_max.insert(i,a2)
      
        else:
            v_max.insert(i,b2)
            v_min.insert(i,a2)



    df["MaxTriangles"]=v_max
    df["MinTriangles"]=v_min




    

    v_min=[]
    v_max=[]
    for i in range(df1.shape[0]):
        a=df1.iloc[i,0]
        b=df1.iloc[i,1]
        a1=query3.loc[query3['node.Name']==a]
        a2=a1.iat[0,1]
        b1=query3.loc[query3['node.Name']==b]
        b2=b1.iat[0,1]
   
        if a2>b2:
            v_min.insert(i,b2)
            v_max.insert(i,a2)
      
        else:
            v_max.insert(i,b2)
            v_min.insert(i,a2)



    df1["MaxTriangles"]=v_max
    df1["MinTriangles"]=v_min



    query4 = graph.run("""
                       CALL gds.localClusteringCoefficient.write({
                           nodeProjection: '*',
                           relationshipProjection: {
                               relType: {
                                   type: '*',
                                   orientation: 'UNDIRECTED',
                                   properties: {}
                                   }
                               },
                           writeProperty: 'coefficient'
                           }); """)
   
    query5= graph.run(""" MATCH (node)
                      WHERE exists(node.`coefficient`)
                      RETURN node.Name, node.`coefficient` AS t
                      ORDER BY t DESC;
                      """).to_data_frame()


    c_min=[]
    c_max=[]
    for i in range(df.shape[0]):
        a=df.iloc[i,0]
        b=df.iloc[i,1]
        a1=query5.loc[query5['node.Name']==a]
        a2=a1.iat[0,1]
        b1=query5.loc[query5['node.Name']==b]
        b2=b1.iat[0,1]
    
        if a2>b2:
            c_min.insert(i,b2)
            c_max.insert(i,a2)
        else:
            c_max.insert(i,b2)
            c_min.insert(i,a2)

    df["MaxCoefficient"]=c_max
    df["MinCoefficient"]=c_min




    
    c_min=[]
    c_max=[]
    for i in range(df1.shape[0]):
        a=df1.iloc[i,0]
        b=df1.iloc[i,1]
        a1=query5.loc[query5['node.Name']==a]
        a2=a1.iat[0,1]
        b1=query5.loc[query5['node.Name']==b]
        b2=b1.iat[0,1]
    
        if a2>b2:
            c_min.insert(i,b2)
            c_max.insert(i,a2)
        else:
            c_max.insert(i,b2)
            c_min.insert(i,a2)

    df1["MaxCoefficient"]=c_max
    df1["MinCoefficient"]=c_min

    query6=graph.run("""MATCH (p1) MATCH (p2)
                     RETURN p1.Name as node1, p2.Name as node2, gds.alpha.linkprediction.commonNeighbors(p1, p2) AS common_neighbors""").to_data_frame()

    df=pd.merge(df, query6, on=['node1', 'node2'])
    df1=pd.merge(df1, query6, on=['node1', 'node2'])


    query7=graph.run("""MATCH (p1) MATCH (p2)
                     RETURN p1.Name as node1, p2.Name as node2, gds.alpha.linkprediction.preferentialAttachment(p1, p2) AS preferential_attachment""").to_data_frame()

    df=pd.merge(df, query7, on=['node1', 'node2'])
    df1=pd.merge(df1, query7, on=['node1', 'node2'])

    query8=graph.run("""MATCH (p1) MATCH (p2)
                     RETURN p1.Name as node1, p2.Name as node2, gds.alpha.linkprediction.totalNeighbors(p1, p2) AS total_neighbors""").to_data_frame()

    df=pd.merge(df, query8, on=['node1', 'node2'])
    df1=pd.merge(df1, query8, on=['node1', 'node2'])
    #print(df_new2)

    query9=graph.run("""CALL gds.alpha.allShortestPaths.stream({
        nodeProjection: '*',
        relationshipProjection: {
            relType: {
                type: '*',
                orientation: 'UNDIRECTED',
                properties: {
                    cooccurrence: {
                        property: 'cooccurrence',
                        defaultValue: 1
                        }
                    }
                }
            },
        relationshipWeightProperty: 'cooccurrence'
        })
  
        YIELD sourceNodeId, targetNodeId, distance
        WITH sourceNodeId, targetNodeId, distance
        WHERE gds.util.isFinite(distance) = true

    MATCH (node1) WHERE id(node1) = sourceNodeId
    MATCH (node2) WHERE id(node2) = targetNodeId
    WITH node1, node2, distance WHERE node1 <> node2

    RETURN node1.Name AS node1, node2.Name AS node2, distance
    ORDER BY distance DESC, node1 ASC, node2 ASC
    LIMIT 10000""").to_data_frame()

    #print(query9)
    df=pd.merge(df, query9, on=['node1', 'node2'])
    df1=pd.merge(df1, query9, on=['node1', 'node2'])
    #print(df_new2)
    
    query10=graph.run("""CALL gds.louvain.stream({
        nodeProjection: '*',
        relationshipProjection: {
            relType: {
                type: '*',
                orientation: 'UNDIRECTED',
                properties: {
                    cooccurrence: {
                        property: 'cooccurrence',
                        defaultValue: 1
                        }
                    }
                }
            },
        relationshipWeightProperty: 'cooccurrence',
        includeIntermediateCommunities: true
        })

        YIELD nodeId, communityId AS community, intermediateCommunityIds
        WITH gds.util.asNode(nodeId) AS node, intermediateCommunityIds[0] AS smallestCommunity
        SET node.louvainData = smallestCommunity
        """).to_data_frame()
    
    query11=graph.run("""MATCH (p1) MATCH (p2)
                      RETURN p1.Name as node1, p2.Name as node2, gds.alpha.linkprediction.sameCommunity(p1, p2, 'louvainData') AS sp
                      """).to_data_frame()
    
    #print(query11)    


    df=pd.merge(df, query11, on=['node1', 'node2'])
    df1=pd.merge(df1, query11, on=['node1', 'node2'])
    
    
    query12 = graph.run("""match (m:disease)-[r]-(n:diet) where toInteger(r.relation)<2 return m.Name as node1, n.Name as node2, toInteger(r.relation) as relation
    """).to_data_frame()
    df=pd.merge(df, query12, on=['node1', 'node2'])
    print(df)
    df1=pd.merge(df1, query12, on=['node1', 'node2'])
    print(df1)
    
    query13 = graph.run("""match (m:disease{Name:'CD'})-[r]-(n:diet) where n.Name in ['corn/corn gluten','cheese (processed)', 'cheese (cottage)', 'chocolate', 'energy drink', 'nuts'] set r.cc=toInteger(1) 
    """).to_data_frame()
    query14 = graph.run("""match (m:disease{Name:'CD'})-[r]-(n:diet) where not n.Name in ['corn/corn gluten','cheese (processed)', 'cheese (cottage)', 'chocolate', 'energy drink', 'nuts'] set r.cc=toInteger(0) 
    """).to_data_frame()
    query15 = graph.run("""match (m:disease{Name:'CD'})-[r]-(n:diet) return m.Name as node1, n.Name as node2, r.cc as cc
    """).to_data_frame()    
    print("#################")
    df1=pd.merge(df1, query15, on=['node1', 'node2'])
    print(df1)
    
    
    sr= [0,0,0,0,0,0,0,1,1,1,
         1,0,1,0,0,0,0,0,0,0,
         0,0,1,0,0,1,0,0,0,0,
         0,0,0,0,0,0,0,0,0,0,
         0,0,0,0,0,0,0,0,0,0,
         0,0,0,0,0,0,0,0,0,0,
         0,0,0,0,0,0,0,0,0,0,
         0,0,0,0,0,0,0,0,0,0,
         0,0,0,0,0,0,0,0,0,0,
         0,0,0,0,0,0,0,0,0,0,
         0,0,0,0,0,0]
    
    df['cc']=sr
    #df1['cc']=se
    
    #print(df)
    print("##############################")
    #print(df1)
    df.to_csv('features1.csv')
    df1.to_csv('features2.csv')
    

    columns=["cooccurrence_sum","MaxTriangles","MinTriangles","MaxCoefficient","MinCoefficient","common_neighbors","preferential_attachment","total_neighbors","distance"]
    columns2=["sp","cc"]
    #columns1=["node1","node2", "label"]
    #columns3=["relation"]
    A=df1[columns]
    
    C=df[columns]
    
    f1=df[columns2]
    f2=df1[columns2]
    encoder = OneHotEncoder(sparse=False)
    # transform data
    onehot = encoder.fit_transform(f1)
    onehot2 = encoder.fit_transform(f2)

    min_max_scaler = preprocessing.MinMaxScaler()


    A = min_max_scaler.fit_transform(A)
    C = min_max_scaler.fit_transform(C)
    seed = 7

    A = np.append(A,onehot2,axis=1)
    C = np.append(C,onehot,axis=1)

    #print(A.shape)
    b=df1['relation']
    d=df['relation']
    #df_row = pd.concat([df, df2])
    #dataset.to_csv('E:\\PhD\\main_work\\ddd.csv')
    #df.to_csv('E:\\PhD\\main_work\\features.csv')


    #v=df.groupby(df['label']).count()
    print(d)


    stratified_kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=1)

    #counter = Counter(b)
    #print(counter)

    

    C_train, C_test, d_train, d_test = train_test_split(C,d,test_size=0.2,stratify=d,random_state=1)

    #m=SVC(random_state=1,probability=True)
    m=GaussianNB(var_smoothing=0.23101297000831597)
    pipeline = imbpipeline(steps = [['smote', SMOTE(random_state=1)],['classifier', m]])
    scores = cross_val_score(pipeline, C, d, scoring='accuracy', cv=stratified_kfold, n_jobs=-1)
    #print(scores)


    #predict1 = m.fit(C_train, d_train).predict(C_test)

    #print(metrics.classification_report(d_test, predict1))

    #print(metrics.confusion_matrix(d_test, predict1))

    predict2 = m.fit(C, d).predict(A)
    df1['prediction']=predict2
    #print(df1[columns1])
    #print(predict2)
    #print(metrics.classification_report(b, predict2))

    #print(metrics.confusion_matrix(b, predict2))


    lr_probs = m.fit(C, d).predict_proba(A)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    
    df1['probability']=lr_probs
    dr_harm=df1.loc[df1['prediction'] == 0]
    dr_help=df1.loc[df1['prediction'] == 1]
    #values=[]
    #for i in range(df1.shape[0]):
     #   if df1.iloc[i,14]>=0.9:
      #      values.insert(i,df1.iloc[i,14])


    #df1['prediction']=values
    #print(df1['prediction'])
    #lr = pd.DataFrame(values, columns = ['node1','node2','prediction'])
    dr_harm['probability']= (1-dr_harm['probability'])*100
    dr_help['probability']= (dr_help['probability'])*100
    
    dr_harm = dr_harm.sort_values(["probability", "node2"], ascending=False)
    dr_help = dr_help.sort_values(["probability", "node2"], ascending=False)
    
    fr_harm = dr_harm.rename({'node2': 'Diet/Food item', 'probability': 'Chances of being harmful (%)'}, axis=1)
    fr_help = dr_help.rename({'node2': 'Diet/Food item', 'probability': 'Chances of being helpful (%)'}, axis=1)
    col1=["Diet/Food item","Chances of being harmful (%)"]
    col2=["Diet/Food item","Chances of being helpful (%)"]
    
    #final=pd.merge(df1[col], lr, on=['node1', 'node2'])
    #ans=pd.concat(final)
    #print(ans)
    #col=["node1","node2","prediction"]
    #print(df1)
    #df1[col].to_csv('E:\\PhD\\main_work\\1results_phase1.csv')
    
    

# x-coordinates of left sides of bars
    left = [1, 2, 3, 4, 5]

# heights of bars
    height = [10, 24, 36, 40, 5]

# labels for bars
    tick_label = ['one', 'two', 'three', 'four', 'five']

# plotting a bar chart
    plt.bar(left, height, tick_label = tick_label,
		width = 0.4, color = ['red', 'green'])

# naming the x-axis
    plt.xlabel('x - axis')
# naming the y-axis
    plt.ylabel('y - axis')
# plot title
    plt.title('My bar chart!')
    diy = getcwd()
    filename = diy + '/templates/fig1.png'
    #plt.figure(figsize=(5, 3))
    plt.savefig(filename)
# function to show the plot
#plt.show()

    


    
    
    
    
    graph.run("""MATCH (n:disease{Name:$dis})-[r:linked_to]-() delete r""",dis=dis)
    graph.run("""MATCH (n:disease{Name:$dis}) delete n""",dis=dis)

    # Formatting in html
    #q2="perfect"
    #file = io.open("/home/ubuntu/disdiet/templates/rnn_index.html", "r", encoding='utf-8')
    #q=file.read()
    #html = '{% extends' + q + '%} {% block content %}'
   


    html='<div class="container">'
    html = addContent(html, header1('Harmful Diets for '+dis, color='black'))
    html = addContent(html, box1(fr_harm[col1].to_html(index=False)))
    html = addContent(html, bar())
    html = addContent(html, header2(
        'Helpful Diets for '+dis, color='black'))
    html = addContent(html, box(fr_help[col2].to_html(index=False)))
   
    return f'<div>{html}</div>'


def header1(text):
    """Create an HTML header"""

    raw_html = '<div class="a1">' + str(
            text) + '</div>'
    return raw_html
    
def header2(text):
    """Create an HTML header"""

    raw_html = '<div class="a2">' + str(
            text) + '</div>'
    return raw_html


def box1(text):
    """Create an HTML box of text"""
    raw_html = '<tr><td><div class="a2" style="border-bottom:1px inset black;border-top:1px inset black;padding:8px;font-size: 14px;float: left;">' + str(
            text) + '</div></td>'
    return raw_html

def box2(text):
    """Create an HTML box of text"""
    raw_html = '<tr><td><div class="a4" style="border-bottom:1px inset black;border-top:1px inset black;padding:8px;font-size: 14px;float: left;">' + str(
            text) + '</div></td>'
    return raw_html

def bar1():
    """Bar chart"""
    data_uri = base64.b64encode(open('/home/ubuntu/disdiet/templates/fig1.png', 'rb').read()).decode('utf-8')
    img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
    raw_html = '<div class="a3">' + img_tag + '</div>'
    return raw_html

def bar1():
    """Bar chart"""
    data_uri = base64.b64encode(open('/home/ubuntu/disdiet/templates/fig1.png', 'rb').read()).decode('utf-8')
    img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
    raw_html = '<div class="a5">' + img_tag + '</div>'
    return raw_html

def addContent(old_html, raw_html):
    """Add html content together"""

    old_html += raw_html
    return old_html
