from py2neo import Graph
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import Pipeline as imbpipeline
import pandas as pd
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



def network_analysis(dis):
    
    #graph = Graph("https://15.207.24.149:7473", auth=("neo4j", "dilpreet"))
    graph = Graph("bolt://15.207.24.149:7687", auth=("neo4j", "dilpreet"))
    #query for creating graph for new node like crohns disease
    graph.run("""Load CSV with headers from "https://docs.google.com/spreadsheets/d/e/2PACX-1vRpVs0bzfJdlzwJFTx2TVQMJIoIsucHjqVTvnw9cL3BCftJUzrzEXNkC7M9vmyD2G51FnVqf6UaXBcy/pub?gid=0&single=true&output=csv" as line merge(n:disease{Name:line.disease}) merge (m:diet{Name:line.diet}) merge (n)-[r:linked_to{cooccurrence:toFloat(line.link)}]->(m)
""").to_data_frame()
    graph.run("""MATCH (n:disease{Name:$dis})-[r:linked_to]-()
              SET r.cooccurrence = toFloat(r.cooccurrence)""",dis=dis)
    existing_links1 = graph.run("""MATCH (m:disease)-[r:linked_to]->(n:diet) 
                                      where m.Name='IBD'
                                      RETURN m.Name as node1, n.Name as node2, 1 as label
                                      """).to_data_frame()
    existing_links2 = graph.run("""MATCH (m:disease)-[r:linked_to]->(n:diet) 
                                      where m.Name='UC' 
                                      RETURN m.Name as node1, n.Name as node2, 1 as label
                                      """).to_data_frame()
    pred_existing_links = graph.run("""MATCH (m:disease)-[r:linked_to]->(n:diet) 
                                    where m.Name=$dis
                                    RETURN m.Name as node1, n.Name as node2, 1 as label
                                    """,dis=dis).to_data_frame()
    pred_missing_links = graph.run("""match (m:disease{Name:$dis})-[r:linked_to]-(n:diet) 
                                    with collect(distinct n.Name) as t, m.Name as m
                                    match (a:diet) where not a.Name in t and 
                                    a.Name in ["bread", "wine", "carbonated beverage", "coffee", "energy drink", 
                                               "kefir", "whey", "soy milk", "tea (with milk)", "tea (without milk)", "spices (thyme)", "spices (ginger)", 
                                               "spices (tamarind)", "butter", "cheese (cottage)", "cheese (processed)", "yogurt", "margarine", 
                                               "rice (white)", "rice (brown)", "chocolate", "pumpkin", "plum", "germinated barley", "extra virgin olive oil", 
                                               "safflower oil", "sesame oil", "soybean oil", "egg (white)", "egg yolk", "fast food", "fruit pulp flour", 
                                               "frozen food", "honey", "manuka honey+sulfasalazine", "meat products/red meat", "fish", "white fish/ shellfish",
                                               "nuts", "apple", "fruit", "apple sauce/stewed", "banana", 
                                               "sugar beet", "blueberry", "cabbage", "carrot", "cornelian cherry", "coriander", "corn/corn gluten", 
                                               "cranberry", "red grapes/grape juice", "grape", "black pepper", "citrus", "orange ", 
                                               "mango", "lettuce", "pear", "milk", "tomato", "cooked potato", "boiled potato", 
                                               "beer", "vegetable (raw)", "vegetable (soft)", "mushroom", "oats", "oatmeal", "green pea (bland)", 
                                               "pineapple", "pistachio nut oil", "spinach (raw)", "spinach (cooked)", "spinach juice", "strawberry extract", "spices (turmeric)", "chewing gum"] 
                                    return distinct a.Name as node2, m as node1, 0 as label
                                """,dis=dis).to_data_frame()
    missing_links1 = graph.run("""match (m:disease{Name:'IBD'})-[r:linked_to]-(n:diet) 
                                     with collect(distinct n.Name) as t, m.Name as m
                                     match (a:diet) where not a.Name in t and 
                                     a.Name in ["bread", "wine", "carbonated beverage", "coffee", "energy drink", 
                                                "kefir", "whey", "soy milk", "tea (with milk)", "tea (without milk)", "spices (thyme)", "spices (ginger)", 
                                                "spices (tamarind)", "butter", "cheese (cottage)", "cheese (processed)", "yogurt", "margarine", 
                                                "rice (white)", "rice (brown)", "chocolate", "pumpkin", "plum", "germinated barley", "extra virgin olive oil", 
                                                "safflower oil", "sesame oil", "soybean oil", "egg (white)", "egg yolk", "fast food", "fruit pulp flour", 
                                                "frozen food", "honey", "manuka honey+sulfasalazine", "meat products/red meat", "fish", "white fish/ shellfish",
                                                "nuts", "apple", "fruit", "apple sauce/stewed", "banana", 
                                               "sugar beet", "blueberry", "cabbage", "carrot", "cornelian cherry", "coriander", "corn/corn gluten", 
                                               "cranberry", "red grapes/grape juice", "grape", "black pepper", "citrus", "orange ", 
                                               "mango", "lettuce", "pear", "milk", "tomato", "cooked potato", "boiled potato", 
                                               "beer", "vegetable (raw)", "vegetable (soft)", "mushroom", "oats", "oatmeal", "green pea (bland)", 
                                               "pineapple", "pistachio nut oil", "spinach (raw)", "spinach (cooked)", "spinach juice", "strawberry extract", "spices (turmeric)", "chewing gum"] 
                                     return distinct a.Name as node2, m as node1,  0 as label
                                """).to_data_frame()
    missing_links2 = graph.run("""match (m:disease{Name:'UC'})-[r:linked_to]-(n:diet) 
                                     with collect(distinct n.Name) as t, m.Name as m
                                     match (a:diet) where not a.Name in t and 
                                     a.Name in ["bread", "wine", "carbonated beverage", "coffee", "energy drink", 
                                                "kefir", "whey", "soy milk", "tea (with milk)", "tea (without milk)", "spices (thyme)", "spices (ginger)", 
                                                "spices (tamarind)", "butter", "cheese (cottage)", "cheese (processed)", "yogurt", "margarine", 
                                                "rice (white)", "rice (brown)", "chocolate", "pumpkin", "plum", "germinated barley", "extra virgin olive oil", 
                                                "safflower oil", "sesame oil", "soybean oil", "egg (white)", "egg yolk", "fast food", "fruit pulp flour", 
                                                "frozen food", "honey", "manuka honey+sulfasalazine", "meat products/red meat", "fish", "white fish/ shellfish",
                                                "nuts", "apple", "fruit", "apple sauce/stewed", "banana", 
                                               "sugar beet", "blueberry", "cabbage", "carrot", "cornelian cherry", "coriander", "corn/corn gluten", 
                                               "cranberry", "red grapes/grape juice", "grape", "black pepper", "citrus", "orange ", 
                                               "mango", "lettuce", "pear", "milk", "tomato", "cooked potato", "boiled potato", 
                                               "beer", "vegetable (raw)", "vegetable (soft)", "mushroom", "oats", "oatmeal", "green pea (bland)", 
                                               "pineapple", "pistachio nut oil", "spinach (raw)", "spinach (cooked)", "spinach juice", "strawberry extract", "spices (turmeric)", "chewing gum"] 
                                     return distinct a.Name as node2, m as node1,  0 as label
                                """).to_data_frame()


    pdList1 = [existing_links1,existing_links2,missing_links1,missing_links2] 
    pdList2 = [pred_existing_links,pred_missing_links] 
    df = pd.concat(pdList1)
    df1=pd.concat(pdList2)
    



    


    
    
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
    #print(df_new2)
    #df.to_csv('E:\\PhD\\main_work\\features.csv')
    

    columns=["cooccurrence_sum","MaxTriangles","MinTriangles","MaxCoefficient","MinCoefficient","common_neighbors","preferential_attachment","total_neighbors","distance"]
    columns2=["sp"]
    columns1=["node1","node2", "label"]
    A=df1[columns]
    b=df1['label']
    C=df[columns]
    d=df['label']
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


    #df_row = pd.concat([df, df2])
    #dataset.to_csv('E:\\PhD\\main_work\\ddd.csv')
    #df.to_csv('E:\\PhD\\main_work\\features.csv')


    v=df.groupby(df['label']).count()
   # print(v)


    stratified_kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=1)

    counter = Counter(b)
    #print(counter)

    

    C_train, C_test, d_train, d_test = train_test_split(C,d,test_size=0.2,stratify=d,random_state=1)

    m=SVC(random_state=1,probability=True)

    pipeline = imbpipeline(steps = [['smote', SMOTE(random_state=1)],['classifier', m]])
    scores = cross_val_score(pipeline, C, d, scoring='roc_auc', cv=stratified_kfold, n_jobs=-1)
    #print(scores)


    #predict1 = m.fit(C_train, d_train).predict(C_test)

    #print(metrics.classification_report(d_test, predict1))

    #print(metrics.confusion_matrix(d_test, predict1))

    predict2 = m.fit(C, d).predict(A)
    df1['pred']=predict2
    #print(df1[columns1])
    #print(predict2)
    #print(metrics.classification_report(b, predict2))

    #print(metrics.confusion_matrix(b, predict2))


    lr_probs = m.fit(C, d).predict_proba(A)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    
    df1['prediction']=lr_probs
    dr=df1.loc[df1['prediction'] >= 0.90]
    #values=[]
    #for i in range(df1.shape[0]):
     #   if df1.iloc[i,14]>=0.9:
      #      values.insert(i,df1.iloc[i,14])


    #df1['prediction']=values
    #print(df1['prediction'])
    #lr = pd.DataFrame(values, columns = ['node1','node2','prediction'])
    #col=["node1","node2","label","prediction","pred"]
    #final=pd.merge(df1[col], lr, on=['node1', 'node2'])
    #ans=pd.concat(final)
    #print(ans)
    #col=["node1","node2","prediction"]
    #print(df1)
    #df1[col].to_csv('E:\\PhD\\main_work\\1results_phase1.csv')

    graph.run("""MATCH (n:disease{Name:$dis})-[r:linked_to]-() delete r""",dis=dis)
    graph.run("""MATCH (n:disease{Name:$dis}) delete n""",dis=dis)

    # Formatting in html
    #q2="perfect"
    html = ''
    html = addContent(html, header(
        'Significant Diets', color='black'))
    html = addContent(html, box(dr['node2'].tolist()))
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



