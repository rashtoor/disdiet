from phase3 import network_analysis

from wtforms import Form, TextField, validators, SubmitField

import os
from json import dumps
import logging

from flask import Flask, g, Response, request, render_template
from neo4j import GraphDatabase, basic_auth

# Create app
app = Flask(__name__, static_url_path='/static/')

password = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver('bolt://15.207.24.149:7687',auth=basic_auth("neo4j", "dilpreet"))


def get_db():
    if not hasattr(g, 'neo4j_db'):
        g.neo4j_db = driver.session()
    return g.neo4j_db


@app.teardown_appcontext
def close_db(error):
    if hasattr(g, 'neo4j_db'):
        g.neo4j_db.close()

def serialize_disease(disease):
    return {
        'name': disease['Name'],

    }


def serialize_cast(diet):
    return {
        'name': diet[1],
        
    }


class ReusableForm(Form):
    """User entry form for entering specifics for generation"""
    # Starting seed
    seed = TextField("Enter a disease:", validators=[
                     validators.InputRequired()])
   
    submit = SubmitField("Enter")


#def load_keras_model():
 #   """Load in the pre-trained model"""
  #  global model
   # model = load_model('../models/train-embeddings-rnn.h5')
    # Required for model to work
    #global graph
    #graph = tf.get_default_graph()


# Home page
@app.route("/", methods=['GET', 'POST'])
def home():
    """Home page of app with form"""
    # Create form
    form = ReusableForm(request.form)

    # On form entry and all conditions met
    if request.method == 'POST' and form.validate():
        # Extract information
        seed = request.form['seed']
        
        # seed is the disease that person has selected, so now we read csv, create its graph and apply all network analysis and return significant diets using a function
        if seed != ' ':
            return render_template('random.html',input=network_analysis(dis=seed))
        else:
            return render_template('random.html')
            # Send template information to index.html
    return render_template('rnn_index.html', form=form)

@app.route("/graph")
def get_graph():
    db = get_db()
    results = db.run("MATCH (m:disease)-[:linked_to]-(a:diet) "
                                                         "RETURN m.Name as disease, collect(a.Name) as diet ")
                                                         
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
   
    return Response(dumps({"nodes": nodes, "links": rels}),
                    mimetype="application/json")



if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    #load_keras_model()
    # Run app
    app.run(host="0.0.0.0", debug=True, port=8002)
    #app.run(port=8002)
