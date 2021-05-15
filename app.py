from flask import Flask, request, json,jsonify
import pickle
import numpy




app = Flask(__name__)
rdf_model = pickle.load(open('rdf_model.pkl', 'rb'))
nb_model = pickle.load(open('nb_model.pkl', 'rb'))
knn_model = pickle.load(open('knn_model.pkl', 'rb'))
le = pickle.load(open('le.pkl', 'rb'))
tfvec = pickle.load(open('tfvec.pkl', 'rb'))







@app.route('/',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    get_symp = json.loads(request.data)
    #print(get_symp)
    int_features = []
    for k in get_symp.keys():
        if get_symp[k] != 'None':
            int_features.append(get_symp[k])


    int_features = [' '.join(int_features)]
    #print(int_features)
    rdf = le.inverse_transform(rdf_model.predict(tfvec.transform(int_features).toarray()))
    nb = le.inverse_transform(nb_model.predict(tfvec.transform(int_features).toarray()))
    knn = le.inverse_transform(knn_model.predict(tfvec.transform(int_features).toarray()))

    output = {}

    output["model_rdf"] = rdf[0]
    output["model_nb"] = nb[0]
    output["model_knn"] = knn[0]
    #print(output)


    return output







if __name__ == "__main__":
    app.run(debug=True)