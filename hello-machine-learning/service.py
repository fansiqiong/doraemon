
# route register, handle the client's request
from flask import Flask, request
import numpy as np
import dill as pickle
app = Flask(__name__)
@app.route('/iris/classify', methods=['POST'])
def IrisClassify():
	clf = 'iris_knn_model.pk'
	with open(clf,'rb') as f:
		loaded_model = pickle.load(f)
		req = np.array([[request.get_json()['sepal-length'], request.get_json()['sepal-width'],request.get_json()['petal-length'],request.get_json()['petal-width']]])
		result = loaded_model.predict(req)
		response = dict({"class": str(result[0])})
		return response

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=int(5000))

# test demo: curl -X POST http://localhost:5000/iris/classify -H "Content-Type: application/json" -d '{"sepal-length":1, "sepal-width":2, "petal-length":3, "petal-width":4}'
