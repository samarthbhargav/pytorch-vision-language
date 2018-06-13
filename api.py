from flask import Flask
from flask_restful import Resource, Api
from flask import url_for
from flask_cors import CORS
import os
from data_api import DataApi
from model_api import ExplanationModel, CounterFactualExplanationModel


app = Flask(__name__, static_folder="data")
api = Api(app)
CORS(app)

data_api = DataApi()
explanation_model = ExplanationModel("checkpoints/best-ckpt.pth")
cf_explanation_model = CounterFactualExplanationModel()

def any_response(data):
  ALLOWED = ['http://localhost:8888']
  response = make_response(data)
  origin = request.headers['Origin']
  if origin in ALLOWED:
    response.headers['Access-Control-Allow-Origin'] = origin
  return response

class AvailableClassesResource(Resource):
    def get(self):
        return data_api.get_classes()

class ExplanationResource(Resource):
    def get(self, image_id1, image_id2):
        image_id = "{}/{}".format(image_id1, image_id2)
        image = data_api.get_image(image_id)
        image["explanation"] = explanation_model.generate_explanation(image)
        return image

class SampleImagesResource(Resource):
    def get(self, n):
        return data_api.sample_images(n)

class CounterFactualResource(Resource):
    def get(self, class_true, class_false):
        true_image = data_api.sample_class(class_true)
        false_image = data_api.sample_class(class_false)
        self.fill_image(true_image, counter_factual=False)
        self.fill_image(false_image, counter_factual=True)

        return {
            "class_true": class_true,
            "class_false": class_false,
            "images": [
                true_image, false_image
            ]
        }
    
    def fill_image(self, image, counter_factual=False):
        image["cf_explanation"] = cf_explanation_model.generate_counterfactual_explanation(image)
        image["explanation"] = explanation_model.generate_explanation(image)

        #print(image["path"])
        path = os.path.join(*image["path"].split("/")[1:])
        #print(path)
        del image["path"]
        image["url"] = url_for('static', filename=path)

api.add_resource(AvailableClassesResource, '/classes')
api.add_resource(SampleImagesResource, '/sample_images/<int:n>')
api.add_resource(ExplanationResource, '/explain/<string:image_id1>/<string:image_id2>')
api.add_resource(CounterFactualResource, '/counter_factual/<string:class_true>/<string:class_false>')

if __name__ == '__main__':    
    app.run(debug=False)