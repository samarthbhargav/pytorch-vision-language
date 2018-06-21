import os
import io
import base64

from data_api import DataApi
from flask import Flask, url_for
from flask_cors import CORS
from flask_restful import Api, Resource
from model_api import ExplanationModel
from PIL import Image
from flask_restful import reqparse
from attribute_chunker import CounterFactualGenerator

app = Flask(__name__, static_folder="data")
api = Api(app)
CORS(app)

data_api = DataApi()
explanation_model = ExplanationModel()
cf_gen = CounterFactualGenerator()


def any_response(data):
    ALLOWED = ["http://localhost:8888"]
    response = make_response(data)
    origin = request.headers["Origin"]
    if origin in ALLOWED:
        response.headers["Access-Control-Allow-Origin"] = origin
    return response


def npimg2base64(img):
    pil_img = Image.fromarray(img)
    buff = io.BytesIO()
    pil_img.save(buff, format="JPEG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


class AvailableClassesResource(Resource):
    def get(self):
        return data_api.get_classes()


class ExplanationResource(Resource):
    def _parser(self):
        parser = reqparse.RequestParser()
        parser.add_argument("adversarial", type=bool, required=False, default=False)
        parser.add_argument("word_highlights", type=bool, required=False, default=False)
        return parser

    def get(self, image_id1, image_id2):
        image_id = "{}/{}".format(image_id1, image_id2)
        image = data_api.get_image(image_id)
        args = self._parser().parse_args()
        print(args)
        image["explanation"], np_image, word_masks = explanation_model.generate(
            image,
            word_highlights=args.word_highlights,
            adversarial=args.word_highlights,
        )
        image["image"] = npimg2base64(np_image)

        if args.word_highlights:
            image["word_highlights"] = [
                {"word": word, "position": pos, "mask": npimg2base64(mask)}
                for ((pos, word), mask) in word_masks.items()
            ]
        return image


class SampleImagesResource(Resource):
    def get(self, n):
        return data_api.sample_images(n)


class CounterFactualResource(Resource):
    def _parser(self):
        parser = reqparse.RequestParser()
        parser.add_argument("cf_limit", type=int, required=False, default=3)
        return parser

    def to_chunks(self, attr):
        return ["{} {}".format(a.description, a.attribute) for a in attr]

    def get(self, class_true, class_false):
        true_image = data_api.sample_class(class_true)
        false_image = data_api.sample_class(class_false)
        self.fill_image(true_image)
        self.fill_image(false_image)
        args = self._parser().parse_args()

        cf_expl, added_other, added = cf_gen.generate(
            true_image["explanation"], false_image["explanation"], addtn_limit=args.cf_limit
        )

        return {
            "class_true": class_true,
            "class_false": class_false,
            "images": [true_image, false_image],
            "cf_explanation": cf_expl
        }

    def fill_image(self, image):
        image["explanation"], _, _ = explanation_model.generate(
            image, word_highlights=False, adversarial=False
        )

        # print(image["path"])
        path = os.path.join(*image["path"].split("/")[1:])
        # print(path)
        del image["path"]
        image["url"] = url_for("static", filename=path)


api.add_resource(AvailableClassesResource, "/classes")
api.add_resource(SampleImagesResource, "/sample_images/<int:n>")
api.add_resource(ExplanationResource, "/explain/<string:image_id1>/<string:image_id2>")
api.add_resource(
    CounterFactualResource, "/counter_factual/<string:class_true>/<string:class_false>"
)

if __name__ == "__main__":
    app.run(debug=False)
