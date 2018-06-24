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

app = Flask(__name__)
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
    return "data:image/jpeg;base64," + base64.b64encode(buff.getvalue()).decode("utf-8")


class AvailableClassesResource(Resource):
    def get(self):
        return data_api.get_classes()


class ExplanationResource(Resource):
    def _parser(self):
        parser = reqparse.RequestParser()
        parser.add_argument("word_highlights", type=bool, required=False, default=False)
        return parser

    def get(self, image_id1, image_id2):
        image_id = "{}/{}".format(image_id1, image_id2)
        image = data_api.get_image(image_id)
        args = self._parser().parse_args()
        print(args)
        image["explanation"], np_image, word_masks = explanation_model.generate(
            image, word_highlights=args.word_highlights
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
        images = data_api.sample_images(n)
        for image in images:
            image["image"] = npimg2base64(explanation_model.get_img(image["id"]))
        return images


class AttackOfTheClones(Resource):
    def _parser(self):
        parser = reqparse.RequestParser()
        parser.add_argument("word_index", type=int, required=False, default=-1)
        parser.add_argument("epsilon", type=float, required=False, default=0.1)
        return parser

    def get(self, image_id1, image_id2):
        image_id = "{}/{}".format(image_id1, image_id2)
        args = self._parser().parse_args()
        args.word_index = None if args.word_index == -1 else args.word_index
        explanation, x_org, explanation_adv, x_adv = explanation_model.generate_adversarial(
            image_id, epsilon=args.epsilon, word_index=args.word_index
        )

        print(x_org.shape, x_adv.shape)

        explanation_attrs = explanation_model.chunker.chunk(explanation)
        explanation_adv_attrs = explanation_model.chunker.chunk(explanation_adv)

        return {
            "explanation": explanation,
            "adv_explanation": explanation_adv,
            "image": npimg2base64(x_org),
            "adv_image": npimg2base64(x_adv),
            "word_highlights": [
                {
                    "word": a.attribute,
                    "position": a.position,
                    "mask": npimg2base64(x_org),
                }
                for a in explanation_attrs
            ],
            "adv_word_highlights": [
                {
                    "word": a.attribute,
                    "position": a.position,
                    "mask": npimg2base64(x_adv),
                }
                for a in explanation_adv_attrs
            ],
        }


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
            false_image["explanation"],
            true_image["explanation"],
            addtn_limit=args.cf_limit,
        )

        # if no attributes were added, then 
        # just put the explanation of the false image
        if (len(added_other) + len(added)) == 0:
            cf_expl = false_image["explanation"]

        cf_chunks = cf_gen.ch.chunk(cf_expl)
        cf_chunks_form = []
        for attr in cf_chunks:
            cf_chunks_form.append(
                {
                    "word": attr.attribute,
                    "position": attr.position,
                    "mask": true_image["image"],
                }
            )

        return {
            "class_true": class_true,
            "class_false": class_false,
            "images": [true_image, false_image],
            "cf_explanation": cf_expl,
            "cf_attributes": cf_chunks_form,
        }

    def fill_image(self, image):
        image["explanation"], _, _ = explanation_model.generate(
            image, word_highlights=False
        )
        image["image"] = npimg2base64(explanation_model.get_img(image["id"]))


api.add_resource(AvailableClassesResource, "/classes")
api.add_resource(SampleImagesResource, "/sample_images/<int:n>")
api.add_resource(ExplanationResource, "/explain/<string:image_id1>/<string:image_id2>")
api.add_resource(
    CounterFactualResource, "/counter_factual/<string:class_true>/<string:class_false>"
)
api.add_resource(AttackOfTheClones, "/attack/<string:image_id1>/<string:image_id2>")

if __name__ == "__main__":
    app.run(debug=False)
