import os
import sys
import json
from data_api import DataApi
from model_api import ExplanationModel, CounterFactualExplanationModel

data_api = DataApi()
explanation_model = ExplanationModel()
cf_explanation_model = CounterFactualExplanationModel()

class AvailableClassesResource():
    def get(self):
        return data_api.get_classes()

class CounterFactualResource():
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

        path = os.path.join(*image["path"].split("/")[2:])
        del image["path"]
        image["url"] = '/' + path

if __name__ == '__main__':
    if sys.argv[1] == 'classes':
        classResource = AvailableClassesResource()
        print(json.dumps(classResource.get()))
    elif sys.argv[1] == 'counterfactual':
        if len(sys.argv) == 4:
            counterFactualResource = CounterFactualResource()
            print(json.dumps(counterFactualResource.get(sys.argv[2], sys.argv[3])))
        else:
            print(json.dumps('Bad Parameters'))
    else:
        print(json.dumps('Unhandled Request'))
