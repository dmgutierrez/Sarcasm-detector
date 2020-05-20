from services.service import NLPService
from helper import global_variables as gv
from flair.data import Sentence
from flair.models import TextClassifier


class SarcasmService(NLPService):
    def __init__(self, name: {str}, parent_model_dir: {str}, nlp_dir: {str}, model_name:{str},
                 classifier=None):

        NLPService.__init__(self, name=name, parent_model_dir=parent_model_dir,
                            nlp_dir=nlp_dir, model_name=model_name,
                            classifier=classifier)
        self.classifier = TextClassifier.load(self.model_full_path)

    def launch_analyzer(self, input_data: {str}):
        response = {"value": "Not Available", "confidence": 0, "sentence": '"' + input_data + '"'}
        try:
            gv.logger.info("Analysing the sentence ... ")
            sentence = Sentence(input_data)
            # predict class and print
            self.classifier.predict(sentence)
            res = sentence.labels[0].to_dict()

            # Process response
            response["confidence"] = round(100*res["confidence"], 2)
            if int(res["value"]) == 1:
                value = "This sentence seems to be sarcastic!"
            else:
                value = "This sentence does not seem to be sarcastic!"
            response["value"] = value

            # Store response in output
            self.output["response"] = response
        except Exception as e:
            gv.logger.error(e)
        return response