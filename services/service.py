import os
from helper import global_variables as gv


class NLPService:
    def __init__(self, name: {str}, parent_model_dir: {str}, nlp_dir:{str}, model_name:{str},
                 classifier=None):
        self.name = name
        self.output = {"service_name": self.name,
                       "response": {}}

        self.parent_model_dir = parent_model_dir
        self.nlp_dir = os.path.join(self.parent_model_dir, nlp_dir)
        self.model_name = model_name
        self.model_full_path = os.path.join(self.nlp_dir, self.model_name)
        self.classifier = classifier

    def train_model(self, ):
        try:
            pass
        except Exception as e:
            gv.logger.error(e)

    def launch_analyzer(self, input_data: {str}):
        try:
            pass
        except Exception as e:
            gv.logger.error(e)