import os
from helper.custom_log import init_logger
from services.sarcasm_service import SarcasmService

host = os.getenv("HOST_PORT") if "HOST_PORT" in os.environ else "localhost"
port = int(os.getenv("API_PORT")) if "API_PORT" in os.environ else 5005

# ========================================================
models_dir = "models"

# Sarcasm Model
sarcasm_dir = "sarcasm"
sarcasm_model = "best-model.pt"
label_sarcasm = "label_sarcastic"
sarcasm_service_name = "Sarcasm Detector Service"
sarcasm_renamed_columns = {'headline': 'text', 'is_sarcastic': 'label_sarcastic'}
learning_rate = 0.1
mini_batch_size = 64
anneal_factor = 0.5
patience = 5
max_epochs = 200

# =========================================================
logger = None
sarcasm_service = None


def init_logger_object():
    global logger
    logger = init_logger(__name__, testing_mode=False)


def init_service():
    global sarcasm_service
    sarcasm_service = SarcasmService(name=sarcasm_service_name, parent_model_dir=models_dir,
                                     nlp_dir=sarcasm_dir, model_name=sarcasm_model,
                                     classifier=None)