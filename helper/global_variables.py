from helper.custom_log import init_logger
from helper import config as cfg
from services.sarcasm_service import SarcasmService

host = cfg.host
port = cfg.port

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