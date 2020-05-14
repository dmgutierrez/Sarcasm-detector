import multiprocessing
import sys
import os
if "research" in os.getcwd():
    sys.path.append('../')
    os.environ["GLOBAL_PARENT_DIR"] = ".."

else:
    os.environ["GLOBAL_PARENT_DIR"] = ".."

from analyzer.sarcasm_analyzer import SarcasmAnalyzer
from helper import global_variables as gv


def train_sarcasm_model(parent_dir_dataset: {str}, dataset_name: {str},
                        parent_dir_data: {str}, parent_dir_model: {str},
                        sentiment_model_dir: {str}, model_params: {dict},
                        hidden_size=512, reproject_words=True,
                        reproject_words_dimension=256):
    try:
        sarcasm_analyzer = SarcasmAnalyzer(parent_dir_dataset=parent_dir_dataset,
                                           dataset_name=dataset_name,
                                           parent_dir_data=parent_dir_data,
                                           parent_dir_model=parent_dir_model,
                                           sentiment_model_dir=sentiment_model_dir)

        # Prepare Original Dataset
        df_news = sarcasm_analyzer.prepare_original_dataset()
        sarcasm_analyzer.prepare_data_for_flair_corpus(df=df_news)

        # Generate Datasets for Flair
        df_train, df_test, df_dev = sarcasm_analyzer.generate_datasets(df=df_news)
        gv.logger.info("\n Train Shape: %s\n Test Shape: %s\n Dev Shape %s\n", str(df_train.shape),
                       str(df_test.shape), str(df_dev.shape))

        # Generate Corpus
        gv.logger.info("Setting up corpus and document embeddings")
        sarcasm_analyzer.set_up_corpus()
        # Generate document embedding
        sarcasm_analyzer.set_up_document_RNNEmbedding(hidden_size=hidden_size,
                                                      reproject_words=reproject_words,
                                                      reproject_words_dimension=reproject_words_dimension)
        corpus = sarcasm_analyzer.get_corpus()
        document_embeddings = sarcasm_analyzer.get_document_RNNEmbedding()
        gv.logger.info("Training model classifier")
        sarcasm_analyzer.train_classifier_model(corpus=corpus,
                                                document_embeddings=document_embeddings,
                                                model_params=model_params)
        gv.logger.info("The Model was trained with success!")
    except Exception as er:
        gv.logger.error(er)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    try:
        if gv.logger is None:
            gv.init_logger_object()

        # Params
        gv.logger.info("Setting up Parameters")
        global_parent_dir = os.getenv("GLOBAL_PARENT_DIR")
        parent_dir_dataset = os.path.join(global_parent_dir, "resources")
        dataset_name = "News-Headlines-Dataset.json"
        parent_dir_data = os.path.join(global_parent_dir, "data")
        parent_dir_model = os.path.join(global_parent_dir, "models")
        sentiment_model_dir = "sarcasm"
        model_params = {"learning_rate": .1, "mini_batch_size": 64,
                        "anneal_factor": .5, "patience": 5,
                        "max_epochs": 200}
        hidden_size = 512
        reproject_words = True
        reproject_words_dimension = 256
        train_sarcasm_model(parent_dir_dataset, dataset_name, parent_dir_data, parent_dir_model,
                            sentiment_model_dir, model_params, hidden_size,
                            reproject_words, reproject_words_dimension)

    except Exception as e:
        gv.logger.error(e)