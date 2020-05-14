from helper import global_variables as gv
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import flair
import torch
from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
from flair.embeddings import FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer


class SarcasmAnalyzer:
    def __init__(self, parent_dir_dataset: {str}, dataset_name: {str}, parent_dir_data: {str},
                 parent_dir_model: {str}, sentiment_model_dir: {str}, word_embeddings: {list} = None):

        self.parent_dir_dataset = parent_dir_dataset
        self.dataset_name = dataset_name
        self.parent_dir_data = parent_dir_data
        self.parent_dir_model = parent_dir_model
        self.sentiment_model_dir = sentiment_model_dir

        self.dataset_filepath = os.path.join(self.parent_dir_dataset, self.dataset_name)
        self.model_filepath = os.path.join(self.parent_dir_model, self.sentiment_model_dir)
        self.train_filename = os.path.join(self.parent_dir_data, "train.csv")
        self.test_filename = os.path.join(self.parent_dir_data, "test.csv")
        self.dev_filename = os.path.join(self.parent_dir_data, "dev.csv")
        self.column_name_map = {}
        if word_embeddings is None:
            self.word_embeddings = [FlairEmbeddings('news-forward'),
                                    FlairEmbeddings('news-backward')]
        else:
            self.word_embeddings = word_embeddings
        self.corpus = None
        self.document_RNNEmbeddings = None
        self.label = gv.label_sarcasm
        self.renamed_columns = gv.sarcasm_renamed_columns
        if gv.logger is None:
            gv.init_logger_object()

    def generate_datasets(self, df: {pd.DataFrame}, test_size: {float} = .3, dev_size: {float} = .2):
        df_train, df_test, df_dev = None, None, None
        try:
            gv.logger.info("Generating train, test and dev sets")
            df_train, df_test = train_test_split(df, test_size=test_size, stratify=df[self.label])
            df_train.to_csv(self.train_filename, index=False)

            # Split test dev
            df_test, df_dev = train_test_split(df_test, test_size=dev_size, stratify=df_test[self.label])
            df_test.to_csv(self.test_filename, index=False)
            df_dev.to_csv(self.dev_filename, index=False)
            gv.logger.info("Done")
        except Exception as e:
            gv.logger.error(e)
        return df_train, df_test, df_dev

    def prepare_original_dataset(self):
        df = None
        try:
            df = pd.read_json(self.dataset_filepath, lines=True)
            # Rename columns
            df.rename(columns=self.renamed_columns, inplace=True)
            # Drop columns
            drop_cols = [i for i in list(df.columns) if i not in list(self.renamed_columns.values())]
            df.drop(drop_cols, axis=1, inplace=True)
        except Exception as e:
            gv.logger.error(e)
        return df

    def prepare_data_for_flair_corpus(self, df: {pd.DataFrame}):
        try:
            columns = list(df.columns)
            n_cols = list(range(len(columns)))
            self.set_column_name_map(n_cols, columns)
        except Exception as e:
            gv.logger.error(e)

    def set_column_name_map(self, n_cols, columns):
        self.column_name_map = dict(zip(n_cols, columns))

    def get_column_name_map(self):
        return self.column_name_map

    def set_up_document_RNNEmbedding(self, hidden_size=512, reproject_words=True,
                                     reproject_words_dimension=256):
        self.document_RNNEmbeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings(
            self.word_embeddings, hidden_size=hidden_size, reproject_words=reproject_words,
            reproject_words_dimension=reproject_words_dimension,)

    def set_up_corpus(self):
        self.corpus: Corpus = CSVClassificationCorpus(self.parent_dir_data,
                                                      self.column_name_map,
                                                      skip_header=True,
                                                      delimiter=',')

    def get_document_RNNEmbedding(self):
        return self.document_RNNEmbeddings

    def get_corpus(self):
        return self.corpus

    def train_classifier_model(self, corpus: Corpus, document_embeddings: DocumentRNNEmbeddings,
                               model_params:{dict} = None):
        try:

            label_dict = corpus.make_label_dictionary()
            # create the text classifier
            classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)
            # initialize the text classifier trainer
            trainer = ModelTrainer(classifier, corpus)

            if model_params is None:
                learning_rate = gv.learning_rate
                mini_batch_size = gv.mini_batch_size
                anneal_factor = gv.anneal_factor
                patience = gv.patience
                max_epochs = gv.max_epochs
            else:
                learning_rate = model_params["learning_rate"]
                mini_batch_size = model_params["mini_batch_size"]
                anneal_factor = model_params["anneal_factor"]
                patience = model_params["patience"]
                max_epochs = model_params["max_epochs"]

            # start the training
            self.select_training_device()

            trainer.train(self.model_filepath,
                          learning_rate=learning_rate,
                          mini_batch_size=mini_batch_size,
                          anneal_factor=anneal_factor,
                          patience=patience,
                          max_epochs=max_epochs)
        except Exception as e:
            gv.logger.error(e)

    @staticmethod
    def select_training_device():
        try:
            # =========================================
            flair.device = None
            if torch.cuda.is_available():
                flair.device = torch.device('cuda:0')
            else:
                flair.device = torch.device('cpu')
            # =========================================

            gv.logger.warning(flair.device)
        except Exception as e:
            gv.logger.error(e)