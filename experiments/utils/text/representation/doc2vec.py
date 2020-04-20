from ... import functions
from ... import text_processing

import argparse
import sys
import os
import csv
import time
import numpy as np 
import pandas as pd

import multiprocessing as mp
from gensim.models.doc2vec import Doc2Vec as Gensim_Doc2Vec
from gensim.models.doc2vec import TaggedDocument

class Doc2Vec:
    def __init__(self, args, process=None):
        params = ["vector_size", "window", "min_count", "max_vocab_size"]
        self.args = pd.Series([vars(args)[param] for param in params], index=params)
        self.process = process
    
    @property
    def params(self):
        return self.args.to_dict()
        
    def initialize(self, data, tokenized=False):
        # data = [add_df1,add_df2,add_df3,actual_df]
        # If tokenized, then a since list of tokenized documents

        if tokenized:
            tokenized_texts = data
        else:
            processor = text_processing.Preprocessing(removeStopwords=True)
            tokenized_texts = processor.Bulk_Tokenizer(functions.flatten(data))    

        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(tokenized_texts)]
        self.print(f"Training Doc2Vec with {len(documents)} Texts ")

        self.model = Gensim_Doc2Vec(
                                        documents, 
                                        vector_size     = self.args.vector_size, 
                                        window          = self.args.window, 
                                        min_count       = self.args.min_count, 
                                        max_vocab_size  = self.args.max_vocab_size, 
                                        workers         = mp.cpu_count()
                                    )
        return self

    def generate_embedding(self, data, retrain_epcohs=1, returnarray=True):
        processor = text_processing.Preprocessing(removeStopwords=True)
        tokenized_texts = processor.Bulk_Tokenizer(list(data))

        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(tokenized_texts)]
        self.model.build_vocab(documents, update=True)
        self.model.train(documents, total_examples=len(documents), epochs=retrain_epcohs)

        embeddings = []
        for text in tokenized_texts:
            embeddings.append(self.model.infer_vector(text))    

        if returnarray:
            embeddings = np.asarray(embeddings)
            return embeddings
        else:
            return [list(each) for each in embeddings]

    def load_model(self, model_file):
        self.model = Gensim_Doc2Vec.load(model_file)

    def save(self, path):
        self.model.save(path)


    def print(self,*message):
        if self.process:           
            self.process.print(*message)
        else:
            print(*message)

    # ALIASES
    train = initialize
    infer_vector = generate_embedding

if __name__ == "__main__":
    from ... import dataset
    