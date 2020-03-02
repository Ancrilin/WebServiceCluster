from gensim.models.doc2vec import TaggedDocument
import time
import gensim
import numpy as np


class Doc2vec():
    def __init__(self, min_count=1, vector_size=50, window=8, sample=1e-5):
        self.vector_size = vector_size
        self.min_count = min_count
        self.window = window
        self.sample = sample

    def _train(self, name, text, save_path, n_epoch = 128):
        self.model = gensim.models.Doc2Vec(min_count=self.min_count, window = self.window,
                                           vector_size = self.vector_size, sample=self.sample, negative=5, workers=4)
        # self.model.build_vocab(x_train)                 #建立词汇表
        assert len(name) == len(text)
        X = []
        for i in range(len(name)):
            X.append(self._getTaggedDocument(name[i], text[i]))
        self.model.build_vocab(X)
        self.model.train(X, total_examples=self.model.corpus_count, epochs=n_epoch)
        #保存模型
        if save_path != None:
            self.model.save(save_path + '/Doc2vec' + '.model')
        return self.model

    def __call__(self, name, text, n_epoch, save_path=None):
        return self._train(name, text, save_path=save_path, n_epoch=n_epoch)

    def _getTaggedDocument(self, name, text):
        return TaggedDocument(text, [name])

    def get_vector(self, name):
        return self.model.docvecs(name)

if __name__ == '__main__':
    doc2vec = Doc2vec()
