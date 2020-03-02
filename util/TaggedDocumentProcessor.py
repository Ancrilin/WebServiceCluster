from gensim.models.doc2vec import TaggedDocument


def getTaggedDocument(name, text):
    return TaggedDocument(text, [name])