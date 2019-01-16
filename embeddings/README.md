# Word embeddings

There may be tens of thousands of words in a corpus.  If we can embed
these words into a vector space of, say 200 or 300 words, then not
only can we surmount the enormous computations inefficiency we would
have from one-hot encoding, we can use it to look for neighbors of
words within the embedding space.  The embeddings can also be projected
to a further smaller space using PCA or t-SNE, which is useful for
visualization.

Some common embeddings are `word2vec` and `GloVe`. Skip-gram and
Continuous Bag of Words (CBOW) are two common ways to train word2vec.
Here, I'm including a notebook showing the word2vec embedding using
Skip-grams.
