# keras-old-metrics
Old Keras metrics with f-score. Does not work completely correctly (that's why it was removed from Keras).

To change the existing file:

sudo cp metrics.py /usr/local/lib/python2.7/dist-packages/keras/

Use example:

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy", "fscore"])

