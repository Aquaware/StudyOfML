import numpy as np

class ImageClassPredictor():

    def __init__(self, class_index):
        self.class_index = class_index

    def predict(self, net_out):
        max_id = np.argmax(net_out.detach().numpy())
        label_name = self.class_index[str(max_id)][1]
        return label_name


