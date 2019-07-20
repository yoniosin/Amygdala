from Models.SingleTransform import SingleTransform
from util.config import LearnerMetaData

if __name__ == '__main__':
    md = LearnerMetaData(batch_size=16,
                         train_ratio=0.9)
    model = SingleTransform(md)
    model.train(500)
