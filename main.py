from Models.SingleTransform import SingleTransform
from util.config import LearnerMetaData

if __name__ == '__main__':
    md = LearnerMetaData(batch_size=16)
    model = SingleTransform(md)
    model.train(500)
