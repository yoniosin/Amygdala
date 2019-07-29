from Models.SingleTransform import BaseModel
from util.config import LearnerMetaData
from util.AmygDataSet import SequenceAmygDataset, GlobalAmygDataset
from Models.SingleTransform import SequenceTransformNet, STNet

if __name__ == '__main__':
    md = LearnerMetaData(batch_size=4,
                         train_ratio=0.9,
                        learning_rate=5e-1)

    model = BaseModel(md, ds_type=SequenceAmygDataset, net_type=SequenceTransformNet, allow_transition=False)
    model.train(100)
