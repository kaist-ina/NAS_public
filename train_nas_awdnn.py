from dataset import DatasetForDASH
from model import MultiNetwork
from option import opt
from trainer import Trainer
import template

model = MultiNetwork(template.get_nas_config(opt.quality))
dataset = DatasetForDASH(opt)
trainer = Trainer(opt, model, dataset)

for epoch in range(opt.num_epoch):
    trainer.train_one_epoch()
    trainer.validate()
    trainer.save_model()

    if epoch == (opt.num_epoch - 1):
        trainer.save_dnn_chunk()
