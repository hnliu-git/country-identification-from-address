import time

import pytorch_lightning as pl
from evaluator import Evaluator
from models.rule_model import *
from models.stat_model import *
from preset import code2country
from torch.utils.data import DataLoader

if __name__ == '__main__':

    s_model_name = 'nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large'
    epochs = 10
    batch_size = 32

    train_data = TextDFData(sorted(code2country.keys()), s_model_name, 'train')
    val_data = TextDFData(sorted(code2country.keys()), s_model_name, 'val')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=TextDFData.collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=TextDFData.collate_fn)

    s_model = StatModel(sorted(code2country.keys()), s_model_name, train_data.steps_per_epoch(batch_size)*epochs)

    # ckpt_callback = ModelCheckpoint(
    #     dirpath=args.ckpt_path,
    #     monitor='val_loss',
    #     mode='min',
    #     filename="%s-{epoch:02d}-{val_loss:.2f}"
    #              % (args.model.split('/')[-1]),
    # )

    trainer = pl.Trainer(
        plugins=[HgCkptIO()],
        max_epochs=epochs,
        # logger=wandb_logger,
        # callbacks=[ckpt_callback]
    )

    trainer.fit(s_model, train_loader, val_loader)

    # r_model = RuleModel(code2country.keys())
    # evaluator = Evaluator(code2country.keys())
    #
    # # Evaluate metrics
    # evaluator.evaluate_metrics(r_model)
    #
    # # Evaluate latency
    # evaluator.evaluate_latency(r_model, n=100)
    #
    # s_model = StatModel(code2country.keys())