import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import utils, config
from models.models import ViTLightningModule, BEiT, LeViT, DeiT, ResNet50, VIT

seed_value = 42
torch.manual_seed(seed_value)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=config.early_stopping_patience,
    strict=False,
    verbose=False,
    mode='min'
)

save = ModelCheckpoint(
    save_top_k=1,
    mode='min',
    monitor='val_loss'
)


csv_logger = CSVLogger(save_dir='../logs/', name='experiment_name', flush_logs_every_n_steps=1)

if config.model_processor == 'SWIN':
    model = ViTLightningModule()
elif config.model_processor == 'BEiT':
    model = BEiT()
elif config.model_processor == 'LeViT':
    model = LeViT()
elif config.model_processor == 'DeiT':
    model = DeiT()
elif config.model_processor == 'ResNet':
    model = ResNet50()
elif config.model_processor == 'VIT':
    model = VIT()


trainer = Trainer(accelerator='gpu', max_epochs=config.num_epochs,
                  callbacks=[early_stop_callback, save], logger=csv_logger, log_every_n_steps=5)

trainer.fit(model)

print("================ Testing... ======================")
trainer.test(model, ckpt_path='best')
print("==================================================")


utils.plot_results()
