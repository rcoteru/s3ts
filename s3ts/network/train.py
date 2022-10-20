from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
import torch

from sklearn.metrics import classification_report

import numpy as np
import tqdm

from s3ts.network.architecture import CNN_DTW, model_wrapper
from s3ts.datasets.modules import dtwDataModule

seed_everything(0, workers=True)

def train_network(
        dataset_name: str, 
        mode: str, 
        window_size: int, 
        batch_size: int, 
        learning_rate: float, 
        max_epochs: int, 
        num_workers: int, 
        PATH: str
        ):

    root_dir = f"data/{dataset_name}/DTW"
    datamodule = dtwDataModule
    arch_model = CNN_DTW

    root_dir = f"{root_dir}/ws_{window_size}"

    AVAIL_GPUS = max(0, torch.cuda.device_count())

    dataMod = datamodule(f'data/{dataset_name}', window_size=window_size, num_workers=num_workers, batch_size=batch_size)
    dataMod.prepare_data()

    model = model_wrapper(model_architecture=arch_model,
                          ref_size=dataMod.dtw_test.dtws.shape[1] if mode == 'dtw' else None,
                          channels=dataMod.channels,
                          labels=dataMod.labels_size,
                          window_size=window_size,
                          lr=learning_rate)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_checkpoint = ModelCheckpoint(f"{root_dir}/checkpoints")


    # Initialize a trainer
    trainer = Trainer(
        default_root_dir=root_dir,
        callbacks=[lr_monitor, model_checkpoint],
        max_epochs=max_epochs,
        check_val_every_n_epoch=1,
        #progress_bar_refresh_rate=30,
        deterministic=True,
    )

    if PATH is None:
        # Train the model ⚡
        trainer.fit(model, datamodule=dataMod)
        trainer.validate(model, datamodule=dataMod)
        trainer.test(model, datamodule=dataMod)

        path = f"{root_dir}/checkpoints/" \
               f"epoch={trainer.current_epoch -1}-step={trainer.global_step}.ckpt"

    else:
        path = PATH

    model = model_wrapper.load_from_checkpoint(path,
                                               model_architecture=arch_model,
                                               ref_size=dataMod.dtw_test.dtws.shape[1] if mode == 'dtw' else None,
                                               channels=dataMod.channels,
                                               labels=dataMod.labels_size,
                                               window_size=window_size)

    model.eval()
    model.freeze()
    #model.cuda()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def evaluate_trained_network(dataMod, batch_size, model, root_dir):

    total_len = len(dataMod.dtw_test)

    y_pred = []
    y_true = []
    predict_dataloader = dataMod.test_dataloader()

    with torch.inference_mode():
        for i, (x, y) in tqdm(enumerate(predict_dataloader), total=total_len // batch_size):
            #x = x.cuda()
            raw_score = model(x)
            y_pred.extend(raw_score.softmax(dim=-1).cpu().numpy())
            y_true.extend(y.cpu().numpy())
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    print('Classification Report')
    target_names = [str(i) for i in range(dataMod.labels_size)]
    print(classification_report(y_true, np.argmax(y_pred, axis=-1)))
    save_path = f"{root_dir}/net_results/"
    # os.makedirs(save_path, exist_ok=True)

    # plot_roc_auc(dataMod.labels_size,
    #              F.one_hot(torch.tensor(y_true), num_classes=dataMod.labels_size).numpy(), y_pred,
    #              save_path)

    # plot_confusion_matrix(y_true, y_pred, target_names, save_path)
