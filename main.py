import torch
from pytorch_lightning import Trainer
import json
from datamodule import CT_Datamodule
from model import Classifier

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
WEIGHT_DECAY = 0.0005
EPOCHS = 200


if __name__ == "__main__":
    results = {}
    data_module = CT_Datamodule("Dataset", batch_size=BATCH_SIZE, num_workers=5)
    data_module.prepare_data()

    for i in range(5):
        model = Classifier(lr_dino=1e-5, lr_class=1e-2, weight_decay=WEIGHT_DECAY, k=i)
        data_module.set_k(i)

        trainer = Trainer(
            accelerator="gpu",
            max_epochs=EPOCHS,
            devices=[0],
            log_every_n_steps=1,
        )

        trainer.fit(model, data_module)

        results[i] = trainer.test(model, data_module)[0]
        print(json.dumps(results, indent=4))

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
