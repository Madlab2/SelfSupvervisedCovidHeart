## Self-Supervised Pre-Training for Covid-19 Heart Pathology Identification

Repository for the self-supervised learning project in DTU course "Deep Learning for Experimental 3D Data".

### How to run on GPU:
1. Clone this repository and navigate to root folder
2. Install the requirements to your conda environment/python distribution (in requirements/requirements.txt)
3. Make sure to adjust the paths for datasets, models etc. in the config.py, pretrain.py, train_segmentation.py and inference.py. Also you will need to adjust the config file remote/local logic if you don't run on the DTU cluster.
4. In the terminal, log into your wandb account.
```bash
$ wandb login
```
5. Train the baseline model: 
```bash
$ python src/train_segmentation.py train_baseline
```
6. Conduct the pretrainint:
```bash
$ python src/pretrain.py
```
7. Train the model with pretraining (make sure to specify the path to your preferred pretrained model in config.py)
```bash
$ python src/train_segmentation.py train_with_pretrain
```
8. Run inference (adapt inference.py depending on whether you run the baseline or pre-trained version). Make sure to adjust the paths
```bash
$ python src/inference.py
```

Declaration: Much of the code in this repository has been taken from DTU course "Deep Learning for Experimental 3D Data" and then adapted.