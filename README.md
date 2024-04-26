# SiameseMSDiagnosis - Diagnosis of multiple sclerosis by detecting asymmetry within the retina using a similarity-based neural network

<br>

![Aggregated](https://github.com/regybean/SiameseMSDiagnosis/blob/main/imgs/aggregated_diagram.png)
Aggregated Siamese network architecture

![Contrastive](https://github.com/regybean/SiameseMSDiagnosis/blob/main/imgs/contrastive_diagram.png)
Contrastive Siamese network architecture

## Method

-"Multiple sclerosis (MS) is a chronic neurological disorder that targets the central nervous system, causing demyelination and neural disruption, which can include retinal nerve damage leading to visual disturbances. The purpose of this study is to demonstrate the capability to automatically diagnose MS by detecting asymmetry within the retina, using a similarity-based neural network, trained on optical coherence tomography images. This work aims to investigate the feasibility of a learning-based system accurately detecting the presence of MS, based on information from pairs of left and right retina images. We also justify the effectiveness of a Siamese Neural Network for our task and present its strengths through experimental evaluation of the approach. We train a Siamese neural network to detect MS and assess its performance using a test dataset from the same distribution as well as an out-of-distribution dataset, which simulates an external dataset captured under different environmental conditions. Our experimental results demonstrate that a Siamese neural network can attain accuracy levels of up to 0.932 using both an in-distribution test dataset and a simulated external dataset. Our model can detect MS more accurately than standard neural network architectures, demonstrating its feasibility in medical applications for the early, cost-effective detection of MS."

[[Bolton, Atapour-Arbarghouei et al, TBD, 2024]()]

---

## Train

* Clone the repository:

```
$ git clone https://github.com/regybean/SiameseMSDiagnosis.git
$ cd SiameseMSDiagnosis
```
* A dataset must be used for training. In our experiments, we use the [IDK]() dataset. Any similar dataset with the same structure according to the custom training dataset classes in (datasets.py) will work. Feel free to modify the scipt or dataset class to fit your own purposes. Our custom dataset follows the following directory structure:

```
OCTdata
├── train
│   ├── MStrain
│   │   ├── 1.OD.png
│   │   ├── 1.OS.png
│   │   ├── 2.OD.png
│   │   ├── 2.OS.png
│   │   ├──    ...
│   ├── Normaltrain
│   │   ├── 1.OD.png
│   │   ├── 1.OS.png
│   │   ├── 2.OD.png
│   │   ├── 2.OS.png
│   │   ├──    ...
├── test
│   ├── MStest
│   │   ├── 46.OD.png
│   │   ├── 46.OS.png
│   │   ├── 47.OD.png
│   │   ├── 47.OS.png
│   │   ├──    ...
│   ├── Normaltest
│   │   ├── 13.OD.png
│   │   ├── 13.OS.png
│   │   ├── 14.OD.png
│   │   ├── 14.OS.png
│   │   ├──    ...
```

* To train the model, run the following command:

```
$ python train.py --model=model --dataset_path=./path/to/data
```

* A full list of arguments can be found in the respective python files

* Files are saved in the \models directory as models_timestamp_.pt

---
## Test

* In order to test the model, run the following command:

```
python test.py --model_path=yourmodel.pt
```

## Reference
[Diagnosis of multiple sclerosis by detecting asymmetry within the retina using a similarity-based neural network]()
(authors)
```
@InProceedings{,
  author = {},
  title = {},
  year = {2024},
  publisher = {IEEE}
}
```



