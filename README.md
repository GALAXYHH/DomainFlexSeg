# DomainFlexSeg: Domain Generalization for Breast Ultrasound Tumour Segmentation

This repository is the official implementation of **DomainFlexSeg: Domain Generalization for Breast Ultrasound Tumour Segmentation**.

# Requirements:
- Python 3.7
- Pytorch 1.7.0

# Datasets
Please download the dataset through [link](https://drive.google.com/file/d/1lhviQEuN537AzI6M5FNFuIBCK9AW2goG/view?usp=sharing). 

The project should be finally organized as follows:
```
./DomainFlexSeg/
  ├── data/
      ├── BUS_A/
      ├── BUS_B/
  ├── losses/
  ├── model/
  ├── dataset.py 
  ├── main.py
  ...... 
```

# Running
```
python main.py --dataset_name BUS_A --model MDFNet --img_size 320 --save_path ./model/
```
