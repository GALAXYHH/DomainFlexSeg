# DomainFlexSeg: Domain Generalization for Breast Ultrasound Tumour Segmentation

This repository is the official implementation of **DomainFlexSeg: Domain Generalization for Breast Ultrasound Tumour Segmentation**.

# Requirements:
- Python 3.7
- Pytorch 1.7.0
  
# Description:
Regarding the domain generalization issue of the following ultrasound images, we need to conduct training on DomainA-D and strive to improve the segmentation performance on DomainE:
<img width="1952" height="567" alt="image" src="https://github.com/user-attachments/assets/3e201430-8f05-4442-bf97-716a51fc7721" />

Among them, we mainly adopt the improved MDFNet backbone:
<img width="1336" height="1069" alt="image" src="https://github.com/user-attachments/assets/2d9ac41d-d10a-40f6-9cbb-6a4446b7dac3" />

The entire training network structure consists of MDFNet, FFT transformation, and multiple layers of Loss. The specific schematic diagram is as follows:
<img width="1986" height="657" alt="image" src="https://github.com/user-attachments/assets/f48ec108-3be4-44e4-8e93-0ddbc3ae55ef" />

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
