[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-360/)                                                                                       ![PyTorch Lightning](https://img.shields.io/badge/pytorch-lightning-blue.svg?logo=PyTorch%20Lightning)

# RetViT
Retinal Disease classification on ODIR-5K dataset based on pretrained Vision Transformer models. The ODIR dataset comprises 6 major retinal diseases such as Diabetic Retinopathy, Glaucoma, Cataract, Macular Degeneration, Hypertension, Myopia along with the classes Normal and Other Diseases. In this study `report.pdf` we focus on classifying 3 main ocular diseases and the Normal class. For this purpose different pretrained Vision Transformers such as ViT, BEiT, DeiT, LeViT and Swin are fine-tuned and compared. Also a classic ResNet-50 is trained, fine-tuned and compared with the Vision Transformers.  

# Methods
Data preprocessing includes the utilization of the preprocessed ODIR-5K retinal images, for which we remove the problematic cases and split them into separate eyes. The data are then split into training, testing and validation sets applying stratification. The csv files for the splits can be accessed from the `data/ODIR` folder. The dataloader will automatically load the images corresponding to each set from the `data/images` folder.

<p align="center">
  <img src="https://github.com/Xritsos/RetViT/assets/57326163/4e284369-4a3c-4ab7-8996-69e94d1efc67" width="600" height="300">  
</p>


# Results
Here all trained transformers are compared and finally the best one is selected. From the overall results and the scores on the most serious diseases (Diabetic Retinopathy & Cataract) the BEiT is considered the best. To consolidate our results attentions maps are generated utilizing the inherent feature (Attention layer) of the transformers.  

<p align="left">
  <figure>
    <img src="https://github.com/Xritsos/RetViT/assets/57326163/e998452a-dbcc-4aba-b6d6-7f7cf926c11d" width="200" height="200">
    <img src="https://github.com/Xritsos/RetViT/assets/57326163/b7243926-b297-4b95-aa6e-60d7c8b4e497" width="200" height="200">
    <img src="https://github.com/Xritsos/RetViT/assets/57326163/8dab6d93-63ce-4a15-9579-1ff7c5468520" width="200" height="200">
    <img src="https://github.com/Xritsos/RetViT/assets/57326163/36775049-9481-4a60-9f67-117be8935dfd" width="200" height="200">   
  </figure>
</p>  


(a) Diabetic Retinopathy & Pathological Myopia and its attention map,   (b) Cataract and its attention map.


## Requirements
The experiments were held on a linux server machine. Use of GPU is advised. All versions of the packages can be found on `requirements.txt`.

## Acknowledgements
The authors would like to acknowledge the National Institute of Health Data Science at Peking University (NIHDS-PKU), the Institute of Artificial Intelligence at Peking University (IAI-PKU), the Shanggong Medical Technology Co.Ltd. (SG) and the Advanced Institute of Information Technology at Peking University (AIIT-PKU) for co-organizing the International Competition on Ocular Disease Intelligent Recognition (ODIR) and providing the respective fundus image dataset [1].

Results presented in this work have been produced using the AUTH Compute Infrastructure and Resources. The authors would like to acknowledge the support provided by the Scientific Computing Office throughout the progress of this research work.

## References
[1] ODIR dataset: https://odir2019.grand-challenge.org/dataset/   

Contact:  
(1) gbotso@csd.auth.gr  
(2) tdrosog@csd.auth.gr  
(3) cpsyc@csd.auth.gr
