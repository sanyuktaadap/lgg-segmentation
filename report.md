## Segmenting Low-Grade Glioma from Brain MRI Images

**Introduction** :

Low-grade gliomas (LGG) are benign tumors in the brain that develop from the glial cells that support and nourish the brain's neurons. Depending on how their cells appear under a microscope, glial tumors, or gliomas, are categorized into four grades. According to the WHO classification, LGGs are grade 1 and 2 tumors. The cause of these tumors may be genetic abnormalities or environmental factors. Different presentations occur depending on the location and how large the tumor is (StatPearls, 2023; Boston Children's Hospital, n.d.).

LGGs are the most common tumors of the central nervous system (CNS) in children. They account for 30% of all pediatric CNS tumors. Children with low-grade gliomas have excellent survival rates, with a 10-year survival rate of more than 85%. Individual outcomes vary greatly depending on the specific type, location, grade, age at diagnosis, and whether the tumor is freshly diagnosed or has reappeared (recurred) (St. Jude Children's Research Hospital, n.d.).

Magnetic Resonance Imaging (MRI) stands out as the most highly sensitive medical imaging technique for detecting all types of brain tumors. It holds a vital role in the identification, treatment, and post-treatment monitoring of gliomas. Within MRIs, low-grade gliomas typically manifest as compact lesions devoid of contrast enhancement (Figure 1) (Haydar et al., 2022).

![image](https://github.com/sanyuktaadap/lgg-segmentation/assets/126644146/b4e9b5c2-74d4-4d19-90a2-ca55c9cb07c0)


**Description** :

Compared to high-grade gliomas (HGG), LGGs are often slow-growing tumors. According to research using serial MRI images, before treatment, these lesions normally increase consistently at a rate of 4.1 mm every year. When compared to those which are more aggressive kinds, LGGs have a reasonably longer period of survival. However, more than 70% of these have the potential to turn into HGG or change their behaviour into aggression within ten years (StatPearls, 2023; Boston Children's Hospital, n.d.).

Thus, early diagnosis of LGGs can help in beginning the treatment before it turns malignant and converts to HGGs.

**Dataset** :

For this study we used the [Brain MRI segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation/data) dataset retrieved from the Kaggle repository. The images in the dataset were obtained from [The Cancer Imaging Archive](https://wiki.cancerimagingarchive.net/display/Public/TCGA-LGG) (TCIA). The dataset consists of 3929 Brain MRI images and corresponding manual fluid-attenuated inversion recovery (FLAIR) segmentation masks as the ground truth. The MRI images are obtained from 110 patients involved in The Cancer Genome Atlas (TCGA) lower-grade glioma collection. The images are in a .tif format with 3 channels (RGB). They are organized into 110 folders representing 110 patients, each consisting of brain scans from individual patients and the corresponding segmented FLAIR mask (Figure 2).

![image](https://github.com/sanyuktaadap/lgg-segmentation/assets/126644146/5f456938-00a9-4ecc-af48-5cfa1471533d)


**Methodology** :

This study used a Deep Learning-based approach to detect FLAIR tumors from Brain MRI images. The code execution of the problem is done in Python using PyTorch, a Python library that is used for training deep-learning models, in a GPU environment (Paszke, et. al., 2019). The dataset was split into train set consisting of 80% of the data, and validation and test set consisting of 10% each.

**Proposed Model:** The U-Net-architecture-based segmentation model is known to be the best-performing Fully Convolutional Network (FCN) model for segmentation of biomedical images (Ronneberger, et. al., 2015). The model was imported and loaded from a [GitHub repository of mateuszbuda](https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py) with torch.hub. The model consists of a symmetric U-shape architecture (Figure 3) made up of an encoder that extracts features from the input image, and a decoder that uses the extracted features to generate a segmentation mask. This architecture contains 4 stacks of batch-normalization, ReLU activations, and max pooling, combined with convolutions for encoding, and deconvolutions for decoding.

![image](https://github.com/sanyuktaadap/lgg-segmentation/assets/126644146/8bb6797d-a1ff-4c7b-8d20-3a575be96dac)


**Preprocessing** : The images and their masks were first converted to Image Tensors with the image having 3 channels (RGB), and the mask with one binary channel. Both were then resized to 256x256 pixels. 3 different augmentation techniques were used: Random Rotation, Random Horizontal Flip, and Random Vertical Flip. All augmentations were applied on both, the image as well as its corresponding mask. Then, the images were converted to PyTorch tensors of data type float, and the corresponding masks to data type integer and normalized all the values between 0 and 1. The final images after the preprocessing steps were used as inputs to the model along with their corresponding segmentation masks.

**Training phase:** After preprocessing steps, the images were now compatible to feed the model. The model was trained from scratch without pretrained weights. Multiple experiments were performed with different combination of hyperparameter values. For example, the batch size was initially set to 6 images, due to which the model was overfitting on the dataset. To increase the generalizability of the model, we experimented with batch sizes of 10, 12 and 16. Similarly, experiments were also performed by tweaking learning rate, lambda, and different optimizers and loss functions. We also increase the number of augmentations to make the model more robust. The final values of each parameter for the proposed model are listed in Table 1.

| Learning Rate | 1.00E-04 |
| --- | --- |
| Lambda | 1.00E-05 |
| Batch Size | 16 |
| Number of Epochs | 40 |
| Loss Function | Binary Cross Entropy |
| Optimizer | Adam |

_Table 1. Hyperparameter of the Proposed Model_

**Validation** : The model performances with different hyperparameters were validated using the validation set after each epoch of training, and their final performances were compared. We calculated the following for each mini batch: (Tiu, 2020)

1. Intersection over Union (IoU) = Area of Overlap / Area of Union
2. Dice score = 2 \* (number of common pixels) / (total number of pixels in both images)
3. Recall score = True Positives / (True Positives + False Negatives)

**Results** :

After multiple rounds of hyperparameter tuning, the results obtained are shown in Table 2. The loss across batches is as low as 99%, and the IoU score for test set is close to 78% with a dice score of 87% and a recall of 82%.

| **Dataset** | **Loss** | **IoU** | **Dice Score** | **Recall Score** |
| --- | --- | --- | --- | --- |
| Train | 0.0068 | 0.7803 | 0.8737 | 0.8647 |
| Validation | 0.0080 | 0.7402 | 0.8476 | 0.7986 |
| Test | 0.0089 | 0.7703 | 0.8680 | 0.8190 |

_Table 2. Evaluating model performances with Training and Validation sets_

The metrics graphs and logging of sample images was done using the Tensor Board (Leskovec, et. al., 2015).

![image](https://github.com/sanyuktaadap/lgg-segmentation/assets/126644146/626ff8e7-99e2-4286-8295-a0334c4444fa)


Figure 4 shows the downward trend of the loss curve. This indicates a smooth continuous decrease in loss over iterations. IoU, Dice and Recall Curves are also seen to be increasing as the training moves forward. This means that the segmentation masks being created have a good overlap with the ground truth.

In Figure 5, the first image, the box to the left, is an example where the model seems to perform well. However, in the other image, the model was not successfully able to segment the tumor.

![image](https://github.com/sanyuktaadap/lgg-segmentation/assets/126644146/9ebdeae0-c28e-4a0c-8d11-5bbfada7da54)

Figure 6 is a depiction of how the model progresses over epochs. The model's prediction after the first epoch seems to be way out of line. It looks like the model is identifying all green pixels as tumor, which is not the case. However, with more and more epochs of training, the performance keeps improving. The second image with result from the 40th epoch seems to be much better than the earlier prediction.

![image](https://github.com/sanyuktaadap/lgg-segmentation/assets/126644146/6a9ebb65-883c-4cd7-a748-fc485e5388ef)

Figure 7. shows that while testing the model on unknown data, the model seems to give a decent performance for segmenting simpler structures of the tumor, however, in cases where the tumor exists in smaller fragments, or mosaic in some fashion, the model tends to generalize and give a smoothened output. We do not want the model to mark a pixel as tumor even though it isn't.

![image](https://github.com/sanyuktaadap/lgg-segmentation/assets/126644146/02204ed4-67db-4999-9be7-06920de858cf)


**Conclusion** :

In this study, a Brain MRI image dataset was used to segment Low-Grade Glioma. The approach used in this study implements a deep learning (Convolutional Neural Network) model called the U-Net. This model was trained on a subset of the dataset from scratch. 3 different metrics were used to evaluate the performance of this model: IoU, Dice Score, and Recall. The loss significantly dropped at the end of training and became as low as 99.4%. Similar increasing trends were observed for IoU, Dice score and Recall.

With a closer look at the sample images, loss curves, and metric scores, we can conclude quite some information. Firstly, by examining the images it is clear that the model is not fully able to identify complex patterns. It is giving a more generalized output. This could mean that the model is underfitting on the dataset. To avoid this issue, there needs to be a dataset that has more of such complex forms of tumor present in it.

Another way to improve the performance is by using pre-trained weights, and then fine-tuning the hyperparameters according to the task. This would significantly reduce the time needed for training, and possibly increase the model performance.

A previous study that used a similar model for segmenting Low-Grade Glioma using a different Brain MRI dataset has achieved an IoU of 92%, whereas this study has achieved a slightly lower IoU of 87%, which is still a comparably good score.

Due to time constraints, further improvement of the model performance could not be achieved. However, the model could be made more robust by gathering more data or using the same dataset and apply more augmentations. Looking at the logged images, a slightly more complex model might be able to segment the images better as the additional layers can help capture complex features of the data.

**References** :

StatPearls. (2023). Low-Grade Gliomas. In W. Aiman, D. P. Gasalberti, & A. Rayi, StatPearls.

Boston Children's Hospital. (n.d.). Low-Grade Gliomas. [https://www.childrenshospital.org/conditions/low-grade-gliomas](https://www.childrenshospital.org/conditions/low-grade-gliomas)

Haydar, N., Alyousef, K., Alanan, U., Issa, R., Baddour, F., Al-shehabi, Z., & Al-janabi, M. H. (2022). Role of Magnetic Resonance Imaging (MRI) in grading gliomas comparable with pathology: A cross-sectional study from Syria. Annals of Medicine and Surgery (London), 82, 104679. [https://doi.org/10.1016/j.amsu.2022.104679](https://doi.org/10.1016/j.amsu.2022.104679)

St. Jude Children's Research Hospital. (n.d.). Low-Grade Glioma. [https://www.stjude.org/disease/low-grade-glioma.html](https://www.stjude.org/disease/low-grade-glioma.html)

[https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation/data?select=kaggle\_3m](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation/data?select=kaggle_3m)

[https://wiki.cancerimagingarchive.net/display/Public/TCGA-LGG](https://wiki.cancerimagingarchive.net/display/Public/TCGA-LGG)

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., … Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. In Advances in Neural Information Processing Systems 32 (pp. 8024–8035). Curran Associates, Inc.

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. Medical Image Computing and Computer-Assisted Intervention (MICCAI).

Leskovec, J., Rajaraman, A., & Ullman, J. D. (2015). Large-Scale Machine Learning. Mining of Massive Datasets, 415–458. doi:10.1017/cbo9781139924801.013

Ranjbarzadeh, R., Bagherian Kasgari, A., Jafarzadeh Ghoushchi, S., Anari, S., Naseri, M., & Bendechache, M. (2021). _Brain tumor segmentation based on deep learning and an attention mechanism using MRI multi-modalities brain images. Scientific Reports, 11(1)._ doi:10.1038/s41598-021-90428-8

Tiu, E. (2020). _Metrics to evaluate your semantic segmentation model_. Medium. https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
