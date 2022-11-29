
## Object recognition and computer vision 2022/2023
 This is my take on the assignement 3 given in the MVA Course on Object Recognition and Computer Vision.
  

### Assignment 3: Image classification

  

#### Requirements

1. Install PyTorch from http://pytorch.org

  

2. Run the following command to install additional dependencies

  

```bash

pip install -r requirements.txt

```

  

#### Dataset

We will be using a dataset containing 200 different classes of birds adapted from the [CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).

Download the training/validation/test images from [here](https://www.di.ens.fr/willow/teaching/recvis18orig/assignment3/bird_dataset.zip). The test image labels are not provided.

  

#### Training and validating your model

Run the script `main.py` to train the model from scratch.

  

```

python main.py [--data D] [--batch-size B] [--epochs N] [--lr LR] [--seed S] [--log-interval N] [--experiment E] [--save-interval SI]

```

  

- By default the images are loaded then cropped using the bird prediction from a RCNN [1]. Then an augmentation pipeline based on AugMix [2] is used before normalizing the image using the ImageNet statistics.

See data.py for the `data_transforms`.

  

- The classification is then done using a ResNet [3] model (*resnet50*) pretrained on ImageNet.

  

#### Evaluating your model on the test set

  

As the model trains, model checkpoints are saved to files such as `model_x.pth` to the current working directory.

You can take one of the checkpoints and run:

  

```

python evaluate.py --data [data_dir] --model [model_file]

```

  

That generates a file `kaggle.csv` that you can upload to the private kaggle competition website.

  

#### Acknowledgments

Adapted from Rob Fergus and Soumith Chintala https://github.com/soumith/traffic-sign-detection-homework.<br/>

Adaptation done by Gul Varol: https://github.com/gulvarol

  
  

#### References

- [[1]](https://arxiv.org/abs/1506.01497) Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks ; Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun

- [[2]](https://arxiv.org/abs/1912.02781) AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty ; Dan Hendrycks, Norman Mu, Ekin D. Cubuk, Barret Zoph, Justin Gilmer, Balaji Lakshminarayanan

- [[3]](https://arxiv.org/abs/1512.03385) Deep Residual Learning for Image Recognition ; Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun