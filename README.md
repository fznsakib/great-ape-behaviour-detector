# great-ape-behaviour-detector

**Thesis Title**: Great Ape Behaviour Recognition using Deep Learning

Whilst species detection and individual identification of great apes have been attempted with success, research in automated behaviour recognition remains a human-centric task. In response, this project successfully introduces a novel deep learning behaviour recognition model specifically for great apes, detecting a wide-ranging set of core behaviours. Among others, they include acts of sitting, walking, climbing and interactions with the camera. As per the domain-specific requirements of performing this task in the wild, the model is capable of simultaneously evaluating multiple great apes, each dynamically displaying behaviours, for a given video. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/15062683/84395341-e7a66380-abf5-11ea-9076-635f12fb64b9.gif" width=400>
  <img src="https://user-images.githubusercontent.com/15062683/84399718-15d97280-abf9-11ea-8db2-0488c21fe4c8.gif" width=400>
  <br>
  <img src="https://user-images.githubusercontent.com/15062683/84400225-b2037980-abf9-11ea-8898-1154aa7c8026.gif" width=400>
  <img src="https://user-images.githubusercontent.com/15062683/84400855-7a490180-abfa-11ea-972b-6f060a080c01.gif" width=400>
</p>

After evaluating 100 test videos in a 4-fold cross validation procedure, the results are as follows:

<p align="center">
  <img src="https://user-images.githubusercontent.com/15062683/84390371-621fb500-abef-11ea-89d4-26092c305dc5.png" width=300>
</p>

The model was trained and evaluated on a dataset consisting of 500 videos, amounting to 180,000 frames. All computation was performed on an NVIDIA P100 GPU with a requirement of 128GB RAM.

## Architecture

The model utilises a deep two-stream architecture inspired by the work of [Simonyan and Zisserman](https://papers.nips.cc/paper/5353-two-stream-convolutional-networks-for-action-recognition-in-videos.pdf). Each stream incorporates a ResNet-18 CNN. The *spatial* stream effectively captures properties based on the appearance of behaviours. Meanwhile, the *temporal* stream captures the motion dependencies. The spatial stream takes in a sequence of RGB images as input, while the temporal stream takes in a sequence of optical flow images. Both streams are fused to get a comprehensive view of the visual features extracted from the displayed behaviour.

After an extreme level of class imbalance was identified within the dataset to be a performance bottleneck, the model underwent a substantial transformation. The introduction of balanced sampling and the [focal loss](https://arxiv.org/pdf/1708.02002.pdf) directly addressed the dominance of the majority classes. Stateful memory retaining LSTMs also proved to be vital, leveraging the inherent sequential temporality of video data.

The techniques above, along with an extensive set of experiments and ablation studies, crucially facilitated the discovery of a locally optimised final model for this task, vastly suppressing the class imbalance issue. The final architecture of the model can be seen below:

<p align="center">
  <img src="https://user-images.githubusercontent.com/15062683/84392031-b5930280-abf1-11ea-804f-c8f04c50fac8.jpg" width=600>
</p>

## Behaviour Classes

A total of 9 commonly exhibited behaviours among gorillas and chimpanzees are attempted to be classified by the model:

<p align="center">
  <img src="https://user-images.githubusercontent.com/15062683/84390905-1faaa800-abf0-11ea-851a-70c4727932f9.jpg" width=600>
</p>

## Paper & Execution

The final thesis paper documenting all work and experiments related to this repository can be found [here](dissertation.pdf). 

Appendix A of the thesis describes the **dataset**. Appendix B outlines in detail the **execution instructions** for the model. [Contact me](mailto:fznsakib@gmail.com) if further help is needed.
