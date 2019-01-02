# Generation of Training Data for Deep Learning 

This repository aims to give an overview of different techniques for the
generation of training data for deep learning procedures.

Inspired by the [3D-Machine-Learning overview by
timzhang642](https://github.com/timzhang642/3D-Machine-Learning#material_synthesis),
I started making this list while working on my master thesis.

## List Requirements

- The paper is relevant in the area of data generation.
- Papers on data generation
    - It is likely that the generated data can be used for training.
    - Works without extensive user assistance.

### Categories

### Type

- :rainbow: Texture application
- :microscope: Detail transfer
- :camera: Image modification
- :movie_camera: Video modification

#### Method Used

- :city_sunset: CNN
- :couple: GAN

### Other

- :mans_shoe: 2D
- :high_heel: 3D

## Get Involved

We are on
[[matrix]](https://matrix.to/#/!wrLaekACqBgSgVNoZo:matrix.org?via=matrix.org) to
discuss, share knowledge and ask questions.

To contribute to the repository, you may add content through pull requests or open an issue.

## Table of Contents

- [Data Augmentation](#data_augmentation)
    - [Popular Techniques](#popular_techniques)
- [Image Synthesis](#image_synthesis)
- [Domain Synthesis](#domain_synthesis)
- [Texture Synthesis](#texture_synthesis)
- [Style Transfer](#style_transfer)
- [Regularization](#regularization)
- [Training](#training)
- [Evaluation](#evaluation)
    - [Inceptiopn Score](#inception_score)

<a name="data_augmentation" />

## Data Augmentation

<a name="popular_techniques" />

### Popular Techniques

- Flip
- Rotation
- Scale
- Crop
- Translation
- Noise
    - Gaussian

### Data Augmentation using Random Image Cropping and Patching for Deep CNNs (2015) [[Paper]](https://arxiv.org/pdf/1811.09030.pdf)

<div align="center"><img width="50%" src="https://gitlab.com/ploth/generation-of-training-data-for-deep-learning/raw/master/images/Data_Augmentation_using_Random_Image_Cropping_and_Patching_for_Deep_CNNs.png" /></div>

<a name="image_synthesis" />

## Image Synthesis

### Overview

- [An Introduction to Image Synthesis with Generative Adversarial Nets (2018)](https://arxiv.org/pdf/1803.04469.pdf)

### Large Scale GAN Training for High Fidelity Natural Image Synthesis (2018) [[Paper]](https://arxiv.org/pdf/1809.11096.pdf) [[Code/Live Demo]](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/biggan_generation_with_tf_hub.ipynb#scrollTo=Cd1dhL4Ykbm7)

<div align="center"><img width="50%" src="https://gitlab.com/ploth/generation-of-training-data-for-deep-learning/raw/master/images/Large_Scale_GAN_Training_for_High_Fidelity_Natural_Image_Synthesis.png" /></div>

###  Image-to-Image Translation with Conditional Adversarial Nets (2017) [[Website]](https://phillipi.github.io/pix2pix/) [[Paper]](https://arxiv.org/pdf/1611.07004.pdf) [[Code]](https://github.com/phillipi/pix2pix)

<div align="center"><img width="50%" src="https://gitlab.com/ploth/generation-of-training-data-for-deep-learning/raw/master/images/Image-to-Image_Translation_with_Conditional_Adversarial_Nets.png" /></div>

### Semi-parametric Image Synthesis (2018) [[Paper]](https://arxiv.org/pdf/1804.10992.pdf)

<div align="center"><img width="50%" src="https://gitlab.com/ploth/generation-of-training-data-for-deep-learning/raw/master/images/Semi-parametric_Image_Synthesis.png" /></div>

### Self-Attention Generative Adversarial Networks (2018) [[Paper]](https://arxiv.org/pdf/1805.08318.pdf)

<div align="center"><img width="50%" src="https://gitlab.com/ploth/generation-of-training-data-for-deep-learning/raw/master/images/Self-Attention_Generative_Adversarial_Networks.png" /></div>

### Conditional generative adversarial networks for convolutional face generation (2015) [[Paper]](http://cs231n.stanford.edu/reports/2015/pdfs/jgauthie_final_report.pdf) [[Code]](https://github.com/hans/adversarial)

<div align="center"><img width="50%" src="https://gitlab.com/ploth/generation-of-training-data-for-deep-learning/raw/master/images/Conditional_generative_adversarial_networks_for_convolutional_face_generation.png" /></div>

<a name="domain_synthesis" />

## Domain Synthesis

### Training Deep Networks with Synthetic Data: Bridging the Reality Gap by Domain Randomization (2018) [[Paper]](https://arxiv.org/pdf/1804.06516.pdf)

<div align="center"><img width="50%" src="https://gitlab.com/ploth/generation-of-training-data-for-deep-learning/raw/master/images/Training_Deep_Networks_with_Synthetic_Data_Bridging_the_Reality_Gap_by_Domain_Randomization.png" /></div>

<a name="texture_synthesis" />

## Texture Synthesis

### Texture Synthesis Using Convolutional Neural Networks (2015) [[Paper]](https://arxiv.org/pdf/1505.07376.pdf)

<div align="center"><img width="50%" src="https://gitlab.com/ploth/generation-of-training-data-for-deep-learning/raw/master/images/Texture_Synthesis_Using_Convolutional_Neural_Networks.jpeg" /></div>

### High-Resolution Multi-Scale Neural Texture Synthesis (2017) [[Paper]](https://wxs.ca/research/multiscale-neural-synthesis/)

<div align="center"><img width="50%" src="https://gitlab.com/ploth/generation-of-training-data-for-deep-learning/raw/master/images/High_Resolution_Multi-Scale_Neural_Texture_Synthesis.jpg" /></div>

### Non-stationary Texture Synthesis by Adversarial Expansion (2018 SIGGRAPH) [[Paper]](http://vcc.szu.edu.cn/research/2018/TexSyn)

<div align="center"><img width="50%" src="https://gitlab.com/ploth/generation-of-training-data-for-deep-learning/raw/master/images/Non_Stationary_Texture_Synthesis_By_Adversarial_Expansion.jpeg" /></div>

### Synthesized Texture Quality Assessment via Multi-scale Spatial and Statistical Texture Attributes of Image and Gradient Magnitude Coefficients (2018 CVPR) [[Paper]](https://arxiv.org/pdf/1804.08020.pdf)

<div align="center"><img width="50%" src="https://gitlab.com/ploth/generation-of-training-data-for-deep-learning/raw/master/images/Synthesized_Texture_Quality_Assessment_via_Multi-scale_Spatial_and_Statistical_Texture_Attributes_of_Image_and_Gradient_Magnitude_Coefficients.png" /></div>


<!-- The focus seems to be on the alignment of textures -->
<!-- ### PhotoShape: Photorealistic Materials for Large-Scale Shape Collections (2018) [[Paper]](https://keunhong.com/publications/photoshape/) -->
<!-- <div align="center"><img width="50%" src="https://gitlab.com/ploth/generation-of-training-data-for-deep-learning/raw/master/images/PhotoShape_Photorealistic_Materials_for_Large-Scale_Shape_Collections.jpeg" /></div> -->

<a name="style_transfer" />

## Style Transfer

### :rainbow: :couple: :mans_shoe: TextureGAN: Controlling Deep Image Synthesis with Texture Patches (2018 CVPR) [[Paper]](https://arxiv.org/pdf/1706.02823.pdf)

<div align="center"><img width="50%" src="https://gitlab.com/ploth/generation-of-training-data-for-deep-learning/raw/master/images/TextureGAN_Controlling_Deep_Image_Synthesis_With_Texture_Patches.png" /></div>

### :rainbow: :high_heel: Unsupervised Texture Transfer from Images to Model Collections (2016) [[Paper]](http://ai.stanford.edu/~haosu/papers/siga16_texture_transfer_small.pdf)

<div align="center"><img width="50%" src="https://gitlab.com/ploth/generation-of-training-data-for-deep-learning/raw/master/images/Unsupervised_Texture_Transfer_from_Images_to_Model_Collections.png" /></div>

### :microscope: :high_heel: Learning Detail Transfer based on Geometric Features (2017) [[Paper]](http://surfacedetails.cs.princeton.edu/)

<div align="center"><img width="50%" src="https://gitlab.com/ploth/generation-of-training-data-for-deep-learning/raw/master/images/Learning_Detail_Transfer_based_on_Geometric_Features.png" /></div>

### :rainbow: :high_heel: Neural 3D Mesh Renderer (2017) [[Paper]](http://hiroharu-kato.com/projects_en/neural_renderer.html) [[Code]](https://github.com/hiroharu-kato/neural_renderer.git)

<div align="center"><img width="50%" src="https://gitlab.com/ploth/generation-of-training-data-for-deep-learning/raw/master/images/Neural_3D_Mesh_Renderer.jpeg" /></div>

### :camera: :mans_shoe: Exploring the structure of a real-time, arbitrary neural artistic stylization network (2017) [[Paper]](https://arxiv.org/pdf/1705.06830.pdf) [[Code]](https://github.com/tensorflow/magenta/tree/master/magenta/models/arbitrary_image_stylization) [[Live Demo]](https://reiinakano.github.io/arbitrary-image-stylization-tfjs/) [[Code (Live Demo)]](Arbitrary Style Transfer in the Browser)

<div align="center"><img width="50%" src="https://gitlab.com/ploth/generation-of-training-data-for-deep-learning/raw/master/images/Exploring_the_structure_of_a_real-time_arbitrary_neural_artistic_stylization_network.png" /></div>

### :camera: :mans_shoe: A Neural Algorithm of Artistic Style (2015) [[Paper]](https://arxiv.org/pdf/1508.06576.pdf)

<div align="center"><img width="50%" src="https://gitlab.com/ploth/generation-of-training-data-for-deep-learning/raw/master/images/A_Neural_Algorithm_of_Artistic_Style.png" /></div>

### :camera: :mans_shoe: Deep Photo Style Transfer (2017) [[Paper]](https://arxiv.org/pdf/1703.07511.pdf) [[Code]](https://github.com/luanfujun/deep-photo-styletransfer)

<div align="center"><img width="50%" src="https://gitlab.com/ploth/generation-of-training-data-for-deep-learning/raw/master/images/Deepl_Photo_Style_Transfer.png" /></div>

### :camera: :couple: :mans_shoe: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (2017) [[Paper]](https://arxiv.org/pdf/1703.10593.pdf) [[Code (PyTorch)]](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) [[Code (Torch)]](https://github.com/junyanz/CycleGAN)

<div align="center"><img width="50%" src="https://gitlab.com/ploth/generation-of-training-data-for-deep-learning/raw/master/images/Unpaired_Image-to-Image_Translation_using_Cycle-Consistent_Adversarial_Networks.jpg" /></div>

### :mans_shoe: Arbitrary Style Transfer with Style-Attentional Networks (2018) [[Paper]](https://arxiv.org/pdf/1812.02342v2.pdf)

<div align="center"><img width="50%" src="https://gitlab.com/ploth/generation-of-training-data-for-deep-learning/raw/master/images/Arbitrary_Style_Transfer_with_Style-Attentional_Networks.png" /></div>

### :microscope: :mans_shoe: Photo-realistic Facial Texture Transfer (2017) [[Paper]](https://arxiv.org/pdf/1706.04306.pdf)

<div align="center"><img width="50%" src="https://gitlab.com/ploth/generation-of-training-data-for-deep-learning/raw/master/images/Photo-realistic_Facial_Texture_Transfer.png" /></div>


<a name="regularization" />

## Regularization

### Improved Regularization of Convolutional Neural Networks with Cutout (2017) [[Paper]](https://arxiv.org/pdf/1708.04552.pdf) [[Code]](https://arxiv.org/pdf/1708.04552.pdf)

<div align="center"><img width="50%" src="https://gitlab.com/ploth/generation-of-training-data-for-deep-learning/raw/master/images/Improved_Regularization_of_Convolutional_Neural_Networks_with_Cutout.png" /></div>

<a name="training" />

## Training

### Improved Techniques for Training GANs (2016) [[Paper]](https://arxiv.org/pdf/1606.03498.pdf)

<a name="evaluation" />

## Evaluation

<a name="inception_score" />

### Inception Score

- [Inception Score — evaluating the realism of your GAN](https://sudomake.ai/inception-score-explained/)
- [What is the rationale behind "Inception Score" as a metric for quality of generative models (e.g. GANs)?](https://www.quora.com/What-is-the-rationale-behind-Inception-Score-as-a-metric-for-quality-of-generative-models-e-g-GANs)

#### A Note on the Inception Score (2018) [[Paper]](https://arxiv.org/pdf/1801.01973.pdf)
