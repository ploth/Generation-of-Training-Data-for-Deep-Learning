# Generation of Training Data for Deep Learning 

This repository aims to give an overview of different techniques for the
generation of training data for deep learning procedures.

Inspired by the [3D-Machine-Learning overview by
timzhang642](https://github.com/timzhang642/3D-Machine-Learning#material_synthesis)
this list is created while working on my master thesis. 

## Requirements to Fulfill

- It is likely that the generated data can be used for training.
- Works without user assistance

### Categories

#### Method Used

- :city_sunset: CNN
- :couple: GAN

### Type

- :rainbow: Texture application
- :microscope: Detail transfer
- :camera: Image modification
- :movie_camera: Video modification

### Other

- :mans_shoe: 2D
- :high_heel: 3D

## Get Involved

We are on
[[matrix]](https://matrix.to/#/!wrLaekACqBgSgVNoZo:matrix.org?via=matrix.org) to
discuss, share knowledge and ask questions.

To contribute to the repository, you may add content through pull requests or open an issue.

## Table of Contents

- [Courses](#courses)
- [Popular Augmentation Techniques](#popular_augmentation_techniques)
- [Texture Synthesis](#texture_synthesis)
- [Style Transfer](#style_transfer)

<a name="courses" />

## Courses

TODO

<a name="popular_augmentation_techniques" />

## Popular Augmentation Techniques

- Flip
- Rotation
- Scale
- Crop
- Translation
- Noise
    - Gaussian

<a name="texture_synthesis" />

## Texture Synthesis

### Texture Synthesis Using Convolutional Neural Networks (2015) [[Paper]](https://arxiv.org/pdf/1505.07376.pdf)

<p align="center"><img width="50%" src="https://gitlab.com/ploth/generation-of-training-data-for-deep-learning/raw/master/images/Texture_Synthesis_Using_Convolutional_Neural_Networks.jpeg" /></p>

### High-Resolution Multi-Scale Neural Texture Synthesis (2017) [[Paper]](https://wxs.ca/research/multiscale-neural-synthesis/)

<p align="center"><img width="50%" src="https://gitlab.com/ploth/generation-of-training-data-for-deep-learning/raw/master/images/High_Resolution_Multi-Scale_Neural_Texture_Synthesis.jpg" /></p>

### Non-stationary Texture Synthesis by Adversarial Expansion (2018 SIGGRAPH) [[Paper]](http://vcc.szu.edu.cn/research/2018/TexSyn)

<p align="center"><img width="50%" src="https://gitlab.com/ploth/generation-of-training-data-for-deep-learning/raw/master/images/Non_Stationary_Texture_Synthesis_By_Adversarial_Expansion.jpeg" /></p>

### Synthesized Texture Quality Assessment via Multi-scale Spatial and Statistical Texture Attributes of Image and Gradient Magnitude Coefficients (2018 CVPR) [[Paper]](https://arxiv.org/pdf/1804.08020.pdf)

<p align="center"><img width="50%" src="https://gitlab.com/ploth/generation-of-training-data-for-deep-learning/raw/master/images/Synthesized_Texture_Quality_Assessment_via_Multi-scale_Spatial_and_Statistical_Texture_Attributes_of_Image_and_Gradient_Magnitude_Coefficients.png" /></p>


<!-- The focus seems to be on the alignment of textures -->
<!-- ### PhotoShape: Photorealistic Materials for Large-Scale Shape Collections (2018) [[Paper]](https://keunhong.com/publications/photoshape/) -->
<!-- <p align="center"><img width="50%" src="https://gitlab.com/ploth/generation-of-training-data-for-deep-learning/raw/master/images/PhotoShape_Photorealistic_Materials_for_Large-Scale_Shape_Collections.jpeg" /></p> -->

<a name="style_transfer" />

## Style Transfer

### :rainbow: :couple: :mans_shoe: TextureGAN: Controlling Deep Image Synthesis with Texture Patches (2018 CVPR) [[Paper]](https://arxiv.org/pdf/1706.02823.pdf)

<p align="center"><img width="50%" src="https://gitlab.com/ploth/generation-of-training-data-for-deep-learning/raw/master/images/TextureGAN_Controlling_Deep_Image_Synthesis_With_Texture_Patches.png" /></p>

### :rainbow: :high_heel: Unsupervised Texture Transfer from Images to Model Collections (2016) [[Paper]](http://ai.stanford.edu/~haosu/papers/siga16_texture_transfer_small.pdf)

<p align="center"><img width="50%" src="https://gitlab.com/ploth/generation-of-training-data-for-deep-learning/raw/master/images/Unsupervised_Texture_Transfer_from_Images_to_Model_Collections.png" /></p>

### :microscope: :high_heel: Learning Detail Transfer based on Geometric Features (2017) [[Paper]](http://surfacedetails.cs.princeton.edu/)

<p align="center"><img width="50%" src="https://gitlab.com/ploth/generation-of-training-data-for-deep-learning/raw/master/images/Learning_Detail_Transfer_based_on_Geometric_Features.png" /></p>

### :high_heel: Neural 3D Mesh Renderer (2017) [[Paper]](http://hiroharu-kato.com/projects_en/neural_renderer.html) [[Code]](https://github.com/hiroharu-kato/neural_renderer.git)

<p align="center"><img width="50%" src="https://gitlab.com/ploth/generation-of-training-data-for-deep-learning/raw/master/images/Neural_3D_Mesh_Renderer.jpeg" /></p>

### :mans_shoe: Exploring the structure of a real-time, arbitrary neural artistic stylization network (2017) [[Paper]](https://arxiv.org/pdf/1705.06830.pdf) [[Code]](https://github.com/tensorflow/magenta/tree/master/magenta/models/arbitrary_image_stylization) [[Live Demo]](https://reiinakano.github.io/arbitrary-image-stylization-tfjs/) [[Code (Live Demo)]](Arbitrary Style Transfer in the Browser) 

<p align="center"><img width="50%" src="https://gitlab.com/ploth/generation-of-training-data-for-deep-learning/raw/master/images/Exploring_the_structure_of_a_real-time_arbitrary_neural_artistic_stylization_network.png" /></p>
