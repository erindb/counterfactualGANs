# Counterfactuals over GANs

## Overview

GANs take independent noise and create movel samples from a distribution that looks like the real distribution they're trained on.

What if we sample small perturbations to a given random noise state, and then condition on that distribution?
We think we might get counterfactuals.

Concretely, if we have a way of determining for each image how likely it is to be female and how likely it is to have a mustache (e.g. a classifier for each label), we can first conditionally sample a femal face from the prior, and *then* counterfactually imagine what she would look like if she had a mustache.

We implement the counterfactual sampling procedure from [An improved probabilistic account of counterfactual reasoning](https://philpapers.org/rec/LUCAIP) on the noise vector input to the GAN to create the "actual" woman's face to create the counterfactual distribution over her "counterfactual" faces with a mustache.

## Queries

* Given a woman, what if she had a moustache?
* Given a man, what if he were wearing lipstick?
* Given a woman, what if she were a man?
* Given a young man, what if he were bald?
* Given a person who's not wearing glasses, what if they were wearing glasses?
* Given a person who's not smiling, what if they were smiling?

## What if this doesn't work?

If this doesn't work, it means that the GAN doesn't know the causal structure.
Maybe we can tell it things, like "If this young man were old, he might be bald," *while* it's training.
Would that help it learn better causal structures?

## Dataset

Download [Celeb A](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) datasets.

CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset with more than 200K celebrity images, each with 40 attribute annotations. The images in this dataset cover large pose variations and background clutter. CelebA has large diversities, large quantities, and rich annotations, including

* 10,177 number of identities,
* 202,599 number of face images, and
* 5 landmark locations, 40 binary attributes annotations per image.

Attributes:

* `5_o_Clock_Shadow`
* `Arched_Eyebrows`
* `Attractive`
* `Bags_Under_Eyes`
* `Bald`
* `Bangs`
* `Big_Lips`
* `Big_Nose`
* `Black_Hair`
* `Blond_Hair`
* `Blurry`
* `Brown_Hair`
* `Bushy_Eyebrows`
* `Chubby`
* `Double_Chin`
* `Eyeglasses`
* `Goatee`
* `Gray_Hair`
* `Heavy_Makeup`
* `High_Cheekbones`
* `Male`
* `Mouth_Slightly_Open`
* `Mustache`
* `Narrow_Eyes`
* `No_Beard`
* `Oval_Face`
* `Pale_Skin`
* `Pointy_Nose`
* `Receding_Hairline`
* `Rosy_Cheeks`
* `Sideburns`
* `Smiling`
* `Straight_Hair`
* `Wavy_Hair`
* `Wearing_Earrings`
* `Wearing_Hat`
* `Wearing_Lipstick`
* `Wearing_Necklace`
* `Wearing_Necktie`
* `Young`

## Causal structure

[CausalGAN: Learning Causal Implicit Generative Models
with Adversarial Training](https://arxiv.org/pdf/1709.02023.pdf) gives a nice, intuitive causal structure for the CelebA dataset.

![](img/causal_structure.png)

They train their model to be able to sample novel images while conditioning on a given set of discrete labels (maintaining causal relationships), or while changing the labels by intervention (creating implausible but imaginable samples).

## Pre-trained GAN

https://github.com/carpedm20/DCGAN-tensorflow

```
python download.py celebA
python main.py --dataset celebA --input_height=108 --crop
```

## Possible timeline

* Week 1 ~ set up environment
	- Alex: 
		* [x] python 3
		* [x] tensorflow 0.12.1
		* [x] download and train [GAN](https://github.com/carpedm20/DCGAN-tensorflow)
		* [x] we *might* need access to `cocoserv2`
	- Erin
		* [x] make repo & integrate with slack
* Week 2 design forward model given noise
	* [x] explore/understand trained model
	* [x] find where the noise vector is passed in and be able to modify it
	* [x] make a function in python that
		- takes in a noise vector and
		- outputs the generated image(s) from that noise vector
		- add/call this function in `main.py`
	* [x] set up X11 forwarding
		- install Xquartz
		- `ssh -Y cocoserv2`
		- to open an image `eog [IMG].png`
	* [x] read up on DCGANs: how many images per noise vector?
		- Ans: one, given that the weights and biases are fixed. it is a deterministic function at that point. duh.)
* Week 3 & 4 debugging and exploring forward model
	* [x] debug and run forward model (`main.py`)
		- image drawing tool that we took from the DCGANs repo *requires* 64 images at a time (or whatever the batch size was in training)
	* [x] record how long this takes to run (loading model / sampling)
		- it's pretty fast
	* [x] explore sampled images for different noise vectors (as needed, move images over `scp cocoserv2:/home/alex/samples/test_arange_42.png .` or `rsync cocoserv2:/home/alex/samples/ .` for the whole directory)
		- get a sense of the full distribution over images
			- between 0 and 1 we definitely get some plausible images. and even up to about 5, they're pretty good for some dimensions at least. outside of that, things get crazy.
			- this makes sense, cause it was trained with uniform(-1, 1)
		- how much do we need to change a noise vector to get noticable changes in the images?
		- are there some dimensions that cause bigger changes than others?
	* [x] short presentation of the distribution over faces that the trained model generates
		- yayyy
* Week 5 ~ counterfactual sampling
	* [x] pick 3 base noise vectors (including the origin) that generate good images and have different properties (like glasses or smiling)
		* origin, not smiling girl, guy, orange guy with glasses
	* [x] make a `gaussian_cf_sampler` function that takes in a base image vector and number of samples (e.g. 64) and outputs that number of "counterfactual samples" by taking a `np.random.randn` vector and scaling it to center around that base image.
	* [x] make a [`esm_cf_sampler`](https://philpapers.org/rec/LUCAIP) function that for each dimension:
		- with probability `stickiness` (start with 0.5) keeps the same number as the base image has for that dimension
		- otherwise samples from `np.random.uniform(-1, 1)`
		- try out different values for `stickiness` and different ranges for `np.random.uniform()`
	* [x] start looking into [`Pyro`](http://pyro.ai/) tutorials
		- make note of anything that's unclear (and/or submit a pull request)
* Week 6 ~ find an image classifier
	* [x] look for pre-existing classifiers for "smiling" on CelebA, ideally in PyTorch
		- http://pytorch.org/docs/master/torchvision/models.html
	* [x] learn about convolutional NNs
		- http://cs231n.github.io/convolutional-networks/
	* [x] start to adapt AlexNet to our 64x64 images
* Week 7 ~ smiling classifier
	* [ ] build smiling classifier
		- figure out how to load CelebA data (e.g. copy CDGAN code for this)
		- 1-class: smiling or not
		- 5 features
		- set up to use CUDA
		- try AlexNet, but if it's too slow, simplify it or grab another model
	* [ ] run classifier on cocoserv2
	* [ ] report performance of the classifier
		- accuracy
		- pull out a few misclassified images
		- pull out a few correctly classified images
	* [ ] Erin: sudo access
* Week 8 ~ conditionally sample from GAN
	* [ ] make a function that samples an image, runs the classifier, and returns only images with classification "smiling"
	* [ ] keep looking into [`Pyro`](http://pyro.ai/) tutorials
		- make note of anything that's unclear (and/or submit a pull request)
* Week 9 ~ extend to more labels
	* [ ] the ones in that picture of the bayes net
* Week 10 ~ evaluate specific causal links
	* [ ] e.g. `old --> bald <-- male`

## Some references/papers

* read about probabilistic models, e.g. http://probmods.org/
* read about probabilistic programming languages, e.g. http://dippl.org/
* read about DCGANs, e.g. https://arxiv.org/abs/1511.06434
* keeping reading through counterfactual model: https://philpapers.org/rec/LUCAIP
* read about causal structure in GANs, e.g. https://arxiv.org/pdf/1709.02023.pdf

## Classifiers

???

## Extras

[Neural Face](http://carpedm20.github.io/faces/) uses a vector `z` that consists of 100 real numbers ranging from 0 to 1.
They visualize what happens when each element in the noise vector is changed.

* smiling people have wrinkles around their mouths. that's causal (muscles and stuff). and it did learn it. so that's cool.
* can we do negative numbers?
* pytorch dcgan: https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/deep_convolutional_gan


