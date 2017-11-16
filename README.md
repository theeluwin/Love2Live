# Love2Live

Zenkai No? Love Live!

Conditional Variational Autoencoder [1] based School Idol image generation, implemented in [PyTorch](http://pytorch.org/).

Used 1,466 face images of members of main school idol teams on the animation "Love Live!", extracted from the game "Love Live! School Idol Festival".

Note that this repo's implementation is also a fine implementation of the CVAE.

### Requirements

* numpy
* scipy
* pytorch

### Usage

Clone the repo and type following:

```bash
python preprocess.py
python train.py
python sample.py
```

Using a nice GPU is heavily recommended.

### Results

###### Autoencoding images in a train set

![train](https://user-images.githubusercontent.com/1303549/32910397-720817cc-cb4c-11e7-988a-4586253b9083.jpg)

###### Autoencoding images in a test set

![test](https://user-images.githubusercontent.com/1303549/32910393-6b5d6ee0-cb4c-11e7-9849-0bccdc9ec20f.jpg)

###### Generating images from a randomly sampled latent vector

![random](https://user-images.githubusercontent.com/1303549/32910657-2eb3dfb4-cb4d-11e7-8326-dc1a0e1c23d2.jpg)

###### Interpolating conditions

Hanayo to Rin

![hanayo-to-rin](https://user-images.githubusercontent.com/1303549/32910922-ebeac7b4-cb4d-11e7-818e-11e37605ff46.jpg)

Maki to Nico

![maki-to-nico](https://user-images.githubusercontent.com/1303549/32910932-f17a0d02-cb4d-11e7-8c7c-34084a05aac9.jpg)

Maki to Yoshiko

![maki-to-yoshiko](https://user-images.githubusercontent.com/1303549/32910941-f614cb22-cb4d-11e7-94e5-c91d03e57bb7.jpg)

Ruby to Dia

![ruby-to-dia](https://user-images.githubusercontent.com/1303549/32910948-fb19044e-cb4d-11e7-893e-443eb20b8ea4.jpg)

Yoshiko to Riko

![yoshiko-to-riko](https://user-images.githubusercontent.com/1303549/32910961-02bd1564-cb4e-11e7-95d0-cd4a62cc79d8.jpg)



[1] Kingma, Diederik P., et al. "Semi-supervised learning with deep generative models." *Advances in Neural Information Processing Systems*. 2014.
