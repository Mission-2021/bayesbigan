# Bayesian Bidrectional GAN
By Julie Jiang and Daniel Dinjian.

Final project for COMP150 Bayesian Deep Learning (Fall 2018), Tufts University.

Instructor: Mike Huges

Code is forked from [Bidrectional GAN](https://github.com/jeffdonahue/bigan), with references to [BayesGAN](https://github.com/andrewgordonwilson/bayesgan/).

You can read our [final paper](https://github.com/julie-jiang/bayesbigan/blob/master/bayesian_gan.pdf).

## Dependencies
This code has the following dependencies (version number crucial):

    python 3.6
    theano 0.9
    cuda 8.0
    cudnn 5.1
    numpy
    

## Permutation-invariant MNIST

### Setup

Download the MNIST dataset to a directory called `mnist` and set your enviroment variable to the path to the `mnist` directory:
    
    export DATADIR=/path/to/datasets/
    
This directory should contain the MNIST data files (or symlinks to them) with these names:

    t10k-images.idx3-ubyte
    t10k-labels.idx1-ubyte
    train-images.idx3-ubyte
    train-labels.idx1-ubyte

Before running the experiment, setup theano by running:
    
    source theanosetup.sh

The `train_mnist.sh` script trains a "permutation-invariant" BiGAN (by default) on the MNIST dataset.


### BiGAN
The BiGAN discriminator (or "joint discriminator") is enabled by setting a non-zero `joint_discrim_weight`, and the number of MC samples of the generator is set with `num_generator`.

    OBJECTIVE="--encode_gen_weight 1 --encode_weight 0 --discrim_weight 0 --joint_discrim_weight 1"
    ./train_mnist.sh $OBJECTIVE --num_generator 1 --exp_dir ./exp

This should produce output like:

    0) JD: 0.6932  E: 0.6932  G0: 0.6932
    NNC_e: 91.53%  NNC_e-: 96.84%  CLS_e-: 91.44%  NND_g0_/100: 13.54  NND_g0_/10: 13.48  NND_g0_: 13.44  EGg0: 3.00  EGg0_b: 3.00
    ...
    100) JD: 0.3539  E: 1.8842  G0: 1.7892
    NNC_e: 94.20%  NNC_e-: 95.41%  CLS_e-: 89.90%  NND_g0_/100: 5.40  NND_g0_/10: 4.80  NND_g0_: 4.41  EGg0: 6.39  EGg0_b: 8.11
    ...
    200) JD: 0.2076  E: 2.5187  G0: 2.7372
    NNC_e: 95.50%  NNC_e-: 96.44%  CLS_e-: 90.96%  NND_g0_/100: 5.49  NND_g0_/10: 4.79  NND_g0_: 4.29  EGg0: 5.46  EGg0_b: 9.33
    ...

The first line of each output shows the loss (objective value) of each module -- in this case the joint discriminator (`JD`), encoder (`E`), and generator (`G`).
Here the encoder and generator losses are always equal, but this is not always the case (as in the latent regressor below).

The second line contains various measures of accuracy.

 * `NND*` measures generation quality (lower is better).
 * `NNC*` and `CLS*` measure "feature" quality by either a 1-nearest-neighbor (NNC) or logistic regression (CLS) classifier (higher is better).
   * `*_e` and `*_e-` denote the feature space, with `_e` being *E(x)* itself, and `_e-` being the layer of encoder features immediately before the output. (The latter normally works better.)
 * `EG*` measures reconstruction error (lower is better).
   * `EGr` is L2 error *|| x - G(E(x)) ||*, averaged across real data samples *x ~ p(x)*
   * `EGg` is also L2 error, but averaged across generated samples *x = G(z), z ~ p(z)*: *|| G(z) - G(E(G(z))) ||*
   * The corresponding `*_b` measures are "baselines", where the reconstruction error is computed against a *random* input, i.e. *|| x' - G(E(x)) ||* where *x* and *x'* are each random samples. The ratio `EGr / EGr_b` gives a more meaningful notion of reconstruction accuracy than `EGr` alone; e.g., if `EGr ~= EGr_b` as in epoch 0 above, no meaningful reconstruction is happening.

After training, the image samples are saved in a subdirectory in the directory specified in `--exp_dir` (in this case, `./exp/bigan_mnist_*` where `*` is the time at which the directory is created).

## CIFAR

Please download the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) and use the make sure that `$DATADIR/cifar/cifar-10-batches-py` is a valid directory. When training indicate that the dataset is `cifar`. 

`train_cifar.sh` contains the script to train the cifar dataset.

## Contact
Please contact julie.jiang@tufts.edu or daniel.dinjian@tufts.edu for any questions. 
