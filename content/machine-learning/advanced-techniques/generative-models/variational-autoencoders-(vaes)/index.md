---
linkTitle: "Variational Autoencoders (VAEs)"
title: "Variational Autoencoders (VAEs): Generative Model that Leverages Autoencoders"
description: "An in-depth look into Variational Autoencoders (VAEs), which are a generative model type leveraging autoencoders."
categories:
- Advanced Techniques
- Generative Models
tags:
- Variational Autoencoders
- Generative Models
- Autoencoders
- Machine Learning
- Deep Learning
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/advanced-techniques/generative-models/variational-autoencoders-(vaes)"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


Variational Autoencoders (VAEs) are a powerful type of generative model that leverages the foundational architecture of autoencoders to generate new, synthetic data that resembles a given training dataset. This article explores VAEs in depth, including examples in Python with TensorFlow and PyTorch, a discussion of related design patterns, additional resources, and a summary.

## Introduction

VAEs, introduced by Kingma and Welling in 2013, are designed to address two key challenges: efficient encoding of data into a latent space and controlled sampling from this latent space to generate new data. Unlike traditional autoencoders, which compress data to a deterministic fixed-size code, VAEs generate a probabilistic latent space.

## Theoretical Foundation

### Autoencoders Recap

Autoencoders are neural networks that learn to compress input data into a lower-dimensional latent representation and then reconstruct the data from this representation. They consist of two main parts:
- The **encoder**, \\( q_\phi(z \mid x) \\): Maps input \\( x \\) to a latent variable \\( z \\).
- The **decoder**, \\( p_\theta(x \mid z) \\): Maps the latent variable \\( z \\) back to the input space.

### Variational Autoencoders

The key innovation in VAEs is that they treat the latent space probabilistically, modelling the encoder as a distribution over the latent variables. Specifically, the encoder is parameterized to produce the mean and standard deviation of a Gaussian distribution for each input. During training, a sample \\( z \\) is drawn from this Gaussian distribution, and the decoder reconstructs the input from \\( z \\).

The loss function for VAEs combines two terms:
1. **Reconstruction loss**: Measures how well the decoder reconstructs the input from the latent variable.
2. **KL divergence**: Measures how closely the learned latent variable distribution matches a prior distribution (typically a standard normal distribution).

The combined loss is expressed as:
{{< katex >}}
\mathcal{L} = \mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] - \text{KL}(q_\phi(z \mid x) \parallel p(z))
{{< /katex >}}

### Mathematical Formulation

Let's define the objective function more formally using the Evidence Lower Bound (ELBO):

{{< katex >}}
\log p(x) \geq \mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] - \text{KL}(q_\phi(z \mid x) \parallel p(z))
{{< /katex >}}

Where:
- \\( q_\phi(z \mid x) \\) is the approximate posterior distribution.
- \\( p_\theta(x \mid z) \\) is the likelihood.
- \\( p(z) \\) is the prior distribution.

## Implementation Examples

Below are examples of implementing a VAE using TensorFlow and PyTorch.

### TensorFlow Example

```python
import tensorflow as tf
from tensorflow.keras import layers

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_encoder(input_shape, latent_dim):
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(128, activation='relu')(inputs)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    model = tf.keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')
    return model

def build_decoder(latent_dim, original_dim):
    inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(128, activation='relu')(inputs)
    outputs = layers.Dense(original_dim, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs, name='decoder')
    return model

class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        self.add_loss(kl_loss)
        return reconstructed

latent_dim = 2
original_dim = 784  # Assuming input data is 28x28 flattened images
encoder = build_encoder((original_dim,), latent_dim)
decoder = build_decoder(latent_dim, original_dim)
vae = VAE(encoder, decoder)
vae.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

import numpy as np
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, [-1, original_dim]) / 255.0
x_test = np.reshape(x_test, [-1, original_dim]) / 255.0

vae.fit(x_train, x_train, epochs=10, batch_size=32, validation_data=(x_test, x_test))
```

### PyTorch Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2_mean = nn.Linear(128, latent_dim)
        self.fc2_logvar = nn.Linear(128, latent_dim)
        
    def forward(self, x):
        h1 = torch.relu(self.fc1(x))
        z_mean = self.fc2_mean(h1)
        z_logvar = self.fc2_logvar(h1)
        return z_mean, z_logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, z):
        h3 = torch.relu(self.fc3(z))
        output = torch.sigmoid(self.fc4(h3))
        return output

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

input_dim = 784  # Assuming input data is 28x28 flattened images
latent_dim = 2
vae = VAE(input_dim, latent_dim)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

def train(epoch, data_loader):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(data_loader):
        data = data.view(-1, 784)
        data = Variable(data)
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(data)
        loss = vae.loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(data_loader.dataset):.4f}')

from torchvision import datasets, transforms
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=32, shuffle=True)

for epoch in range(1, 11):
    train(epoch, train_loader)
```

## Related Design Patterns

### Generative Adversarial Networks (GANs)

GANs are another type of generative model. Unlike VAEs, GANs use a pair of networks—a generator and a discriminator—that compete against each other in a zero-sum game. The generator produces synthetic data, while the discriminator evaluates whether each piece of data is real or fake. This adversarial process allows GANs to generate highly realistic data.

### Denoising Autoencoders

Denoising autoencoders are a variant of autoencoders designed to reconstruct clean data from corrupted input. They are trained by corrupting input data and then minimizing the reconstruction error of the cleaned data. This helps in learning more robust feature representations.

### Normalizing Flows

Normalizing flows are a class of generative models that transform a simple probability distribution (e.g., Gaussian) into a complex one by applying a series of invertible functions. Like VAEs, they allow density estimation and sampling from the learned distribution but provide an exact likelihood computation.

## Additional Resources

For those looking to deepen their understanding of VAEs, the following resources are recommended:
- [Kingma and Welling (2013) - Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- [Ian Goodfellow et al. (2016) - Deep Learning Book](https://www.deeplearningbook.org/)
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/) - Lecture notes and video lectures covering various generative models.

## Summary

Variational Autoencoders (VAEs) are a significant advancement in the field of generative models, providing a probabilistic approach to data encoding and generation. By leveraging the strengths of autoencoders and the power of variational inference, VAEs facilitate the creation of intricate and high-quality generative models. From theoretical underpinnings to practical implementations in frameworks like TensorFlow and PyTorch, VAEs offer an accessible yet powerful tool for synthetic data generation.

By understanding VAEs and their related design patterns, practitioners can develop advanced machine learning systems capable of generating a diverse range of high-quality data, pushing the boundaries of what's possible in AI and machine learning.
