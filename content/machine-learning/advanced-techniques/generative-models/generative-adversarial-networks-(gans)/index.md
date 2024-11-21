---
linkTitle: "Generative Adversarial Networks (GANs)"
title: "Generative Adversarial Networks (GANs): Training a Generator and Discriminator in Tandem"
description: "Detailed description of GANs where a generator and discriminator network are trained simultaneously, their architecture, training process, challenges, code examples, and related design patterns."
categories:
- Advanced Techniques
- Generative Models
tags:
- GANs
- Deep Learning
- Generative Models
- Neural Networks
- Unsupervised Learning
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/advanced-techniques/generative-models/generative-adversarial-networks-(gans)"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Generative Adversarial Networks (GANs): Training a Generator and Discriminator in Tandem

Generative Adversarial Networks (GANs) represent a breakthrough concept in the domain of generative models within machine learning. Invented by Ian Goodfellow and his colleagues in 2014, GANs consist of two neural networks—the generator and the discriminator—that are trained simultaneously in a competitive setting. The generator creates data that mimics the training data, while the discriminator evaluates their authenticity. The objective is to improve the generator's ability to create data indistinguishable from real data over time.

### What Are GANs?

GANs are essentially a game between two players: the generator \\(G\\) and the discriminator \\(D\\). The generator's job is to generate fake data, while the discriminator's job is to distinguish between real and fake data. Formally:

- The generator \\(\mathbf{G}(z; \theta_g)\\) samples from a latent space (random noise \\(z\\)) and generates fake data \\(\tilde{x}\\).
- The discriminator \\(\mathbf{D}(x; \theta_d)\\) is a binary classifier that distinguishes between real data \\(x\\) and fake data \\(\tilde{x}\\).

The GAN framework can be formulated as a min-max game:
{{< katex >}} \min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log (1 - D(G(z)))] {{< /katex >}}

### Training Process

1. **Initialize** the parameters of both networks.
2. **Repeat** the following steps until convergence:
    - **Discriminator update**: Train the discriminator \\(D\\) on a batch consisting of real data and fake data produced by the generator. The goal is to maximize the probability of correctly classifying real and fake data.
    - **Generator update**: Train the generator \\(G\\) to produce data which maximizes the probability of the discriminator classifying it as real. This is equivalent to minimizing \\(\log (1 - D(G(z)))\\) which often translates into maximizing \\(\log D(G(z))\\), providing better gradients early in training.


### Code Example

#### Example Using TensorFlow and Keras

```python
import tensorflow as tf
from tensorflow.keras import layers

latent_dim = 100
batch_size = 64
epochs = 10000

def build_generator(latent_dim):
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=latent_dim),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(28 * 28 * 1, activation='sigmoid'),
        layers.Reshape((28, 28, 1)),
    ])
    return model

def build_discriminator(img_shape):
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=img_shape),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ])
    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = layers.Input(shape=(latent_dim,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = tf.keras.Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

generator = build_generator(latent_dim)
discriminator = build_discriminator((28, 28, 1))
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
gan = build_gan(generator, discriminator)

def train_gan(generator, discriminator, gan, epochs, batch_size, latent_dim):
    (X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    X_train = (X_train / 127.5) - 1.0
    X_train = np.expand_dims(X_train, axis=-1)

    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # Select a random batch of images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]

        # Generate fake images
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = generator.predict(noise)

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(real_imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, real)

        # Print progress
        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}%] [G loss: {g_loss}]")

train_gan(generator, discriminator, gan, epochs, batch_size, latent_dim)
```

### Examples in Different Frameworks

#### PyTorch Example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

latent_dim = 100

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.main(x)

generator = Generator()
discriminator = Discriminator()

lr = 0.0002
optim_gen = optim.Adam(generator.parameters(), lr=lr)
optim_disc = optim.Adam(discriminator.parameters(), lr=lr)

adversarial_loss = torch.nn.BCELoss()

for epoch in range(epochs):
    for _ in range(batch_size):
        real = torch.ones(batch_size, 1)
        fake = torch.zeros(batch_size, 1)

        # Train Discriminator on real and fake images
        optimizer_disc.zero_grad()
        real_imgs = images_from_data_loader()
        output_real = discriminator(real_imgs)
        loss_real = adversarial_loss(output_real, real)

        z = torch.randn(batch_size, latent_dim)
        fake_imgs = generator(z)
        output_fake = discriminator(fake_imgs)
        loss_fake = adversarial_loss(output_fake, fake)

        loss_disc = 0.5 * (loss_real + loss_fake)
        loss_disc.backward()
        optimizer_disc.step()

        # Train Generator
        optimizer_gen.zero_grad()
        z = torch.randn(batch_size, latent_dim)
        fake_imgs = generator(z)
        output_fake = discriminator(fake_imgs)
        loss_gen = adversarial_loss(output_fake, real)
        loss_gen.backward()
        optimizer_gen.step()
    
    print(f"Epoch [{epoch}/{epochs}] : D_loss: {loss_disc.item()}, G_loss: {loss_gen.item()}")

```

### Challenges

1. **Mode Collapse**: The generator may produce limited diversity, focusing on specific samples.
2. **Training Instability**: The simultaneous training of both networks can lead to instability.
3. **Hyperparameter Tuning**: GANs are highly sensitive to hyperparameter choices.
4. **Evaluation Metrics**: Evaluating the quality of generated samples lacks standardized quantitative metrics; thus subjective or application-specific methods are often used.

### Related Design Patterns

1. **Variational Autoencoder (VAE)**: Another popular generative model that tries to model the data distribution and uses a distinct probabilistic approach.
2. **CycleGAN**: An extension of GANs for unpaired image-to-image translation, also uses two GANs to map source to target and vice versa ensuring cycle-consistency.
3. **Conditional GANs**: Introducing conditional information (such as class labels) to GANs to generate class-specific data.

### Additional Resources

1. **Paper**: ([Generative Adversarial Nets](https://arxiv.org/abs/1406.2661)) by Ian Goodfellow et al.
2. **Book**: "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville, which includes a comprehensive section on GANs.
3. **Tutorials**: TensorFlow and PyTorch official documentation and tutorials about GANs.

### Summary

Generative Adversarial Networks (GANs) are a powerful and influential class of generative models. Through an adversarial process, two networks (generator and discriminator) are trained in tandem to improve their respective abilities: the generator to produce realistic fake data and the discriminator to distinguish between real and fake. Despite challenges such as instability and mode collapse, GANs have been successfully applied to a range of tasks from image generation to style transfer, pushing forward the boundaries of what artificial intelligence can achieve in generation tasks. With continual research and development, GANs continue to be at the forefront of advancements in generative models.
