import datetime
import os

import numpy as np

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from keras.optimizers import Adam
from discriminator import build_discriminator
from generator import build_generator
from preprocessor import preprocess_inbreast_for_pretraining


class PretrainGAN:

    def __init__(self):
        # Input shape
        self.img_rows = 512
        self.img_cols = 512
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64


        self.d = None
        self.g = None
        self.combined = None


    def construct(self):
        # Build the discriminators
        self.d = build_discriminator(self.img_shape, self.df)

        # Build the generators
        self.g = build_generator(self.img_shape, self.gf, self.channels)
        self.build()


    def save(self, cyclegan_folder):
        os.makedirs(cyclegan_folder, exist_ok=True)

        # Save discriminators to disk
        self.d.save(os.path.join(cyclegan_folder, 'discriminator_pretrained.h5'))

        # Save generator to disk
        self.g.save(os.path.join(cyclegan_folder, 'generator_pretrained.h5'))

    def build(self):
        optimizer = Adam(0.0002, 0.5)

        self.d.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])


        # Input images from both domains
        img = Input(shape=self.img_shape)

        img_id = self.g(img)

        self.d.trainable = False

        # Discriminators determines validity of translated images
        valid = self.d(img_id)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img],
                              outputs=[valid, img_id])

        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 10],
                              optimizer=optimizer)

    def train(self, my_dataset, epochs, batch_size=1, print_interval=100):

        my_dataset = my_dataset.batch(batch_size, drop_remainder=True)


        # Configure data loader
        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)


        for epoch in range(epochs):
            for batch_i, img in enumerate(my_dataset):
                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                img_id = self.g.predict(img)

                # Train the discriminators (original images = real / translated = Fake)
                d_loss_real = self.d.train_on_batch(img, valid)
                d_loss_fake = self.d.train_on_batch(img_id, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


                # ------------------
                #  Train Generators
                # ------------------


                g_loss = self.combined.train_on_batch([img],
                                                      [valid, img])

                elapsed_time = datetime.datetime.now() - start_time

                progress_str = f"[Epoch: {epoch}/{epochs}] [Batch: {batch_i}] [D_loss: {d_loss[0]:.5f}, acc: {100 * d_loss[1]:.5f}] " \
                                   f"[adv: {g_loss[1]:.5f}], [identity: {np.mean(g_loss[2]):.5f}]," \
                                   f"time: {elapsed_time}"

                # Plot the progress
                if batch_i % print_interval == 0:
                    print(progress_str)


            # Comment this in if you want to save checkpoints:
            self.save(os.path.join('..','models','GAN','ep_' + str(epoch)))





if __name__ == '__main__':
    dataset = preprocess_inbreast_for_pretraining('trainval')
    gan = PretrainGAN()
    gan.construct()

    gan.train(my_dataset=dataset, epochs=1000, batch_size=16, print_interval=10)
    #gan.save(os.path.join('..', 'models', 'GAN'))

