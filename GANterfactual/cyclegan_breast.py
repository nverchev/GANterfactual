import datetime
import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow
from keras.saving.save import load_model

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from keras.optimizers import Adam
from tensorflow_addons.layers import InstanceNormalization
from custom_layers import ForegroundLayerNormalization, ReflectionPadding2D

from GANterfactual.preprocessor import preprocess_inbreast_for_ganterfactual, preprocess_vindr_for_ganterfactual
from discriminator import build_discriminator
from generator import build_generator

class CycleGAN:

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

        # Loss weights
        self.lambda_cycle = 1  # Cycle-consistency loss
        self.lambda_id = 10  # Identity loss

        self.d_N = None
        self.d_P = None
        self.g_NP = None
        self.g_PN = None
        self.combined = None
        self.classifier = None

    def construct(self, classifier_path=None, classifier_weight=None):
        # Build the discriminators
        self.d_N = build_discriminator(self.img_shape, self.df)
        self.d_P = build_discriminator(self.img_shape, self.df)

        # Build the generators
        self.g_NP = build_generator(self.img_shape, self.gf, self.channels)
        self.g_PN = build_generator(self.img_shape, self.gf, self.channels)

        self.build_combined(classifier_path, classifier_weight)

    def load_pretrained(self, cyclegan_pretrained_folder, classifier_path=None, classifier_weight=None):
        custom_objects = {"InstanceNormalization": InstanceNormalization,
                          "ForegroundLayerNormalization": ForegroundLayerNormalization,
                          "ReflectionPadding2D":  ReflectionPadding2D}
        # self.d_N = build_discriminator(self.img_shape, self.df)
        # self.d_P = build_discriminator(self.img_shape, self.df)

        # Load discriminators from disk
        self.d_N = keras.models.load_model(os.path.join(cyclegan_pretrained_folder, 'discriminator_pretrained.h5'),
                                            custom_objects=custom_objects, compile=False)
        self.d_N._name = "d_N"
        self.d_P = keras.models.load_model(os.path.join(cyclegan_pretrained_folder, 'discriminator_pretrained.h5'),
                                            custom_objects=custom_objects, compile=False)
        self.d_P._name = "d_P"

        # Load generators from disk
        self.g_NP = keras.models.load_model(os.path.join(cyclegan_pretrained_folder, 'generator_pretrained.h5'),
                                            custom_objects=custom_objects, compile=False)
        self.g_NP._name = "g_NP"
        self.g_PN = keras.models.load_model(os.path.join(cyclegan_pretrained_folder, 'generator_pretrained.h5'),
                                            custom_objects=custom_objects, compile=False)
        self.g_PN._name = "g_PN"

        self.build_combined(classifier_path, classifier_weight)

    def load_existing(self, cyclegan_folder, classifier_path=None, classifier_weight=None):
        custom_objects = {"ForegroundLayerNormalization": ForegroundLayerNormalization,
                          "ReflectionPadding2D":  ReflectionPadding2D }

        # Load discriminators from disk
        self.d_N = keras.models.load_model(os.path.join(cyclegan_folder, 'discriminator_n.tf'),
                                           custom_objects=custom_objects)
        self.d_N._name = "d_N"
        self.d_P = keras.models.load_model(os.path.join(cyclegan_folder, 'discriminator_p.tf'),
                                           custom_objects=custom_objects)
        self.d_P._name = "d_P"

        # Load generators from disk
        self.g_NP = keras.models.load_model(os.path.join(cyclegan_folder, 'generator_np.tf'),
                                            custom_objects=custom_objects)
        self.g_NP._name = "g_NP"
        self.g_PN = keras.models.load_model(os.path.join(cyclegan_folder, 'generator_pn.tf'),
                                            custom_objects=custom_objects)
        self.g_PN._name = "g_PN"

        self.build_combined(classifier_path, classifier_weight)

    def save(self, cyclegan_folder):
        os.makedirs(cyclegan_folder, exist_ok=True)

        # Save discriminators to disk
        self.d_N.save(os.path.join(cyclegan_folder, 'discriminator_n.tf'))
        self.d_P.save(os.path.join(cyclegan_folder, 'discriminator_p.tf'))

        # Save generators to disk
        self.g_NP.save(os.path.join(cyclegan_folder, 'generator_np.tf'))
        self.g_PN.save(os.path.join(cyclegan_folder, 'generator_pn.tf'))

    def build_combined(self, classifier_path=None, classifier_weight=None):
        optimizer_d = Adam(0.0002, 0.5)
        optimizer_g = Adam(0.0002, 0.5)
        self.d_N.compile(loss='mse',
                         optimizer=optimizer_d,
                         metrics=['accuracy'])
        self.d_P.compile(loss='mse',
                         optimizer=optimizer_d,
                         metrics=['accuracy'])

        # Input images from both domains
        img_N = Input(shape=self.img_shape)
        img_P = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_P = self.g_NP(img_N)
        fake_N = self.g_PN(img_P)
        # Translate images back to original domain
        reconstr_N = self.g_PN(fake_P)
        reconstr_P = self.g_NP(fake_N)
        # Identity mapping of images
        img_N_id = self.g_PN(img_N)
        img_P_id = self.g_NP(img_P)

        # For the combined model we will only train the generators
        self.d_N.trainable = False
        self.d_P.trainable = False

        # Discriminators determines validity of translated images
        valid_N = self.d_N(fake_N)
        valid_P = self.d_P(fake_P)

        if classifier_path is not None and os.path.isfile(classifier_path):
            self.classifier = load_model(classifier_path)
            self.classifier._name = "classifier"
            self.classifier.trainable = False

            class_N_loss = self.classifier(fake_N)
            class_P_loss = self.classifier(fake_P)

            # Combined model trains generators to fool discriminators
            self.combined = Model(inputs=[img_N, img_P],
                                  outputs=[valid_N, valid_P,
                                           class_N_loss, class_P_loss,
                                           reconstr_N, reconstr_P,
                                           img_N_id, img_P_id])

            self.combined.compile(loss=['mse', 'mse',
                                        'binary_crossentropy', 'binary_crossentropy',
                                        'mae', 'mae',
                                        'mae', 'mae'],
                                  loss_weights=[1, 1,
                                                classifier_weight, classifier_weight,
                                                self.lambda_cycle, self.lambda_cycle,
                                                self.lambda_id, self.lambda_id],
                                  optimizer=optimizer_g)

        else:
            # Combined model trains generators to fool discriminators
            self.combined = Model(inputs=[img_N, img_P],
                                  outputs=[valid_N, valid_P,
                                           reconstr_N, reconstr_P,
                                           img_N_id, img_P_id])

            self.combined.compile(loss=['mse', 'mse',
                                        'mae', 'mae',
                                        'mae', 'mae'],
                                  loss_weights=[1, 1,
                                                self.lambda_cycle, self.lambda_cycle,
                                                self.lambda_id, self.lambda_id],
                                  optimizer=optimizer_g)

    def train(self, dataset, epochs, batch_size=1, print_interval=100,
              sample_interval=1000):

        dataset = dataset.batch(batch_size)
        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        class_N = np.stack([np.zeros(batch_size)])
        class_P = np.stack([np.ones(batch_size)])

        for epoch in range(epochs):
            for batch_i, (imgs_N, imgs_P) in enumerate(dataset):
                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                fake_P = self.g_NP.predict(imgs_N)
                fake_N = self.g_PN.predict(imgs_P)

                # Train the discriminators (original images = real / translated = Fake)
                dN_loss_real = self.d_N.train_on_batch(imgs_N, valid)
                dN_loss_fake = self.d_N.train_on_batch(fake_N, fake)
                dN_loss = 0.5 * np.add(dN_loss_real, dN_loss_fake)

                dP_loss_real = self.d_P.train_on_batch(imgs_P, valid)
                dP_loss_fake = self.d_P.train_on_batch(fake_P, fake)
                dP_loss = 0.5 * np.add(dP_loss_real, dP_loss_fake)

                # Total disciminator loss
                d_loss = 0.5 * np.add(dN_loss, dP_loss)

                # ------------------
                #  Train Generators
                # ------------------

                if self.classifier is not None:
                    # Train the generators
                    g_loss = self.combined.train_on_batch([imgs_N, imgs_P],
                                                          [valid, valid,
                                                           class_N, class_P,
                                                           imgs_N, imgs_P,
                                                           imgs_N, imgs_P])
                else:
                    g_loss = self.combined.train_on_batch([imgs_N, imgs_P],
                                                          [valid, valid,
                                                           imgs_N, imgs_P,
                                                           imgs_N, imgs_P])

                elapsed_time = datetime.datetime.now() - start_time

                if self.classifier is not None:
                    progress_str = f"[Epoch: {epoch}/{epochs}] [Batch: {batch_i}] [D_loss: {d_loss[0]:.5f}, acc: {100 * d_loss[1]:.5f}] " \
                                   f"[G_loss: {g_loss[0]:.5f}, adv: {np.mean(g_loss[1:3]):.5f}, classifier_N: {g_loss[3]:.5f}, classifier_P: {g_loss[4]:.5f}, " \
                                   f"recon: {np.mean(g_loss[5:7]):.5f}, id: {np.mean(g_loss[7:9]):.5f}] " \
                                   f"time: {elapsed_time}"
                else:
                    progress_str = f"[Epoch: {epoch}/{epochs}] [Batch: {batch_i}] [D_loss: {d_loss[0]:.5f}, acc: {100 * d_loss[1]:.5f}] " \
                                   f"[G_loss: {g_loss[0]:.5f}, adv: {np.mean(g_loss[1:3]):.5f}, recon: {np.mean(g_loss[3:5]):.5f}, id: {np.mean(g_loss[5:7]):.5f}] " \
                                   f"time: {elapsed_time}"

                # Plot the progress
                if batch_i % print_interval == 0:
                    print(progress_str)

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i, imgs_N[0], imgs_P[0])

            # Comment this in if you want to save checkpoints:
            self.save(os.path.join('..','models','GANterfactual','ep_' + str(epoch)))

    def sample_images(self, epoch, batch_i, testN, testP):
        os.makedirs('images', exist_ok=True)
        r, c = 2, 3

        img_N = testN[np.newaxis, :, :, :]
        img_P = testP[np.newaxis, :, :, :]

        # Translate images to the other domain
        fake_P = self.g_NP.predict(img_N)
        fake_N = self.g_PN.predict(img_P)
        # Translate back to original domain
        reconstr_N = self.g_PN.predict(fake_P)
        reconstr_P = self.g_NP.predict(fake_N)

        imgs = [img_N, fake_P, reconstr_N, img_P, fake_N, reconstr_P]
        classification = [['benign', 'malignant'][int(np.argmax(self.classifier.predict(x)))] for x in imgs]

        gen_imgs = np.concatenate(imgs)
        correct_classification = ['benign', 'malignant', 'benign', 'malignant', 'benign', 'malignant']


        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c, figsize=(15, 10))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt][:, :, 0], cmap='gray', vmin=0, vmax=1)
                axs[i, j].set_title(f'{titles[j]} ({correct_classification[cnt]} | {classification[cnt]})')
                axs[i, j].set_title(f'{titles[j]} ({correct_classification[cnt]})')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%d_%d.png" % (epoch, batch_i))
        plt.close()

    def predict(self, image, malignant_target=True):
        assert (self.classifier is not None)
        class_prob = self.classifier.predict(image)
        fake = self.g_NP.predict(image) if malignant_target else self.g_PN.predict(image)
        fake_class_prob = self.classifier.predict(fake)
        return class_prob, fake, fake_class_prob





if __name__ == '__main__':
    dataset = preprocess_inbreast_for_ganterfactual('trainval')
    gan = CycleGAN()
    classifier_path = os.path.join('..', 'models', 'classifier_inbreast', 'model_200.h5')
    gan.construct(classifier_path=classifier_path, classifier_weight=0.01)
    gan.train(dataset=dataset, epochs=30, batch_size=1, print_interval=10,
          sample_interval=100)
    gan.save(os.path.join('..', 'models', 'GANterfactual_inbreast'))

