import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 n_input=1,
                 stride=16,
                 n_channel=12,
                 act_fn: object = nn.GELU):
        """
        Args:
            - latent_dim : Dimensionality of latent representation z
            - n_input : Number of channels of the audio (1) to reconstruct
            - stride : Stride of convolutional layers
            - n_channel : Channels of convolutional layers in encoder
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(168, latent_dim),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride, padding=1),
            act_fn(),
            nn.Conv1d(n_channel, n_channel, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3, stride=int(stride / 4), padding=1)
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, x.shape[1] * x.shape[2])
        x = self.linear(x)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 n_input=1,
                 stride=16,
                 n_channel=12,
                 act_fn: object = nn.GELU):
        """
        Inputs:
            - latent_dim : Dimensionality of latent representation z
            - n_input : Number of channels of the audio (1) to reconstruct
            - stride : Stride of convolutional layers
            - n_channel : Channels of convolutional layers in decoder
            - act_fn : Activation function used throughout the decoder network
        """
        self.n_channel = n_channel
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 168),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose1d(2 * n_channel, n_channel, kernel_size=3, output_padding=22, stride=int(stride / 4)),
            act_fn(),
            nn.Conv1d(n_channel, n_channel, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose1d(n_channel, n_input, kernel_size=80, stride=stride, output_padding=120)
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 24, 7)
        x = self.net(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 n_input=1,
                 n_output=35,
                 stride=16,
                 n_channel=12,
                 encoder_class: object = Encoder,
                 decoder_class: object = Decoder,
                 ):
        """
        Inputs:
            - latent_dim : Dimensionality of latent representation z
            - n_input : Number of channels of the audio (1) to reconstruct
            - stride : Stride of convolutional layers
            - n_channel : Channels of convolutional layers in decoder
            - encoder_class : Encoder class to be used
            - decoder_class : Decoder class to be used
        """
        super().__init__()
        # Creating encoder and decoder
        self.encoder = encoder_class(latent_dim, n_input, stride, n_channel)
        self.decoder = decoder_class(latent_dim, n_input, stride, n_channel)

    def forward(self, x):
        """
        The forward function takes in a noisy audio and returns the reconstructed audio
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

