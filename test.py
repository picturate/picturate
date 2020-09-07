from picturate.config import CAttnGANConfig
from picturate.nets import CAttnGAN

config = CAttnGANConfig('bird')

config.CUDA = False

gan = CAttnGAN(config, pretrained=True)
gan.generate_image("This little bird is blue with short beak and white underbelly")

