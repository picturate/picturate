from picturate.config import CAttnGANConfig
from picturate.nets import CAttnGAN

config = CAttnGANConfig('bird')

gan = CAttnGAN(config, pretrained=True)

caption = "This little bird is blue with short beak and white underbelly"
filename = 'bird'
gan.generate_image(caption, filename)

