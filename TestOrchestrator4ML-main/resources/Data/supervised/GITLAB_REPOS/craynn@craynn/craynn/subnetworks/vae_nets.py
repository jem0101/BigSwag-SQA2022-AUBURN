from ..layers.noise_ops import gaussian_rv
from .meta import achain

__all__ = [
  'vae'
]

def vae(encoder, decoder, latent_op=gaussian_rv(name='latent')):
  def subnet(*incoming):
    code_params = achain(encoder)(*incoming)

    if len(code_params) != 2:
      raise Exception('VAE encoder must produce 2 outputs: mean and std of posterior code distribution.')

    mean, std = code_params
    code = latent_op(mean, std)
    reconstructed, = achain(decoder)(code)
    return reconstructed, mean, std

  return subnet