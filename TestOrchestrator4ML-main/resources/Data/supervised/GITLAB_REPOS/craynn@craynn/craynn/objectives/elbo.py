import tensorflow as tf

from .meta import Objective

__all__ = [
  'elbo_norm'
]

class elbo_norm(Objective):
  def __init__(self, sigma_reconstructed=1.0, beta=None, exact=False):
    """
     Returns Evidence Lower Bound for normally distributed (z | X), (X | z) and z:
       P(z | X) = N(`code_mean`, `code_std`);
       P(X | z) = N(`X_reconstructed`, `sigma_reconstructed`);
       P(z) = N(0, 1).
     :param sigma_reconstructed: variance for reconstructed sample, i.e. X | z ~ N(X_original, sigma_reconstructed)
       If a scalar, `Var(X | z) = sigma_reconstructed * I`, if tensor then `Var(X | z) = diag(sigma_reconstructed)`
     :param beta: coefficient for beta-VAE
     :param exact: if true returns exact value of ELBO, otherwise returns rearranged ELBO equal to the original
       up to a multiplicative constant, possibly increasing computational stability for low `sigma_reconstructed`.

     :return: (rearranged) ELBO objective: (X_original, X_reconstructed, code_mean, code_std) -> ELBO.
     """

    if beta is None:
      ### code_penalty below is missing 1/2 coefficient.
      self._beta = tf.constant(0.5, dtype='float32')
    else:
      self._beta = tf.constant(beta / 2, dtype='float32')

    if exact:
      self.normalization = tf.constant(0.5 / sigma_reconstructed ** 2, dtype='float32')
    else:
      self.normalization = tf.constant(2 * self._beta * sigma_reconstructed ** 2, dtype='float32')

    self.exact = exact

  def __call__(self, X_original, X_reconstructed, code_mean, code_std):
    reconstruction_loss = tf.reduce_mean((X_original - X_reconstructed) ** 2)

    code_penalty = tf.reduce_mean(
      code_std ** 2 + code_mean ** 2 - 2 * tf.math.log(code_std)
    )

    if self.exact:
      return self.normalization * reconstruction_loss + self._beta * code_penalty
    else:
      return reconstruction_loss + self.normalization * code_penalty
