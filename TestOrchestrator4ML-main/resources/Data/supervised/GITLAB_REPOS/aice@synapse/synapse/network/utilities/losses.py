"""
Copyright (C) 2020  Syed Hasibur Rahman

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
# region : Imports
from tensorflow.keras import losses as keras_losses
# endregion : Imports


class Losses:
    BinaryCrossEntropy = keras_losses.BinaryCrossentropy
    CategoricalCrossEntropy = keras_losses.CategoricalCrossentropy
    CategoricalHinge = keras_losses.CategoricalHinge
    CosineSimilarity = keras_losses.CosineSimilarity
    Hinge = keras_losses.Hinge
    Huber = keras_losses.Huber
    KLDivergence = keras_losses.KLDivergence
    LogCosh = keras_losses.LogCosh
    MeanAbsoluteError = keras_losses.MeanAbsoluteError
    MeanAbsolutePercentageError = keras_losses.MeanAbsolutePercentageError
    MeanSquareError = keras_losses.MeanSquaredError
    MeanSquaredLogarithmicError = keras_losses.MeanSquaredLogarithmicError
    Poisson = keras_losses.Poisson
    SparseCategoricalCrossEntropy = keras_losses.SparseCategoricalCrossentropy
    SquaredHinge = keras_losses.SquaredHinge
    KLD = keras_losses.KLD
    MAE = keras_losses.MAE
    MAPE = keras_losses.MAPE
    MSE = keras_losses.MSE
    MSLE = keras_losses.MSLE
    binary_crossentropy = keras_losses.binary_crossentropy
    categorical_crosentropy = keras_losses.categorical_crossentropy
    categorical_hinge = keras_losses.categorical_hinge
    cosine_similarity = keras_losses.cosine_similarity
    hinge = keras_losses.hinge
    kullback_leibler_divergence = keras_losses.kullback_leibler_divergence
    logcosh = keras_losses.logcosh
    poisson = keras_losses.poisson
    sparse_categorical_cross_entropy = keras_losses.sparse_categorical_crossentropy

