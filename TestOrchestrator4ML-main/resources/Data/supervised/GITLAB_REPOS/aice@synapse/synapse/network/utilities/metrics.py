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
from tensorflow.keras import metrics as keras_metrics
# endregion : Imports


class Metrics:
    AuC = keras_metrics.AUC
    Accuracy = keras_metrics.Accuracy
    BinaryAccuracy = keras_metrics.BinaryAccuracy
    BinaryCrossEntropy = keras_metrics.binary_crossentropy
    CategoricalAccuracy = keras_metrics.CategoricalAccuracy
    CategoricalCrossEntropy = keras_metrics.CategoricalCrossentropy
    CategoricalHinge = keras_metrics.CategoricalHinge
    CosineSimilarity = keras_metrics.CosineSimilarity
    FalseNegatives = keras_metrics.FalseNegatives
    FalsePositives = keras_metrics.FalsePositives
    Hinge = keras_metrics.Hinge
    KLDivergence = keras_metrics.KLDivergence
    LogCoshError = keras_metrics.LogCoshError
    Mean = keras_metrics.Mean
    MeanAbsoluteError = keras_metrics.MeanAbsoluteError
    MeanAbsolutePercentageError = keras_metrics.MeanAbsolutePercentageError
    MeanIoU = keras_metrics.MeanIoU
    MeanRelativeError = keras_metrics.MeanRelativeError
    MeanSquaredError = keras_metrics.MeanSquaredError
    MeanSquaredLogarithmicError = keras_metrics.MeanSquaredLogarithmicError
    MeanTensor = keras_metrics.MeanTensor
    Poisson = keras_metrics.Poisson
    Precision = keras_metrics.Precision
    PrecisionAtRecall = keras_metrics.PrecisionAtRecall
    Recall = keras_metrics.Recall
    RootMeanSquaredError = keras_metrics.RootMeanSquaredError
    SensitivityAtSpecificity = keras_metrics.SensitivityAtSpecificity
    SparseCategoricalAccuracy = keras_metrics.SparseCategoricalAccuracy
    SparseCategoricalCrossEntropy = keras_metrics.SparseCategoricalCrossentropy
    SparseTopKCategoricalAccuracy = keras_metrics.SparseTopKCategoricalAccuracy
    SpecificityAtSensitivity = keras_metrics.SpecificityAtSensitivity
    SquaredHinge = keras_metrics.SquaredHinge
    Sum = keras_metrics.Sum
    TopKCategoricalAccuracy = keras_metrics.TopKCategoricalAccuracy
    TrueNegatives = keras_metrics.TrueNegatives
    TruePositives = keras_metrics.TruePositives
    binary_accuracy = keras_metrics.binary_accuracy
    categorical_accuracy = keras_metrics.categorical_crossentropy
    sparse_categorical_accuracy = keras_metrics.sparse_categorical_accuracy
    sparse_top_k_categorical_accuracy = keras_metrics.sparse_top_k_categorical_accuracy
    top_k_categorical_accuracy = keras_metrics.top_k_categorical_accuracy

