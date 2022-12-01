"""
Feature calculations.
"""
import inspect
import logging
import numpy as np
import multiprocessing
from typing import Any, Dict, List, Iterable, Sequence, Tuple, Union

from deepchem.utils import get_print_threshold
from deepchem.utils.typing import PymatgenStructure

logger = logging.getLogger(__name__)


class Featurizer(object):
  """Abstract class for calculating a set of features for a datapoint.

  This class is abstract and cannot be invoked directly. You'll
  likely only interact with this class if you're a developer. In
  that case, you might want to make a child class which
  implements the `_featurize` method for calculating features for
  a single datapoints if you'd like to make a featurizer for a
  new datatype.
  """

  def featurize(self, datapoints: Iterable[Any],
                log_every_n: int = 1000) -> np.ndarray:
    """Calculate features for datapoints.

    Parameters
    ----------
    datapoints: Iterable[Any]
      A sequence of objects that you'd like to featurize. Subclassses of
      `Featurizer` should instantiate the `_featurize` method that featurizes
      objects in the sequence.
    log_every_n: int, default 1000
      Logs featurization progress every `log_every_n` steps.

    Returns
    -------
    np.ndarray
      A numpy array containing a featurized representation of `datapoints`.
    """
    datapoints = list(datapoints)
    features = []
    for i, point in enumerate(datapoints):
      if i % log_every_n == 0:
        logger.info("Featurizing datapoint %i" % i)
      try:
        features.append(self._featurize(point))
      except:
        logger.warning(
            "Failed to featurize datapoint %d. Appending empty array")
        features.append(np.array([]))

    features = np.asarray(features)
    return features

  def __call__(self, datapoints: Iterable[Any]):
    """Calculate features for datapoints.

    Parameters
    ----------
    datapoints: Iterable[Any]
      Any blob of data you like. Subclasss should instantiate this.
    """
    return self.featurize(datapoints)

  def _featurize(self, datapoint: Any):
    """Calculate features for a single datapoint.

    Parameters
    ----------
    datapoint: Any
      Any blob of data you like. Subclass should instantiate this.
    """
    raise NotImplementedError('Featurizer is not defined.')

  def __repr__(self) -> str:
    """Convert self to repr representation.

    Returns
    -------
    str
      The string represents the class.

    Examples
    --------
    >>> import deepchem as dc
    >>> dc.feat.CircularFingerprint(size=1024, radius=4)
    CircularFingerprint[radius=4, size=1024, chiral=False, bonds=True, features=False, sparse=False, smiles=False]
    >>> dc.feat.CGCNNFeaturizer()
    CGCNNFeaturizer[radius=8.0, max_neighbors=12, step=0.2]
    """
    args_spec = inspect.getfullargspec(self.__init__)  # type: ignore
    args_names = [arg for arg in args_spec.args if arg != 'self']
    args_info = ''
    for arg_name in args_names:
      value = self.__dict__[arg_name]
      # for str
      if isinstance(value, str):
        value = "'" + value + "'"
      # for list
      if isinstance(value, list):
        threshold = get_print_threshold()
        value = np.array2string(np.array(value), threshold=threshold)
      args_info += arg_name + '=' + str(value) + ', '
    return self.__class__.__name__ + '[' + args_info[:-2] + ']'

  def __str__(self) -> str:
    """Convert self to str representation.

    Returns
    -------
    str
      The string represents the class.

    Examples
    --------
    >>> import deepchem as dc
    >>> str(dc.feat.CircularFingerprint(size=1024, radius=4))
    'CircularFingerprint_radius_4_size_1024'
    >>> str(dc.feat.CGCNNFeaturizer())
    'CGCNNFeaturizer'
    """
    args_spec = inspect.getfullargspec(self.__init__)  # type: ignore
    args_names = [arg for arg in args_spec.args if arg != 'self']
    args_num = len(args_names)
    args_default_values = [None for _ in range(args_num)]
    if args_spec.defaults is not None:
      defaults = list(args_spec.defaults)
      args_default_values[-len(defaults):] = defaults

    override_args_info = ''
    for arg_name, default in zip(args_names, args_default_values):
      if arg_name in self.__dict__:
        arg_value = self.__dict__[arg_name]
        # validation
        # skip list
        if isinstance(arg_value, list):
          continue
        if isinstance(arg_value, str):
          # skip path string
          if "\\/." in arg_value or "/" in arg_value or '.' in arg_value:
            continue
        # main logic
        if default != arg_value:
          override_args_info += '_' + arg_name + '_' + str(arg_value)
    return self.__class__.__name__ + override_args_info


class ComplexFeaturizer(object):
  """"
  Abstract class for calculating features for mol/protein complexes.
  """

  def featurize(self, mol_files: Sequence[str],
                protein_pdbs: Sequence[str]) -> Tuple[np.ndarray, List]:
    """
    Calculate features for mol/protein complexes.

    Parameters
    ----------
    mols: List[str]
      List of PDB filenames for molecules.
    protein_pdbs: List[str]
      List of PDB filenames for proteins.

    Returns
    -------
    features: np.ndarray
      Array of features
    failures: List
      Indices of complexes that failed to featurize.
    """

    pool = multiprocessing.Pool()
    results = []
    for i, (mol_file, protein_pdb) in enumerate(zip(mol_files, protein_pdbs)):
      log_message = "Featurizing %d / %d" % (i, len(mol_files))
      results.append(
          pool.apply_async(ComplexFeaturizer._featurize_callback,
                           (self, mol_file, protein_pdb, log_message)))
    pool.close()
    features = []
    failures = []
    for ind, result in enumerate(results):
      new_features = result.get()
      # Handle loading failures which return None
      if new_features is not None:
        features.append(new_features)
      else:
        failures.append(ind)
    features = np.asarray(features)
    return features, failures

  def _featurize(self, mol_pdb: str, complex_pdb: str):
    """
    Calculate features for single mol/protein complex.

    Parameters
    ----------
    mol_pdb : str
      The PDB filename.
    complex_pdb : str
      The PDB filename.
    """
    raise NotImplementedError('Featurizer is not defined.')

  @staticmethod
  def _featurize_callback(featurizer, mol_pdb_file, protein_pdb_file,
                          log_message):
    logging.info(log_message)
    return featurizer._featurize(mol_pdb_file, protein_pdb_file)


class MolecularFeaturizer(Featurizer):
  """Abstract class for calculating a set of features for a
  molecule.

  The defining feature of a `MolecularFeaturizer` is that it
  uses SMILES strings and RDKit molecule objects to represent
  small molecules. All other featurizers which are subclasses of
  this class should plan to process input which comes as smiles
  strings or RDKit molecules.

  Child classes need to implement the _featurize method for
  calculating features for a single molecule.

  Notes
  -----
  The subclasses of this class require RDKit to be installed.
  """

  def featurize(self, molecules, log_every_n=1000) -> np.ndarray:
    """Calculate features for molecules.

    Parameters
    ----------
    molecules: rdkit.Chem.rdchem.Mol / SMILES string / iterable
      RDKit Mol, or SMILES string or iterable sequence of RDKit mols/SMILES
      strings.
    log_every_n: int, default 1000
      Logging messages reported every `log_every_n` samples.

    Returns
    -------
    features: np.ndarray
      A numpy array containing a featurized representation of `datapoints`.
    """
    try:
      from rdkit import Chem
      from rdkit.Chem import rdmolfiles
      from rdkit.Chem import rdmolops
      from rdkit.Chem.rdchem import Mol
    except ModuleNotFoundError:
      raise ImportError("This class requires RDKit to be installed.")

    # Special case handling of single molecule
    if isinstance(molecules, str) or isinstance(molecules, Mol):
      molecules = [molecules]
    else:
      # Convert iterables to list
      molecules = list(molecules)

    features = []
    for i, mol in enumerate(molecules):
      if i % log_every_n == 0:
        logger.info("Featurizing datapoint %i" % i)

      try:
        if isinstance(mol, str):
          # mol must be a RDKit Mol object, so parse a SMILES
          mol = Chem.MolFromSmiles(mol)
          # SMILES is unique, so set a canonical order of atoms
          new_order = rdmolfiles.CanonicalRankAtoms(mol)
          mol = rdmolops.RenumberAtoms(mol, new_order)

        features.append(self._featurize(mol))
      except Exception as e:
        if isinstance(mol, Chem.rdchem.Mol):
          mol = Chem.MolToSmiles(mol)
        logger.warning(
            "Failed to featurize datapoint %d, %s. Appending empty array", i,
            mol)
        logger.warning("Exception message: {}".format(e))
        features.append(np.array([]))

    features = np.asarray(features)
    return features


class MaterialStructureFeaturizer(Featurizer):
  """
  Abstract class for calculating a set of features for an
  inorganic crystal structure.

  The defining feature of a `MaterialStructureFeaturizer` is that it
  operates on 3D crystal structures with periodic boundary conditions.
  Inorganic crystal structures are represented by Pymatgen structure
  objects. Featurizers for inorganic crystal structures that are subclasses of
  this class should plan to process input which comes as pymatgen
  structure objects.

  This class is abstract and cannot be invoked directly. You'll
  likely only interact with this class if you're a developer. Child
  classes need to implement the _featurize method for calculating
  features for a single crystal structure.

  Notes
  -----
  Some subclasses of this class will require pymatgen and matminer to be
  installed.
  """

  def featurize(self,
                structures: Iterable[Union[Dict[str, Any], PymatgenStructure]],
                log_every_n: int = 1000) -> np.ndarray:
    """Calculate features for crystal structures.

    Parameters
    ----------
    structures: Iterable[Union[Dict, pymatgen.Structure]]
      Iterable sequence of pymatgen structure dictionaries
      or pymatgen.Structure. Please confirm the dictionary representations
      of pymatgen.Structure from https://pymatgen.org/pymatgen.core.structure.html.
    log_every_n: int, default 1000
      Logging messages reported every `log_every_n` samples.

    Returns
    -------
    features: np.ndarray
      A numpy array containing a featurized representation of
      `structures`.
    """
    try:
      from pymatgen import Structure
    except ModuleNotFoundError:
      raise ImportError("This class requires pymatgen to be installed.")

    structures = list(structures)
    features = []
    for idx, structure in enumerate(structures):
      if idx % log_every_n == 0:
        logger.info("Featurizing datapoint %i" % idx)
      try:
        if isinstance(structure, Dict):
          structure = Structure.from_dict(structure)
        features.append(self._featurize(structure))
      except:
        logger.warning(
            "Failed to featurize datapoint %i. Appending empty array" % idx)
        features.append(np.array([]))

    features = np.asarray(features)
    return features


class MaterialCompositionFeaturizer(Featurizer):
  """
  Abstract class for calculating a set of features for an
  inorganic crystal composition.

  The defining feature of a `MaterialCompositionFeaturizer` is that it
  operates on 3D crystal chemical compositions.
  Inorganic crystal compositions are represented by Pymatgen composition
  objects. Featurizers for inorganic crystal compositions that are
  subclasses of this class should plan to process input which comes as
  Pymatgen composition objects.

  This class is abstract and cannot be invoked directly. You'll
  likely only interact with this class if you're a developer. Child
  classes need to implement the _featurize method for calculating
  features for a single crystal composition.

  Notes
  -----
  Some subclasses of this class will require pymatgen and matminer to be
  installed.
  """

  def featurize(self, compositions: Iterable[str],
                log_every_n: int = 1000) -> np.ndarray:
    """Calculate features for crystal compositions.

    Parameters
    ----------
    compositions: Iterable[str]
      Iterable sequence of composition strings, e.g. "MoS2".
    log_every_n: int, default 1000
      Logging messages reported every `log_every_n` samples.

    Returns
    -------
    features: np.ndarray
      A numpy array containing a featurized representation of
      `compositions`.
    """
    try:
      from pymatgen import Composition
    except ModuleNotFoundError:
      raise ImportError("This class requires pymatgen to be installed.")

    compositions = list(compositions)
    features = []
    for idx, composition in enumerate(compositions):
      if idx % log_every_n == 0:
        logger.info("Featurizing datapoint %i" % idx)
      try:
        c = Composition(composition)
        features.append(self._featurize(c))
      except:
        logger.warning(
            "Failed to featurize datapoint %i. Appending empty array" % idx)
        features.append(np.array([]))

    features = np.asarray(features)
    return features


class UserDefinedFeaturizer(Featurizer):
  """Directs usage of user-computed featurizations."""

  def __init__(self, feature_fields):
    """Creates user-defined-featurizer."""
    self.feature_fields = feature_fields
