# -*- coding: utf-8 -*-
""" Orange.data.Table related unit-tests
"""


try:
    import unittest2 as unittest
except:
    import unittest
from Orange.testing import testing

import Orange
import cPickle
import tempfile
import gc


def native(table):
    table = table.native()
    for i in range(len(table)):
        table[i] = [v.native() for v in table[i].native()]
    return table


def names_iter():
    for name in testing.ALL_DATASETS:
        yield name.replace(" ", "_").replace("-", "_"), (name,)


@testing.data_driven(data_iter=names_iter())
class TestLoading(unittest.TestCase):

    @testing.test_on_data
    def test_load_on(self, name):
        """ Test the loading of the data set
        """
        table = Orange.data.Table(name)
        self.assertIsNotNone(getattr(table, "attributeLoadStatus"),
                             "No attributeLoadStatus")

    @testing.test_on_data
    def test_pickling_on(self, name):
        """ Test data table pickling.
        """
        table = Orange.data.Table(name)
        s = cPickle.dumps(table)
        table_clone = cPickle.loads(s)

        self.assertSequenceEqual(list(table.domain),
                                 list(table_clone.domain))
        self.assertEqual(native(table), native(table_clone),
                         "Native representation is not equal!")


@testing.datasets_driven
class TestSaving(unittest.TestCase):
    @testing.test_on_data
    def test_R_on(self, name):
        data = Orange.data.Table(name)
        with tempfile.NamedTemporaryFile(suffix=".R") as f:
            data.save(f.name)

#    @testing.test_on_data
#    def test_toC50(self, name):
#        data = Orange.data.Table(name)

    @testing.test_on_datasets(datasets=testing.CLASSIFICATION_DATASETS + \
                              testing.REGRESSION_DATASETS)
    def test_arff_on(self, data):
        with tempfile.NamedTemporaryFile(suffix=".arff") as f:
            data.save(f.name)
            f.flush()
            data_arff = Orange.data.Table(f.name)

    @testing.test_on_datasets(datasets=testing.CLASSIFICATION_DATASETS + \
                              testing.REGRESSION_DATASETS)
    def test_svm_on(self, data):
        with tempfile.NamedTemporaryFile(suffix=".svm") as f:
            data.save(f.name)
            f.flush()
            data_svm = Orange.data.Table(f.name)

    @testing.test_on_datasets
    def test_csv_on(self, data):
        with tempfile.NamedTemporaryFile(suffix=".csv") as f:
            Orange.data.io.save_csv(f, data, dialect="excel-tab")
            f.flush()
            f.seek(0)
            Orange.data.io.load_csv(f, has_header=True,
                                    has_types=True, has_annotations=True)


@testing.datasets_driven
class TestUnicodeFilenames(unittest.TestCase):

    @testing.test_on_data
    def test_tab_on(self, name):
        """ Test the loading and saving to/from unicode (utf-8) filenames.
        """
        table = Orange.data.Table(name)
        with tempfile.NamedTemporaryFile(suffix=u"??-??-??.tab") as f:
            table.save(f.name)
            f.flush()
            table1 = Orange.data.Table(f.name)

    @testing.test_on_datasets(datasets=testing.CLASSIFICATION_DATASETS + \
                              testing.REGRESSION_DATASETS)
    def test_txt_on(self, name):
        """ Test the loading and saving to/from unicode (utf-8) filenames.
        """
        table = Orange.data.Table(name)
        with tempfile.NamedTemporaryFile(suffix=u"??-??-??.txt") as f:
            table.save(f.name)
            f.flush()
            table1 = Orange.data.Table(f.name)

    @testing.test_on_datasets(datasets=testing.CLASSIFICATION_DATASETS + \
                              testing.REGRESSION_DATASETS)
    def test_arff_on(self, name):
        """ Test the loading and saving to/from unicode (utf-8) filenames.
        """
        table = Orange.data.Table(name)
        with tempfile.NamedTemporaryFile(suffix=u"??-??-??.arff") as f:
            table.save(f.name)
            f.flush()
            table1 = Orange.data.Table(f.name)

    def test_basket(self):
        """ Test the loading and saving to/from unicode (utf-8) filenames.
        """
        table = Orange.data.Table("inquisition.basket")
        with tempfile.NamedTemporaryFile(suffix=u"??-??-??.basket") as f:
            table.save(f.name)
            f.flush()
            table1 = Orange.data.Table(f.name)


@testing.datasets_driven
class TestHashing(unittest.TestCase):

    @testing.test_on_data
    def test_uniqueness(self, name):
        """ Test the uniqueness of hashes. This is probabilistic,
        but if we hit a collision in one of documentation datasets,
        then it's time to open a bottle of Champagne ...
        """
        table = Orange.data.Table(name)
        self.assertEquals(len(set(table)), len(set(hash(i) for i in table)))

    @testing.test_on_data
    def test_repetitiveness(self, name):
        """ Test whether a data instance gets the same hash twice.
        """
        table = Orange.data.Table(name)
        a = [hash(i) for i in table]
        # Copy and reverse the table prior to hashing - just to hopefully
        # make more bugs stand out.
        b = list(reversed([hash(i) for i in
                           reversed(Orange.data.Table(table))]))

        self.assertEquals(a, b)


class TestDataOwnership(unittest.TestCase):
    def test_clone(self):
        """Test that `clone` method returns a table with it's own copy
        of the data.

        """
        iris = Orange.data.Table("iris")
        clone = iris.clone()

        self.assertTrue(iris.owns_instances and clone.owns_instances)
        self.assertTrue(all(e1.reference() != e2.reference()
                            for e1, e2 in zip(iris, clone)))

        clone[0][0] = -1
        self.assertTrue(iris[0][0] != clone[0][0])

        del clone
        gc.collect()

    def test_reference(self):
        iris = Orange.data.Table("iris")

        ref = Orange.data.Table(iris, True)

        self.assertTrue(iris.owns_instances)
        self.assertFalse(ref.owns_instances)

        self.assertTrue(all(e1.reference() == e2.reference()
                            for e1, e2 in zip(iris, ref)))

        ref[0][0] = -1
        self.assertEqual(iris[0][0], -1)

        with self.assertRaises(TypeError):
            ref.append(
                Orange.data.Instance(ref.domain,
                                     [0, 0, 0, 0, "Iris-setosa"])
            )

        del ref
        gc.collect()


class TestConstructor(unittest.TestCase):
    def test_from_list(self):
        A = [[0., 1., 2.],
             [3., 4., 5.],
             [6., 7., 8.]]

        domain = Orange.data.Domain(
            [Orange.feature.Continuous("F%i" % i)
             for i in range(1, 4)],
            None
        )

        data = Orange.data.Table(domain, A)

        self.assertEqual(native(data), A)

        lenses = Orange.data.Table("lenses")
        domain = lenses.domain

        A = [[0, 1, 0, 1, 0],
             [2, 0, 1, 0, 2]]

        data = Orange.data.Table(domain, A)

        self.assertEqual(
            A,
            [[int(v) for v in row] for row in data]
        )

        A[0][0] = 4
        with self.assertRaises(ValueError):
            data = Orange.data.Table(domain, A)

        A[0][0] = -1
        with self.assertRaises(ValueError):
            data = Orange.data.Table(domain, A)

    def test_from_array(self):
        """Test Table construction from numpy arrays.
        """
        from numpy import array, arange

        A = arange(9).reshape(3, 3)
        data = Orange.data.Table(A)

        self.assertTrue(all(isinstance(f, Orange.feature.Continuous)
                            for f in data.domain))
        self.assertEqual(data.domain.class_var, None)

        self.assertEqual(native(data), A.tolist())

        lenses = Orange.data.Table("lenses")
        domain = lenses.domain

        A = [[0, 1, 0, 1, 0],
             [2, 0, 1, 0, 2]]

        data = Orange.data.Table(domain, array(A))

        self.assertEqual(
            A,
            [[int(v) for v in row] for row in data]
        )

        A[0][0] = 4
        with self.assertRaises(ValueError):
            data = Orange.data.Table(domain, array(A))

        A[0][0] = -1
        with self.assertRaises(ValueError):
            data = Orange.data.Table(domain, array(A))

    def test_from_masked_array(self):
        """Test Table construction from numpy masked arrays.
        """
        from numpy.ma import masked_array, masked, arange

        A = arange(9).reshape(3, 3)
        A[1, 1] = masked

        data = Orange.data.Table(A)

        self.assertTrue(all(isinstance(f, Orange.feature.Continuous)
                            for f in data.domain))
        self.assertEqual(data.domain.class_var, None)

        self.assertTrue(data[1][1].isSpecial())

        lenses = Orange.data.Table("lenses")
        domain = lenses.domain

        A = masked_array([[0, 1, 0, 1, 0],
                          [2, 0, 1, 0, 2]])

        data = Orange.data.Table(domain, A)

        self.assertEqual(
            A.tolist(),
            [[int(v) for v in row] for row in data]
        )

        A[0, 0] = 4
        with self.assertRaises(ValueError):
            data = Orange.data.Table(domain, A)

        A[0, 0] = -1
        with self.assertRaises(ValueError):
            data = Orange.data.Table(domain, A)

        A[0, 0] = masked
        data = Orange.data.Table(domain, A)

        self.assertTrue(data[0][0].isSpecial())


if __name__ == "__main__":
    unittest.main()
