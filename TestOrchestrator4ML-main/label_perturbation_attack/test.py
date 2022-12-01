import unittest
import knn
import main
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier

common_error_string = 'DOES NOT MATCH:::Should be '

class TestKnn( unittest.TestCase ):
    
    def test_euc_dist(self):     
        oracle_value = 2
        dist      = knn.euc_dist( 4,2 ) 
        self.assertEqual( oracle_value,  dist , common_error_string + str(oracle_value)  ) 
        
    def test_predict(self):
        mnist = load_digits()
        X = mnist.data 
        y = mnist.target
        
        oracle_value = y[0]
        model = KNeighborsClassifier(n_neighbors = 1)
        model.fit(X, y)
        pred = model.predict(X[0].reshape(1, -1))
        self.assertEqual( oracle_value,  pred , common_error_string + str(oracle_value)  ) 
        

class TestAttack( unittest.TestCase ):
    
    def test_attack(self):     
        precision_before, recall_before, fscore_before = main.run_experiment()
        precision_after, recall_after, fscore_after = main.run_random_perturbation_experiment()
        self.assertGreater( precision_before,  precision_after , "precision should reduce after attack" ) 

if __name__=='__main__':
    unittest.main()