import unittest
import label_perturbation_main
import SimpleRNN


class TestAttack( unittest.TestCase ):
	def test_attack(self):
		change_unit = 0.5
		precision4model1, recall4model1, fscore4model1, auc4model1 = label_perturbation_main.run_experiment(algo)
		precision4model2, recall4model2, fscore4model2, auc4model2 = label_perturbation_main.run_random_perturbation_experiment(change_unit, algo)
		self.assertEqual(auc4model1, auc4model2, "DECREASE IN AUC VALUE ... POSSIBLE ATTACK?"  )
