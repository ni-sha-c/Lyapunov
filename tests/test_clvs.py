import numpy as np
import unittest
import sys
sys.path.insert(0, '../examples')
sys.path.insert(0, '..')
from lorenz63 import Runner
from clv import CLV
class ClvTest(unittest.TestCase):
    def setUp(self):
        self.runner = Runner()
        self.CLV = CLV()

    def test_les(self):
        les = self.CLV.compute_les_and_clvs()
        print(les)

if __name__=="__main__":
    unittest.main()
