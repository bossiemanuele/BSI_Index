import unittest
import numpy as np
from scipy.spatial.distance import jensenshannon
from bsi import distributions_bsi, labeled_bsi, from_extents

class TestBSI(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(distributions_bsi([1,0],[0,1]),0)
        self.assertEqual(distributions_bsi([.7,.3],[.7,.3]),1)
        self.assertAlmostEqual(distributions_bsi([1,0],[.5,.5]), .6887218755408672)
        self.assertEqual(distributions_bsi([1,0,0],[1,0,0]),1)
    def test_independent_implementation(self):
        rng=np.random.default_rng(81)
        for _ in range(100):
            p,q=rng.dirichlet(np.ones(7),size=2)
            self.assertAlmostEqual(distributions_bsi(p,q),1-jensenshannon(p,q,base=2)**2)
    def test_covariance_and_affine(self):
        rng=np.random.default_rng(5)
        X=rng.normal(size=(400,3)); y=np.repeat([0,1],[100,300]); X[y==1]*=2
        r=labeled_bsi(X,y)
        volumes=np.array([np.sqrt(np.linalg.det(np.cov(X[y==k].T))) for k in [0,1]])
        np.testing.assert_allclose(r['q'],volumes/volumes.sum())
        A=np.array([[2,.2,0],[0,1,.3],[.1,0,.5]])
        t=labeled_bsi(X@A.T+[3,-4,8],y)
        np.testing.assert_allclose(r['q'],t['q'],atol=1e-12)
        self.assertAlmostEqual(r['bsi'],labeled_bsi(X,np.where(y==0,7,2))['bsi'])
    def test_failures(self):
        for p,q in [([2,0],[0,1]),([-1,2],[1,0]),([np.nan,1],[1,0]),([],[])]:
            with self.assertRaises(ValueError): distributions_bsi(p,q)
        with self.assertRaises(ValueError): labeled_bsi(np.zeros((12,2)),np.repeat([0,1],6))
        with self.assertRaises(ValueError): from_extents([1,1],[0,0])
    def test_additive_coarsening(self):
        rng=np.random.default_rng(8)
        for _ in range(100):
            p,q=rng.dirichlet(np.ones(8),size=2)
            self.assertGreaterEqual(distributions_bsi(p.reshape(4,2).sum(1),q.reshape(4,2).sum(1))+1e-14,distributions_bsi(p,q))
if __name__=='__main__': unittest.main()
