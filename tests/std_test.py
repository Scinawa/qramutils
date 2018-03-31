import unittest 

from qramutils import QramUtils

class TestStatistics(unittest.TestCase): 
  def t_forbenius_norm():

    A = QramUtils()


    self.assertEqual(A.frobenius(), 5)   # Y U NO PEP8 unittest library!!!?!?!?!??


  def t_sparsity():

    Test_1 = np.zero(shape(10,10))
    Test_2 = np.zero(shape(10,10))

    self.assertEqual(QramUtils.sparsity(Test_1), 0)
    self.assertEqual(QramUtils.sparsity(Test_2), 1)




  def t_find_p():
    self.assertEqual()




  def t_s():
    self.assertEqual()




  def t_hacks():
    pass




if __name__ == '__main__':
  unittest.main()
