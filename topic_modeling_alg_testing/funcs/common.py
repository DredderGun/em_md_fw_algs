class SmoothFunction:
    def __call__(self, x):
        assert 0, "SmoothFunction: __call__(x) is not defined"
        
    def gradient(self, x):
        assert 0, "SmoothFunction: gradient(x) is not defined"
 
    def func_grad(self, x, flag):
        """
        flag=0: function, flag=1: gradient, flag=2: function & gradient 
        """
        assert 0, "SmoothFunction: func_grad(x, flag) is not defined"