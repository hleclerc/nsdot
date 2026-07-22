from sdot import Tensor
import numpy as np

if True:
    # Test rank-1 tensor iteration with strides
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    t = Tensor._from_numpy(data)
    
    # This should compile and work
    print("Tensor iterator test (basic)")
    print(f"Tensor data: {data}")
