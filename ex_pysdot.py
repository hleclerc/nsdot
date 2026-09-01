import timeit

import numpy as np
from pysdot import PowerDiagram
from pysdot.domain_types import ConvexPolyhedraAssembly

positions = np.array( np.random.rand( 1_000_000, 2 ) )
domain = ConvexPolyhedraAssembly()
domain.add_box( [ 0, 0 ], [ 1, 1 ] )

# diracs
pd = PowerDiagram( positions, domain = domain )

start = timeit.timeit()
for d in range( 10 ):
    i = pd.integrals()
end = timeit.timeit()


print( sum( i ), ( end - start ) / 10 )
