from applications.reconstruction.viz.live import LiveSimulation

live = LiveSimulation("reconstruction-lung")
canvas = live.canvas( "positions", extent=1.0, point_radius=0.002 )
print(canvas.url)

canvas.update([ [ 0, 0 ], [ 1, 0 ], [ 0, .1 ] ], step = 0 )
canvas.update([ [ 0, 0 ], [ 1, 0 ], [ 0, 1 ] ], step = 1 )
