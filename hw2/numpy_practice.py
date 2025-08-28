import numpy


p = [numpy.zeros((10, 10), dtype=numpy.float32) for _ in range(0, 10)]
p = numpy.array(p)
assert isinstance(p, numpy.ndarray)

print(1)
