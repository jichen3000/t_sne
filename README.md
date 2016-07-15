# t_sne

Implementation of t-SNE in Python. 

More t-SNE information on: https://lvdmaaten.github.io/tsne/

## requirements

python 2.7
numpy
theano

## run mnist example

download the mnist2500\_X.txt and mnist2500\_labels.txt in the data fold.

using numpy
```bash
python tsne_numpy.py 1000
```

using theano
```bash
python tsne_theano.py 1000
```

using theano with GPU
```bash
THEANO_FLAGS=mode=FAST_RUN,device=gpu,lib.cnmem=1,floatX=float32 python tsne_theano.py 1000
```

## performance test

### test environment
OS:   Ubuntu 14.04.4 LTS
CPU:  Intel(R) Core(TM) i7-4790 CPU @ 3.60GHz, 8 cores.
GPU:  GeForce GTX 660

python:  2.7.10
theano:  0.8.2
numpy:  1.11.1

### result
bumpy code: 162 seconds
theano code on cpu: 72 seconds
theano code on gpu: 29 seconds
