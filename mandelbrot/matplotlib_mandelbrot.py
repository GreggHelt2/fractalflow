import matplotlib as mpl
# if want to specify backend, MUST do matplotlib.use() BEFORE importing matplotlib.pyplot
#  (see https://matplotlib.org/faq/usage_faq.html)
# mpl.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import colors

import numpy as np
import tensorflow as tf
import time
import sys

class Timer:
    def __init__(self):
        self.perf_start = time.perf_counter()
        self.proc_start = time.process_time()
    def report(self):
        return str(round((time.perf_counter()-self.perf_start),3))
    def reset(self):
        result = self.report()
        self.perf_start = time.perf_counter()
        self.proc_start = time.process_time()
        return result

# def initGraph():
tim1 = Timer()
maxiter = 200
# numpy setup, result Z is 1D array of complex numbers
nY, nX = np.mgrid[-2.0:2.0:0.005, -2.0:2.0:0.005]
nZ = nX+1j*nY
print(type(nZ))
print(nZ.shape)
print("numpy: " + tim1.reset())

# sess.close()
# tf.reset_default_graph()
sess = tf.InteractiveSession()
print("sess startup:", tim1.reset())

tzoom = tf.placeholder(np.complex64, shape=())
toffset = tf.placeholder(np.complex64, shape=())
C = tf.Variable(nZ.astype(np.complex64))
C = C/tzoom + toffset
Z = C
M = tf.Variable(tf.zeros_like(C, tf.float32))

for i in range(maxiter):
    Z = Z*Z + C
    not_diverged = (tf.abs(Z) <= 4)
    M = M + tf.cast(not_diverged, tf.float32)

print("graph setup:", tim1.reset())
tf.global_variables_initializer().run()
print("graph init:", tim1.reset())

# turn off matplotlib toolbar
mpl.rcParams['toolbar'] = 'None'

# DPI, here, has _nothing_ to do with your screen's DPI.
dpi = 80.0
xpixels, ypixels = 800, 800
fig = plt.figure(figsize=(ypixels/dpi, xpixels/dpi), dpi=dpi)
gamma = 0.3
norm = colors.PowerNorm(gamma)

scale = 1.0
zoffset = 0.0 + 0.0j
zscaling = 1.0*scale + 0j
# zoffset = xoff + 1j * yoff
zoffset = 0.0 + 0.0j
result = M.eval(feed_dict={tzoom: zscaling, toffset: zoffset})
result[result >= maxiter] = 0

fimage = fig.figimage(result, cmap='hot', norm=norm)

def onclick(event):
    etim = Timer()
    global zscaling, zoffset
    # print(event)
    if (event.button == 1):
        zscaling = zscaling * 1.2
    elif (event.button == 3):
        zscaling = zscaling * (1.0/1.2)
    result = M.eval(feed_dict={tzoom: zscaling, toffset: zoffset})
    #print(etim.reset())
    rmax = result.max()
    # print(rmax)
    result[result>=maxiter] = 0
    #print(etim.reset())
    fimage.set_data(result)
    #print(etim.reset())
    # fimage.set_data(np.random.random((xpixels, ypixels)))
    plt.draw()
    #print(etim.reset())

def onkey(event):
    global zscaling, zoffset
    key = event.key
    # print('keypress:', event.key)
    sys.stdout.flush()   # is flush of stdout actually needed??
    if key=='a' or key=='left':
        zoffset = zoffset - (0.1 / zscaling)
    elif key=='d' or key=='right':
        zoffset = zoffset + (0.1 / zscaling)
    elif key=='w' or key=='up':
        zoffset = zoffset - (0.1j / zscaling)
    elif key=='s' or key=='down':
        zoffset = zoffset + (0.1j / zscaling)
    result = M.eval(feed_dict={tzoom: zscaling, toffset: zoffset})
    result[result>=maxiter] = 0
    fimage.set_data(result)
    plt.draw()

buttonid = fig.canvas.mpl_connect('button_press_event', onclick)
keyid = fig.canvas.mpl_connect('key_press_event', onkey)

plt.show()
