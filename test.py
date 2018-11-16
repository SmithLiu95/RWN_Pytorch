import torch
import time

from RWN_original import RWN
try_time=10
for image_size in [64,128,256,512]:
    for R in [5]:
        model=RWN(n=image_size,R=R)
        model.eval()
        for batch_size in range(1,5):
            t0=time.time()
            for i in range(try_time):
                x=torch.randn((batch_size,3,image_size,image_size))
                y=model(x)
            print("{},{},{},{}".format(image_size,R,batch_size,(time.time()-t0)/try_time))

from RWN import RWN
try_time=10
for image_size in [64,128,256,512]:
    for R in [5]:
        model=RWN(n=image_size,R=R)
        model.eval()
        for batch_size in range(1,5):
            t0=time.time()
            for i in range(try_time):
                x=torch.randn((batch_size,3,image_size,image_size))
                y=model(x)
            print("{},{},{},{}".format(image_size,R,batch_size,(time.time()-t0)/try_time))