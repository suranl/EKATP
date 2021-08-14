import pickle
import numpy as np
np.set_printoptions(threshold=np.inf) #解决显示不完全问题

f = open('model.pkl','rb')
inf = pickle.load(f)
f.close()
inf = str(inf)
ft = open("print.txt",'w')
ft.write(inf)