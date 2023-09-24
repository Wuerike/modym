import pyfmi
from matplotlib import pyplot as plt 

pyfmi.check_packages()

fmu = pyfmi.load_fmu('./cartpole/models/ModelicaGym_CartPole_CS.fmu')

fmu.set('f',1)
res=fmu.simulate(0., 1.0)

plt.title("Matplotlib demo") 
plt.plot(res['time'], res['x'])
plt.plot(res['time'], res['x_dot'])
plt.show()