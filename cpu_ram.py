import matplotlib.pyplot as plt 
import time
from psutil import cpu_percent, virtual_memory

# it is kind of difficult to measure the CPU/RAM usage 
# from within the program, so we first ran the program 
# and measured how long it took,
# then, the program and these two were run in parallel

start = time.time()
cpu = []
ram = []

t = 900
while(int(time.time() - start) < t):
    time.sleep(5)
    cpu.append((cpu_percent()))
    ram.append(int(virtual_memory()[2]))


end = time.time()

len = len(cpu); width = (int(end) - int(start))/len
x = []
j = 0
for i in range(0, len):
    x.append(j)
    j += width

plt.plot(x, cpu, color = 'red', label = 'CPU')
plt.plot(x, ram, color = 'green', label = 'RAM')

plt.xlabel('Time elapsed in seconds')
plt.ylabel('Percentage of CPU/RAM')
plt.title('CPU/RAM usage over time')
plt.legend()

plt.show()