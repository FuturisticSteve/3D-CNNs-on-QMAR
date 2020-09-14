import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(8,6))
ax1 = fig.subplots()
ax2 = ax1.twinx()


loss_file = open(r"I:\results-flow-tvl1-resnet50\results-Parkinson-Sit-Stand-fold1\train.log", 'r')
epoch = []
loss = []
for i, line in enumerate(loss_file):
    if i != 0 and i % 2 == 0:
        line = line.split()
        epoch.append(int(line[0]))
        loss.append(round(float(line[1]),3))

ax2.plot(epoch,loss,label='fold1')

loss_file = open(r"I:\results-flow-tvl1-resnet50\results-Parkinson-Sit-Stand-fold2\train.log", 'r')
epoch = []
loss = []
for i, line in enumerate(loss_file):
    if i != 0 and i % 2 == 0:
        line = line.split()
        epoch.append(int(line[0]))
        loss.append(round(float(line[1]),3))

ax2.plot(epoch,loss,label='fold2')


loss_file = open(r"I:\results-flow-tvl1-resnet50\results-Parkinson-Sit-Stand-fold3\train.log", 'r')
epoch = []
loss = []
for i, line in enumerate(loss_file):
    if i != 0 and i % 2 == 0:
        line = line.split()
        epoch.append(int(line[0]))
        loss.append(round(float(line[1]),3))

ax2.plot(epoch,loss,label='fold3')

loss_file = open(r"I:\results-flow-tvl1-resnet50\results-Parkinson-Sit-Stand-fold4\train.log", 'r')
epoch = []
loss = []
for i, line in enumerate(loss_file):
    if i != 0 and i % 2 == 0:
        line = line.split()
        epoch.append(int(line[0]))
        loss.append(round(float(line[1]),3))

ax2.plot(epoch,loss,label='fold4')



input = open(r"I:\results-flow-tvl1-resnet50\results-Parkinson-Sit-Stand-fold1\test.log", 'r')
epoch = []
correlation = []
for i, line in enumerate(input):
    if i != 0 and i % 2 == 0:
        line = line.split()
        epoch.append(int(line[0]))
        correlation.append(float(line[1]))

ax1.plot(epoch,correlation,label='fold1')


input = open(r"I:\results-flow-tvl1-resnet50\results-Parkinson-Sit-Stand-fold2\test.log", 'r')
epoch = []
correlation = []
for i, line in enumerate(input):
    if i != 0 and i % 2 == 0:
        line = line.split()
        epoch.append(int(line[0]))
        correlation.append(float(line[1]))

ax1.plot(epoch,correlation,label='fold2')

input = open(r"I:\results-flow-tvl1-resnet50\results-Parkinson-Sit-Stand-fold3\test.log", 'r')
epoch = []
correlation = []
for i, line in enumerate(input):
    if i != 0 and i % 2 == 0:
        line = line.split()
        epoch.append(int(line[0]))
        correlation.append(float(line[1]))

ax1.plot(epoch,correlation,label='fold3')


input = open(r"I:\results-flow-tvl1-resnet50\results-Parkinson-Sit-Stand-fold4\test.log", 'r')
epoch = []
correlation = []
for i, line in enumerate(input):
    if i != 0 and i % 2 == 0:
        line = line.split()
        epoch.append(int(line[0]))
        correlation.append(float(line[1]))

ax1.plot(epoch,correlation,label='fold4')





ax1.legend(loc=0)

ax2.set_xticks(np.arange(1,1+epoch[-1],2))
ax1.set_yticks(np.arange(0,1.1,0.1))
ax2.set_yticks(np.arange(0,4.2,0.4))
ax1.set_xlabel('epoch')
ax1.set_ylabel('correlation')
ax2.set_ylabel('loss')
#plt.savefig('D:\Shared\\figures\\flow stream',dpi=400)
plt.show()