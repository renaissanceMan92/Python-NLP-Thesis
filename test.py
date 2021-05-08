import matplotlib.pyplot as plt
'''
10 topics has coherence -1.4676461775384568
15 topics has coherence -1.6204421809734877
20 topics has coherence -1.643742986823063
25 topics has coherence -1.7189897431679242
30 topics has coherence -1.988051516763304
35 topics has coherence -1.8051987537160774
40 topics has coherence -2.048730711898405
'''

test = dict()
test[10]= -1.4676461775384568
test[15]= -1.6204421809734877
test[20]= -1.643742986823063
test[25]= -1.7189897431679242
test[30]= -1.988051516763304
test[35]= -1.8051987537160774
test[40]= -2.048730711898405
for k,v in test.items():
    print(k,'::',v)

plt.plot(test.keys(),test.values())
plt.xlabel('Nr. of topics')
plt.ylabel('Coherence score')
ax = plt.gca()
ax.invert_xaxis()
plt.show()