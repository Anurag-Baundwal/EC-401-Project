# SLOTTED ALOHA
# dur = G/N
# slotnumber in which packet with arrival time t falls is floor(t/dur)
# only slots where there is exactly 1 packet is successful

# A substantially simpler approach is as follows
# Recall that the difference between plainALOHA and slotted ALOHA 
# is that the vulnerable period is twice as high for plain ALOHA
# This can be approximated by saying that we only need clearance on one side
# (next packet should be more than dur away, 
# in plain ALOHA both  next AND previous packets should be distance dur away)

N=100000; # Number of packets in unit time
PacketArrivalTimes=[random.randrange(1,N) for i in range(N)];  #random numbers between 0 and 1
G = []
sucslot = [] # slots in which transmission was successful
for m in range(1:101):
  G[m]=0.2+m/50; #varying G between 0.2 and 2.2 in 100 steps
  dur=G[m]/N;
  NumberSuccessful = sum((y1>dur)); #checking only one side
  sucslot[m] = NumberSuccessful*dur; 
end