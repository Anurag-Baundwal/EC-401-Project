#To Do:

# Implement devices:
They will have a specific power
They will have a specific distance (random btw 0 and 10)
They will have a q table (with timeslot-channel pairs as the state-action pairs)

# Transmission model:
Path loss - function of distance
Devices will transmit in a specific timeslot channel pair after looking at values in their Q-Table

# SIC Decoder:
Receive messages (k messages)
Sort in decreasing order of SINR
k messages were received
for i in range(k):
    sinr will be a function of i and k
    check whether sinr >= 2**beta - 1

D -> m -> R -> fb -> 