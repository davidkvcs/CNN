"""

Merge logs from tensorboard if breaks happened during training
Input arg: path to folder with logs

"""

import tensorflow as tf
import sys
import os

events = {}
events_val = {}

files = os.listdir(sys.argv[1])
files.reverse()
summary_writer = tf.summary.FileWriter(sys.argv[1]+'/merged')

for f in files:
    for event in tf.train.summary_iterator(sys.argv[1]+'/'+f):
        for value in event.summary.value:
            if value.tag == "loss" and not int(event.step) in events:
                events[int(event.step)] = event
            if value.tag == "val_loss" and not int(event.step) in events_val:
                events_val[int(event.step)] = event
                
for i in range(max(events.keys())):
    print(i)
    summary_writer.add_event(events[i])
    summary_writer.add_event(events_val[i])
