"""

Merge logs from tensorboard if breaks happened during training
Input arg: path to folder with logs

"""

import tensorflow as tf
import sys

steps = []
steps_val = []

events = {}
events_val = {}

files = os.listdir(sys.argv[1])
summary_writer = tf.summary.FileWriter(sys.argv[1]+'/merged')

for f in files:
    for event in tf.train.summary_iterator(f):
        for value in event.summary.value:
            if value.HasField('simple_value') and value.tag == "loss" and not event.step in steps:
                steps.append(event.step)
                events[int(event.step)] = event
            if value.tag == "val_loss" and not event.step in steps_val:
                steps_val.append(event.step)
                events_val[int(event.step)] = event
                
for i in range(max(events.keys())):
    summary_writer.add_event(events[i])
    summary_writer.add_event(events_val[i])
