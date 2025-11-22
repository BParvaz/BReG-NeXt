import tensorflow as tf
import os

# ---------------------------
# CONFIG
# ---------------------------
train_input = "/mnt/iusers01/fse-ugpgt01/compsci01/b84547bp/scratch/BReG-NeXt/tfrecords/training_FER2013_sample.tfrecords"
val_input   = "/mnt/iusers01/fse-ugpgt01/compsci01/b84547bp/scratch/BReG-NeXt/tfrecords/validation_FER2013_sample.tfrecords"

train_output = "training_single_class.tfrecords"
val_output   = "validation_single_class.tfrecords"

target_class = 0  # The class you want to keep

# ---------------------------
# TFRecord feature description
# ---------------------------
feature_description = {
    'image_raw': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
}

# ---------------------------
# Helper functions
# ---------------------------
def parse_example(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)

def write_filtered_tfrecord(input_path, output_path, target_class):
    raw_dataset = tf.data.TFRecordDataset(input_path)
    parsed_dataset = raw_dataset.map(parse_example)
    filtered_dataset = parsed_dataset.filter(lambda x: x['label'] == target_class)

    with tf.io.TFRecordWriter(output_path) as writer:
        for record in filtered_dataset:
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['image_raw'].numpy()])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[record['label'].numpy()]))
            }))
            writer.write(example.SerializeToString())
    print(f"Saved filtered TFRecord: {output_path}")

# ---------------------------
# Run filtering
# ---------------------------
write_filtered_tfrecord(train_input, train_output, target_class)
write_filtered_tfrecord(val_input, val_output, target_class)