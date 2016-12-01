
def read_and_decode(filepath):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filepath)
    features = tf.parse_single_example(
    serialized_example,
    dense_keys=['image_raw', 'label'],
    # Defaults are not specified since both keys are required.
    dense_types=[tf.string, tf.FloatList])

    # Convert from a scalar string tensor (whose single string has
    image = tf.decode_raw(features['image_raw'], tf.uint8)

#     image = tf.reshape(image, [my_cifar.n_input])
#     image.set_shape([my_cifar.n_input])

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    #label = tf.cast(features['label'], tf.int32)

    return image, label