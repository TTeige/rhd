import tensorflow as tf
import numpy as np
import os
import argparse
import cv2


def convert_img(img):
    img_data = np.array(img, dtype=np.float32)
    img_data = img_data.flatten()
    img_data = [float(x) * 1.0 / 255.0 for x in img_data]
    img_data = np.reshape(img_data, [64, 64])
    img_data = [img_data]
    img_data = np.expand_dims(img_data, axis=3)
    img_data = img_data.tolist()
    return img_data


def run():
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph(args.meta_graph)
        if args.model != "":
            pass
        if args.checkpoint != "":
            saver.restore(session, tf.train.latest_checkpoint(args.checkpoint))
        print("Model restored")
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('x:0')
        y_pred_cls = graph.get_tensor_by_name('y_pred_cls:0')

        w_c3 = graph.get_tensor_by_name('w_fc1:0')
        #
        predictor = tf.transpose(tf.argmax(w_c3))

        feed_dict = {
            x: convert_img(cv2.imread("/mnt/remote/extracted_fields/fs10061402175187/4_27fs10061402175187.jpg", 0))
        }
        print(session.run(predictor, feed_dict=feed_dict))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description="Predict images at the given path. Model based on a synthetic mnist dataset")
    arg_parser.add_argument("--images", "-i", type=str,
                            help="path to root of directory structure that is to be predicted")
    arg_parser.add_argument("--output", "-o", type=str, help="Output path of csv file containing the predictions")
    arg_parser.add_argument("--model", "-m", type=str, help="Path to the model that is to be restored", default="")
    arg_parser.add_argument("--meta_graph", "-M", type=str, help="Path to the metagraph for restoration of the model",
                            default="")
    arg_parser.add_argument("--checkpoint", "-c", type=str, help="Restore from checkpoint if path is provided",
                            default="")
    args = arg_parser.parse_args()
    print(args)
    run()
