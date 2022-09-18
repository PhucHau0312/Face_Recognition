import pickle
import cv2
import numpy as np
import tensorflow as tf
import os 
import re 
import sys
import argparse 
import glob 
from pathlib import Path
from CenterFace.prj_python.centerface import CenterFace


def load_model(model):

    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with tf.gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='begin recognition')
    parser.add_argument('--model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
                        default='./arch/pretrained_model')
    parser.add_argument('--image_size', default=[1, 112, 112, 3], help='the image size')
    parser.add_argument('--embedding_size', default=[192, 1], help='the embedding size')
    return parser.parse_args(argv)


def main(args):
    data_path = '/home/hau/Desktop/FaceRecognition/MobileFaceNet_TF/embedding_pkl/'
    all_embed = glob.glob(data_path + '*.pkl')
    
    cap = cv2.VideoCapture(0)

    cap.set(3,1280)
    cap.set(4,720)
    centerface = CenterFace()
    
    with tf.Session() as sess:  
        load_model(args.model)
        inputs = tf.get_default_graph().get_tensor_by_name("input:0")
        embedding = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        
        while cap.isOpened():

            isSuccess,frame = cap.read()
            img = frame.copy()
            hf, wf = frame.shape[:2]

            if isSuccess:   
                cv2.putText(frame,
		                    'Press q to quit.....',
		                    (400,50), 
		                    cv2.FONT_HERSHEY_SIMPLEX, 
		                    2,
		                    (0,255,0),
		                    3,
		                    cv2.LINE_AA)

            dets, lms = centerface(frame, hf, wf, threshold=0.35)
            bboxes = []
            for box in dets:
                (x1, y1, x2, y2) = (int(box[i]) for i in range(4))
                bboxes.append([x1, y1, x2 - x1, y2 - y1])
            
            # for lm in lms:
            #     for i in range(0, 5):
            #         cv2.circle(frame, (int(lm[i * 2]), int(lm[i * 2 + 1])), 2, (0, 0, 255), -1)
            
            # if cv2.waitKey(1)&0xFF == ord('r'):
            try:                    
                for box in bboxes:
                    print(box)
                    [x, y, w, h] = box
                    face = img[y:y+h, x:x+w]
                    face = cv2.resize(face, [112, 112])   
                    face = face.reshape(args.image_size)

                    feed_dict = {inputs: face}
                    embed_box = sess.run(embedding, feed_dict=feed_dict)
                    
                    embed_box = np.array(embed_box).reshape([embed_box.shape[1]])
                    print(embed_box.shape)
                    
                    minimum = 99 
                    person = None 
                    for emb in all_embed:
                        with open(emb, 'rb') as f:
                            embed = pickle.load(f)

                        name = emb[len(data_path):]
                        name = name.split('.')[0]
                        print(name)

                        dist = np.linalg.norm(embed_box - embed)
                        print(dist)
                        
                        if dist < minimum:
                            minimum = dist
                            person = name
                            
                    print(person)

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (2, 255, 0), 2)

                    cv2.putText(frame, str(person), (int(x+w/2-25), int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 
                                        1, (0, 255, 0), 2, cv2.LINE_AA)                        
            except:     
                print('detect error') 
                
            cv2.imshow("My Capture", frame)
            if cv2.waitKey(1)&0xFF == ord('q'):
                break
        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
