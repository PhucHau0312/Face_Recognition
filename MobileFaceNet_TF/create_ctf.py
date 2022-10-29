import cv2
import sklearn.preprocessing
import argparse
import pickle 
import numpy as np
import tensorflow as tf
import re 
import os
import sys
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

    parser = argparse.ArgumentParser(description='take a picture')
    parser.add_argument('--name','-n', type=str,help='input the name of the recording person')
    parser.add_argument('--model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
                        default='./arch/pretrained_model')
    parser.add_argument('--emb_path', default='embeddings', help='the embeddings path')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    return parser.parse_args(argv)

def main(args):

    emb_path = Path(args.emb_path)
    print(emb_path)
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)
    centerface = CenterFace()

    with tf.Session() as sess:
        load_model(args.model)
        inputs = tf.get_default_graph().get_tensor_by_name("input:0")
        embedding = tf.get_default_graph().get_tensor_by_name("embeddings:0")
		
        while cap.isOpened():   
            success,frame = cap.read()    
            hf, wf = frame.shape[:2]
            
            if success:   
                cv2.putText(frame,
			    'Press t to take a picture,q to quit.....',
			    (10,100), 
			    cv2.FONT_HERSHEY_SIMPLEX, 
			    2,
			    (0,255,0),
			    3,
			    cv2.LINE_AA)

            if cv2.waitKey(1)&0xFF == ord('t'):
                try:        
                    dets, _ = centerface(frame, hf, wf, threshold=0.35)
                except:
                            cv2.putText(frame, 'No face detected', (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                                    (0,255,0), 1, cv2.LINE_AA)
                for det in dets:
                    box, _ = det[:4], det[4]
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (2, 255, 0), 1)
                            
                (x1, y1, x2, y2) = (int(i) for i in box)
                (x, y, w, h) = (x1, y1, x2 - x1, y2 - y1)
                face = frame[y:y+h, x:x+w]
                cv2.imshow("My Face", face)

                face = cv2.resize(np.array(face), args.image_size)                
                face = (face - 127.5)*0.0078125
                face = np.expand_dims(face, 0)

                feed_dict = {inputs: face}
                embed = sess.run(embedding, feed_dict=feed_dict)
                embed = sklearn.preprocessing.normalize(embed)
                embed = embed.flatten()
                print(embed.shape)
                        
                with open(str(emb_path/('{}.pkl'.format(str(args.name)))), "wb") as f:
                        pickle.dump(embed, f)
					
            cv2.imshow("My Capture", frame)
            if cv2.waitKey(1)&0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
