import cv2
import argparse
from pathlib import Path
import pickle 
import numpy as np
import tensorflow as tf
import re 
import os
import sys
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
    parser.add_argument('--name','-n', default='unknown', type=str,help='input the name of the recording person')
    parser.add_argument('--model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
                        default='./arch/pretrained_model')
    parser.add_argument('--image_size', default=[1, 112, 112, 3], help='the image size')
    return parser.parse_args(argv)

def main(args):

    data_path = Path('embedding_pkl')
    img_path = Path('img')
    os.makedirs(img_path/'{}'.format(str(args.name)), exist_ok = True)
    name_path = Path(img_path/str(args.name))

    cap = cv2.VideoCapture(0)

    cap.set(3,1280)
    cap.set(4,720)
    centerface = CenterFace()

    with tf.Session() as sess:
        load_model(args.model)
        inputs = tf.get_default_graph().get_tensor_by_name("input:0")
        embedding = tf.get_default_graph().get_tensor_by_name("embeddings:0")
		
        n = 0   
        total_embed = np.zeros((192, ))
        while cap.isOpened():   
            isSuccess,frame = cap.read()    
            img = frame.copy()  
            hf, wf = frame.shape[:2]
            
            if isSuccess:   
                cv2.putText(frame,
		                    'Press t to take a picture,q to quit.....',
		                    (10,100), 
		                    cv2.FONT_HERSHEY_SIMPLEX, 
		                    2,
		                    (0,255,0),
		                    3,
		                    cv2.LINE_AA)
			
            if cv2.waitKey(1)&0xFF == ord('t'):
                n += 1 
                try:            
                    dets, lms = centerface(frame, hf, wf, threshold=0.35)
                    for det in dets:
                        boxes, _ = det[:4], det[4]
                        cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
                    
                    for lm in lms:
                        for i in range(0, 5):
                            cv2.circle(frame, (int(lm[i * 2]), int(lm[i * 2 + 1])), 2, (0, 0, 255), -1)
		            
                    (x1, y1, x2, y2) = (int(i) for i in boxes)
                    (x, y, w, h) = (x1, y1, x2 - x1, y2 - y1)
                    
                    warped_face = img[y:y+h, x:x+w]
                    warped_face = cv2.resize(warped_face, [112, 112])
                    cv2.imwrite(str(name_path/('{}.jpg'.format(n))), warped_face)
                    
                    face = np.reshape(warped_face, args.image_size)

                    feed_dict = {inputs: face}
                    embed = sess.run(embedding, feed_dict=feed_dict)
					
                    embed = np.array(embed).reshape([embed.shape[1]])
                    
                    total_embed += embed 
                    print(total_embed.shape)
                    
                except:
                    print('no face captured')    
				
            if n >= 5:
                avr_embed = total_embed/n 
                print(avr_embed.shape)
                with open(str(data_path/('{}.pkl'.format(str(args.name)))), "wb") as f:
                    pickle.dump(avr_embed, f)
                break
            
            cv2.imshow("My Capture", frame)
            if cv2.waitKey(1)&0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
