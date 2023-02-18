import pickle
import cv2
import numpy as np
import tensorflow as tf
import os 
import re 
import sys
import argparse 
import glob
import sklearn.preprocessing
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
    parser.add_argument('--emb_path', default='/home/hau/Desktop/FaceRecognition/MobileFaceNet_TF/embeddings/', help='the embeddings path')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--threshold', default=0.8, help='the threshold')
    return parser.parse_args(argv)


def main(args):
emb_path = args.emb_path
    all_embed = glob.glob(emb_path + '*.pkl')
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)
    centerface = CenterFace()
    
    with tf.Session() as sess:  
        load_model(args.model)
        inputs = tf.get_default_graph().get_tensor_by_name("input:0")
        embedding = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        
        while cap.isOpened():
            success, frame = cap.read()
            hf, wf = frame.shape[:2]

            if success:   
                try:
                    dets, _ = centerface(frame, hf, wf, threshold=0.35)
                    input_images = np.zeros((dets.shape[0], 112, 112, 3))
                except:     
                    cv2.putText(frame, 'No face detected', (400,50), cv2.FONT_HERSHEY_SIMPLEX, 
                                        1, (0,255,0), 1, cv2.LINE_AA) 
				
                faces = []
                for i, box in enumerate(dets):
                    (x1, y1, x2, y2) = (int(box[i]) for i in range(4))
                    x, y, w, h = x1, y1, x2 - x1, y2 - y1
                    faces.append([x, y, w, h])
					
                    face = frame[y:y+h, x:x+w] 
                    face = (face - 127.5)*0.0078125
                    face = cv2.resize(face, args.image_size)   
                    input_images[i, :] = face 

                feed_dict = {inputs: input_images}
                emb_array = sess.run(embedding, feed_dict=feed_dict)
                emb_array = sklearn.preprocessing.normalize(emb_array)
                print(emb_array.shape)
				
		maximum = 0 
                for i, em in enumerate(emb_array):
                    em = em.flatten()
                    for emb in all_embed:
                        with open(emb, 'rb') as f:
                            embed = pickle.load(f)

                        name = emb[len(emb_path):]
                        name = name.split('.')[0]
                        sim  = np.dot(em, embed.T)
						
			if sim > maximum:
                            maximum = sim 
                            p = name
						
		    if maximum > args.threshold:
			person = p
		    else: 
			person = 'unknown'

		    x, y, w, h = faces[i]
		    cv2.rectangle(frame, (x, y), (x+w, y+h), (2, 255, 0), 2)
		    cv2.putText(frame, str(person), (int(x+w/50), int(y+h+20)), cv2.FONT_HERSHEY_SIMPLEX, 
						0.5, (0, 255, 0), 1, cv2.LINE_AA)	
		
            cv2.imshow("My Capture", frame)
	    if cv2.waitKey(1)&0xFF == ord('q'):
		break
        
    cap.release()
    cv2.destroyAllWindows() 


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
