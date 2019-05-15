import tensorflow as tf
import numpy as np
import random,csv
import time

global feature
global clabel

def test(test_file):
    feature=[]
    clabel=[]
    # 统计五分类过程五大类数据的具体分类情况
    wtf = np.zeros((5, 5))
    #加载测试集和标签
    file_path =test_file
    with (open(file_path,'r')) as data_from:
        csv_reader=csv.reader(data_from)
        for i in csv_reader:
            # print i
            if i[41]!='5':
                feature.append(i[:36])
                correct_label=i[41]
                clabel.append(correct_label)
            else:
                print()
            #print(feature)




    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('ckpt-5/')  # 通过检查点文件锁定最新的模型
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')  # 载入图结构，保存在.meta文件中
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('CNN Model Loading Success')
        else:
            print('No Checkpoint')
            #ii=0
        graph = tf.get_default_graph()
        xs = graph.get_tensor_by_name("inputs/pic_data:0")
        keep_prob = graph.get_tensor_by_name("inputs/keep_prob:0")
        logits = graph.get_tensor_by_name("prediction_eval:0")
        prediction = sess.run(logits, feed_dict={xs: feature,keep_prob: 1.0})  ##?
        print('Prediction Matrix of Test Data Set:')
        print(prediction)
        max_index = np.argmax(prediction, 1)
        print('Prediction Vector of Test Data Set:')
        print(max_index)
        m0 = max_index
        print('Size of Test Data Set:   ',m0.shape)
       

        for ii in range(len(feature)):
            
            k = int(clabel[ii])
            #print(m0[5])
            if m0[ii] in range(5):
                wtf[k][m0[ii]] += 1
    print('Type0:   Normal:',sum(wtf[0]),' ',wtf[0][0])
    print('         Accuracy:', wtf[0][0] / sum(wtf[0]))
    print('Type1:   Dos:',sum(wtf[1]),' ',wtf[1][1])
    print('         Accuracy:', wtf[1][1] / sum(wtf[1]))
    print('Type2:   Probing:',sum(wtf[2]),' ',wtf[2][2])
    print('         Accuracy:', wtf[2][2] / sum(wtf[2]))
    print('Type3:   R2L(Remote to Local):',sum(wtf[3]),' ',wtf[3][3])
    print('         Accuracy:', wtf[3][3] / sum(wtf[3]))
    print('Type4:   U2R(User to Root):',sum(wtf[4]),' ',wtf[4][4])
    print('         Accuracy:', wtf[4][4] / sum(wtf[4]))
    print('Confusion Matrix:')
    print(wtf)
    return wtf

if __name__ == '__main__':
    start_time=time.clock()
    
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('ckpt-5/')  # 通过检查点文件锁定最新的模型
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')  # 载入图结构，保存在.meta文件中
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success')
        else:
            print('No checkpoint')
    # 获得几乎所有的operations相关的tensor
    print('Get almost all operations related tensors before testing')
    ops = [o for o in sess.graph.get_operations()]
    for o in ops:
        print(o.name)
    test('corrected_handled-5-label-1.csv')


    end_time=time.clock()
    print("Running time:",(end_time-start_time))









