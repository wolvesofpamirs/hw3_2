import pickle
import sys,os
import numpy as np
import copy

#im = Image.open("bride.jpg")
#im.rotate(45).show()
#im = Image.fromarray(np.uint8(cm.gist_earth(myarray)*255))
height = 32
width = 32
nb_channel = 3
def generate_altered_images(folder):
    #import matplotlib.cm as cm
    from PIL import Image
    import PIL
    for i in range(1,2):
        filename = 'data_batch_'+str(i)
        with open(os.path.join(folder,filename),'rb') as fr:
            save = pickle.load(fr,encoding='iso-8859-1')
        #label_data = np.array(label_data,dtype='uint8')
        saves = []
        X = save['data']
        #X = np.divide(save['data'], 255.)
        X.shape = (len(X), nb_channel, height, width)
        print(X.shape)
        X = X.transpose(0,2,3,1)  
        print(X.shape)
        
        for j in range(5):
            saves.append(copy.deepcopy(save))
        for j in range(len(X)):
            im = Image.fromarray(np.uint8(X[j]))
            #saves[0][j] = im.rotate(15)
            #saves[1][j] = im.rotate(-15)
            #saves[2][j] = im.rotate(30)
            #saves[3][j] = im.rotate(-30)
            #d = (np.array(im.transpose(PIL.Image.FLIP_LEFT_RIGHT).getdata()))
            #d.shape = (height, width, 3)
            #print(d.shape)
            im2 = im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            saves[0]['data'][j] = (np.array(im2.getdata())).reshape((height, width, 3)).transpose(2, 0, 1)  
            im3 = im.transpose(PIL.Image.FLIP_TOP_BOTTOM)
            saves[1]['data'][j] = (np.array(im3.getdata())).reshape((height, width, 3)).transpose(2, 0, 1)   
            im4 = im.transpose(PIL.Image.TRANSPOSE)
            saves[2]['data'][j] = (np.array(im4.getdata())).reshape((height, width, 3)).transpose(2, 0, 1)   
            im5 = im.transpose(PIL.Image.ROTATE_90)
            saves[3]['data'][j] = (np.array(im5.getdata())).reshape((height, width, 3)).transpose(2, 0, 1)   
            im6 = im.transpose(PIL.Image.ROTATE_270)
            saves[4]['data'][j] = (np.array(im6.getdata())).reshape((height, width, 3)).transpose(2, 0, 1)  
        #s_data = save['data']
        for j in range(5):
            filename = 'data_batch_'+str(i)+str(j)
            with open(os.path.join(folder,filename), 'wb') as handle:
                pickle.dump(saves[j], handle, protocol=pickle.HIGHEST_PROTOCOL)
        del save
        

def load_sup(folder):
    from keras.utils import to_categorical
    with open(os.path.join(folder,'data_batch_1'),'rb') as fr:
        save = pickle.load(fr,encoding='iso-8859-1')
    #label_data = np.array(label_data,dtype='uint8')
    s_data = save['data']
    s_labels = np.array(save['labels'])
    s_f_names = np.array(save['filenames'])
    # 删除列表
    
    #print(s_data[0])
    height = 32
    width = 32
    nb_channel = 3

    #print(len(s_data))
    #s_data.shape = (len(s_data),3,32,32)
    print(s_data[0])
    
    # preproc
    
    nb_class = 10
    nb_pic_in_class = 500
    pic_count = np.zeros(nb_class)
    condition = True
    numb_class_finished = 0
    
            
    
    X = np.array(save['data'])
    X = np.divide(X, 255.)
    X.shape = (len(X), nb_channel, height, width)
    #X = X.transpose(0, 2, 3, 1)
    Y = np.array(save['labels'])
    del save
    
    Y = to_categorical(Y, nb_class)
    '''
    for j in range(3):
        with open(os.path.join(folder,'data_batch_1'+(str)(j)),'rb') as fr:
            save = pickle.load(fr,encoding='iso-8859-1')
        #label_data = np.array(label_data,dtype='uint8')
        s_data = save['data']
        s_labels = np.array(save['labels'])
        s_f_names = np.array(save['filenames'])
        X_ = np.array(np.array(save['data']))
        X_ = np.divide(X_, 255.)
        X = np.r_[X, X_]
        
        X.shape = (len(X), nb_channel, height, width)
        Y_ = np.array(save['labels'])
        del save
        
        Y_ = to_categorical(Y_, nb_class)
        Y = np.r_[Y, Y_]
    '''
    return X, Y
def load(folder):
    with open(os.path.join(folder,'data_batch_1'),'rb') as fr:
        save = pickle.load(fr,encoding='iso-8859-1')
    #label_data = np.array(label_data,dtype='uint8')
    s_data = save['data']
    s_labels = np.array(save['labels'])
    s_f_names = np.array(save['filenames'])
    # 删除列表
    del save
    #print(s_data[0])
    height = 32
    width = 32
    nb_channel = 3
    print(min(s_labels))
    print('data set', s_data.shape, s_labels.shape)
    print('data', s_data[0].shape, s_labels[0].shape)
    #print(len(s_data))
    #s_data.shape = (len(s_data),3,32,32)
    print(s_data[0])
    
    # preproc
    
    X = []
    Y = []
    XU= []
    nb_class = 10
    nb_pic_in_class = 500
    pic_count = np.zeros(nb_class)
    condition = True
    numb_class_finished = 0
    i = 0
    for i in range(len(s_data)):
        if pic_count[s_labels[i]] < nb_pic_in_class:
            X.append(s_data[i])
            Y.append(s_labels[i])
            pic_count[s_labels[i]] += 1
            if pic_count[s_labels[i]] == nb_pic_in_class:
                numb_class_finished += 1
        else:
            XU.append(s_data[i])
    
            
    
    X = np.array(X)
    X = np.divide(X, 255.)
    X.shape = (len(X), nb_channel, height, width)
    Y = np.array(Y)
    from keras.utils import to_categorical
    Y = to_categorical(Y, nb_class)
    
    XU= np.array(XU)
    for i in range(2,6):
        filename = 'data_batch_'+str(i)
        with open(os.path.join(folder,filename),'rb') as fr:
            save = pickle.load(fr,encoding='iso-8859-1')
        #label_data = np.array(label_data,dtype='uint8')
        s_data = save['data']
        del save
        #s_data.shape = (len(s_data),nb_channel,height,width)
        XU = np.concatenate([XU, s_data])
    #print(XU.shape)
    #XU= np.array(XU)
    XU = np.divide(XU, 255.)
    XU.shape = (len(XU), nb_channel, height, width)
    print(X.shape)
    
    print(Y.shape)
    print(XU.shape)
    #X = X.transpose(0, 2, 3, 1)
    #XU = XU.transpose(0, 2, 3, 1)
    return (X,Y, XU)
    
    #return s_data, s_labels, s_f_names
        #numpy提供了numpy.append(arr, values, axis=None)函数。对于参数规定，
    # 要么一个数组和一个数值；要么两个数组，不能三个及以上数组直接append拼接。append函数返回的始终是一个一维数组。
    '''
    data = np.append(data, s_data, axis=0) if data is not None else s_data
    labels = np.append(labels, s_labels, axis=0) if labels is not None else s_labels
    f_names = np.append(f_names, s_f_names, axis=0) if f_names is not None else s_f_names
    '''
    
    '''
    # preproc
    nb_class = 10
    nb_pic_in_class = 500
    height = 32
    width = 32
    nb_channel = 3
    X = label_data
    X.shape = (nb_class*nb_pic_in_class,nb_channel,height,width)
    from keras.utils import to_categorical
    Y = to_categorical([i for i in range(nb_class) for j in range(nb_pic_in_class)])
    return (X,Y)
    '''