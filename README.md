# AI challenger比赛—农作物病害检测

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=JinbaoSite/plants_disease_detection&type=Date)](https://star-history.com/#JinbaoSite/plants_disease_detection&Date)

## 1 赛题简介

对近5万张按“物种-病害-程度”分成61类的植物叶片照片进行分类

比赛地址：[AI challenger比赛—农作物病害检测](https://challenger.ai/competition/pdr2018)

## 2 框架

我使用的是Keras，以TensorFlow为后端，手动实现了DenseNet用于图片分类
由于Kaggle现在可以免费使用GPU，所以采用将数据上传至Kaggle的私人Dataset上，在其上创建Kernel进行模型训练
（上传需要翻墙，有梯子最好）

## 3 DenseNet模型实现

```
def dense_block(x, blocks, name):
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x
def transition_block(x, reduction, name):
    bn_axis = 3
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      name=name + '_conv')(x)
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x
def conv_block(x, growth_rate, name):
    bn_axis = 3
    x1 = layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv2D(4 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x
def DenseNet(blocks, input_shape=(150,150,3), classes=61):

    img_input = Input(shape=input_shape)

    bn_axis = 3

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
    x = layers.Activation('relu', name='conv1/relu')(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')

    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='bn')(x)

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation='softmax', name='fc61')(x)

    inputs = img_input

    model = Model(inputs, x, name='densenet')
    
    return model
```
调用DenseNet函数即可创建
```
model = DenseNet(blocks=[6, 12, 48, 32], input_shape=(150,150,3),classes=61)
model.summary()
```

## 4 数据准备

1、训练集、验证集生产器
这里对图片进行图像预处理，增加图片归一化、适度旋转、随机缩放、上下翻转
```
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1. / 255)
```
2、读取数据
从目录中读取数据
```
img_width, img_height = 150, 150
train_data_dir = '../input/train/train'
validation_data_dir = '../input/val/val'
batch_size = 64
classes = 61

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical') #多分类

validation_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical') #多分类
```

## 5 模型训练

1、先对模型进行预编译
```
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001),
              metrics=['accuracy'])
```
2、训练模型
增加自动更新学习率和保存在验证集最后的模型参数
```
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1,factor=0.5, min_lr=0.000001)
checkpoint = ModelCheckpoint(model_name, monitor='val_acc', save_best_only=True)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[checkpoint, learning_rate_reduction])
```
训练次数由于受Kaggle中Kernel的使用时间受限，只能训练6小时，所以只能暂时训练30，不过可以多次迭代训练。

## 6 模型预测

由于文件夹存放顺序跟window上不一样，所以实际上文件夹在Kaggle上Dataset上的存放顺序如下
```
rr = [0,
      1,10,11,12,13,14,15,16,17,18,19,
      2,20,21,22,23,24,25,26,27,28,29,
      3,30,31,32,33,34,35,36,37,38,39,
      4,40,41,42,43,44,45,46,47,48,49,
      5,50,51,52,53,54,55,56,57,58,59,
      6,60,
      7,
      8,
      9]

images = os.listdir('../input/ai-challenger-pdr2018/testa/testA')

result = []
for img1 in images:
    image_path = '../input/ai-challenger-pdr2018/testa/testA/' + img1
    img = image.load_img(image_path, target_size=(150, 150))
    x = image.img_to_array(img)/255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    tmp = dict()
    tmp['image_id'] = img1
    tmp['disease_class']=rr[int(np.argmax(preds))]
    result.append(tmp)
```
最后保存为json
```
import json
json2 = json.dumps(result)
f = open('result.json','w',encoding='utf-8')
f.write(json2)
f.close()
```

## 7 提交结果

最终的结果是0.87395的成绩
![plants_disease_detection](https://img-blog.csdnimg.cn/2018121615174919.PNG)


## 8 完整代码参考

[DenseNet模型训练 plants_disease_detection](https://github.com/JinbaoSite/plants_disease_detection/blob/master/plants_disease_detection-jinbaosite.ipynb)

如果你觉得我写的不错，请给我一下Star(`^_^`)，谢谢！
