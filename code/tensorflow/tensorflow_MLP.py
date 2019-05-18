# -*- coding: utf-8 -*-
# @Time    : 2018/7/19/019 15:21
# @Author  : ZhengHao
# @File    : tensorflow_MLP.py
# @Software: PyCharm

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt


def my_relu_def(x):

    # return -0.555931 * x * x * x - 0.035258*x*x + 1.053903 * x+0.008155
    # return 0.817215*x*x+0.020562*x+0.271116
    return 2/(x+1)

def my_relu_grad_def(x, threshold=0.05):

    # return -1.667793*x*x - 0.70516*x +1.053903
    # return 1.63443*x+0.020562
    return -2/((x+1)*(x+1))
# making a common function into a numpy function
my_relu_np = np.vectorize(my_relu_def)
my_relu_grad_np = np.vectorize(my_relu_grad_def)
# numpy uses float64 but tensorflow uses float32
my_relu_np_32 = lambda x: my_relu_np(x).astype(np.float32)
my_relu_grad_np_32 = lambda x: my_relu_grad_np(x).astype(np.float32)



def my_relu_grad_tf(x, name=None):
    with ops.name_scope(name, "my_relu_grad_tf", [x]) as name:
        y = tf.py_func(my_relu_grad_np_32,
                       [x],
                       [tf.float32],
                       name=name,
                       stateful=False)
        return y[0]

def my_py_func(func, inp, Tout, stateful=False, name=None, my_grad_func=None):
    # Need to generate a unique name to avoid duplicates:
    random_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(random_name)(my_grad_func)  # see _my_relu_grad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": random_name, "PyFuncStateless": random_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

# The grad function we need to pass to the above my_py_func function takes a special form:
# It needs to take in (an operation, the previous gradients before the operation)
# and propagate(i.e., return) the gradients backward after the operation.
def _my_relu_grad(op, pre_grad):
    x = op.inputs[0]
    cur_grad = my_relu_grad_tf(x)
    next_grad = pre_grad * cur_grad
    return next_grad

def my_relu_tf(x, name=None):
    with ops.name_scope(name, "my_relu_tf", [x]) as name:
        y = my_py_func(my_relu_np_32,
                       [x],
                       [tf.float32],
                       stateful=False,
                       name=name,
                       my_grad_func=_my_relu_grad)  # <-- here's the call to the gradient
        return y[0]

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf



mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

#初始化设置参数,对于w1使用截断的正态分布，并且为了防止完全对称的0梯度，需要加上一点噪声
in_units = 784
h1_units = 300
w1 = tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
w2 = tf.Variable(tf.zeros([h1_units,10]))
b2 = tf.Variable(tf.zeros([10]))

#定义输入的placeholder,包括输入的数据和dropout的比率
x = tf.placeholder("float",[None,in_units])
keep_prob = tf.placeholder(tf.float32)

#定义模型结构
hidden1 =tf.nn.relu(tf.matmul(x,w1)+b1)  #你可以在这里修改想要的激活函数，nn.relu是tensorflow自带函数，如果使用my_relu_tf函数，你可以在上面这个对应的函数改成你想要的函数
# hidden1 = my_relu_tf(tf.matmul(x,w1)+b1)
# hidden1 = -0.8555*tf.pow(tf.matmul(x,w1)+b1,3)+1.7751*(tf.matmul(x,w1)+b1)
hidden1_drop = tf.nn.dropout(hidden1,keep_prob)
# y = tf.nn.softmax(tf.matmul(hidden1_drop,w2)+b2)
y = tf.matmul(hidden1_drop,w2)+b2

#定义loss函数和优化器
y_ = tf.placeholder(tf.float32,[None,10])
# cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
# cross_entropy = -tf.reduce_mean(y_*tf.log(y))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#训练步骤
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    step_total = []
    loss_total =[]
    accuracy_total = []
    while step * 10 < 100000:
        batch_xs,batch_ys = mnist.train.next_batch(100)
        # batch_xs = 1.003269*np.exp(-3.20423*batch_xs)-0.504762
        sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys,keep_prob:0.75})
        if step%10 == 0:
            loss = sess.run(cross_entropy,feed_dict={x:batch_xs,y_:batch_ys,keep_prob:0.75})
            acc = sess.run(accuracy,feed_dict={x:batch_xs,y_:batch_ys,keep_prob:0.75})
            accuracy_total.append(acc)
            loss_total.append(loss)
            step_total.append(step)
            print("loss = "+"{:.4f}".format(loss)+"accuracy = "+"{:.4f}".format(acc))
        step+=1

    data = mnist.test.images
    # data = 1.003269*np.exp(-3.20423*data)-0.504762
    label = mnist.test.labels
    correct_prediction_1 = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy_1 = tf.reduce_mean(tf.cast(correct_prediction_1, tf.float32))
    print(accuracy_1.eval({x: data, y_:label, keep_prob: 1.0}))

#
# import pandas as pd
#
# data ={}
# data['loss'] = loss_total
# data['accuracy'] = accuracy_total
# output = pd.DataFrame(data,index=step_total)
# output.to_csv("C:/Users/Administrator/Desktop/data3.csv")
#
# fig = plt.figure()
# ax1 = fig.add_subplot(211)
# ax1.plot(step_total,loss_total,label="Loss",color='b')
# ax1.set_xlabel("step")
# ax1.set_ylabel("Loss")
# plt.legend(loc=1)
# ax2 = fig.add_subplot(212)
# ax2.plot(step_total,accuracy_total,label="Accurcacy",color='r')
# ax2.set_xlabel("step")
# ax2.set_ylabel("Accuracy")
# plt.legend(loc=4)
# plt.show()

# 模型评测
