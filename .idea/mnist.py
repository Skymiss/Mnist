import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
n_inputs = 28   # 输入一行，一行共有28个数据
max_time = 28   # 一共28行
lstm_size = 100
n_class = 10
batch_size = 50
n_batch = mnist.train.num_examples // batch_size

# # 参数统计
# def variable_summaries(var):
#     with tf.name_scope('summaries'):
#         mean = tf.reduce_mean(var)
#         tf.summary.scalar('mean', mean)
#         with tf.name_scope('stddev'):
#             stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#         tf.summary.scalar('stddev', stddev)
#         tf.summary.scalar('max', tf.reduce_max(var))
#         tf.summary.scalar('min', tf.reduce_min(var))
#         tf.summary.histogram('histogram', var)

with tf.name_scope('input'):
    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 784], name="x_put")
    y = tf.placeholder(tf.float32, [None, 10], name="y_put")

weights = tf.Variable(tf.truncated_normal([lstm_size, n_class], stddev=0.1))
biases = tf.Variable(tf.constant(0.1, shape=[n_class]))

def RNN(X, weights, biases):
    inputs = tf.reshape(X, [-1, max_time, n_inputs])
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1], weights) + biases)
    return results

prediction = RNN(x, weights, biases)

with tf.name_scope('loss'):
    # 二次代价函数
    # loss = tf.reduce_mean(tf.square(y - prediction))
    # 交叉熵代价函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    #tf.summary.scalar('loss', loss)
# 梯度下降法
#train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
# AdamOptimizer优化
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)


# 初始化
init = tf.global_variables_initializer()

# 结果存放在一个布尔型变量中
with tf.name_scope('Accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #tf.summary.scalar('accuracy', accuracy)



with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x: batch_xs, y: batch_ys})
            #writer.add_summary(summary, i)

        #learning_rate = sess.run(lr)
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter " + str(epoch) + " Testing Accuracy " + str(test_acc))
