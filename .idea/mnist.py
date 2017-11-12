import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
# 批次大小
batch_size = 100
# 批次数量
n_batch = mnist.train.num_examples // batch_size
# 运行次数
max_steps = 1001
# 目录
DIR = "C:/Users/Shinelon/PycharmProjects/mnist/.idea/"
# 图片张数
image_num = 3000

# 定义会话
sess = tf.Session()

# 载入图片
embedding = tf.Variable(tf.stack(mnist.test.images[:image_num]), trainable=False, name='embedding')

# 参数统计
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

with tf.name_scope('input'):
    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 784], name="x_put")
    y = tf.placeholder(tf.float32, [None, 10], name="y_put")
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)
keep_prob = tf.placeholder(tf.float32)
lr = tf.Variable(0.001, dtype=tf.float32)

with tf.name_scope('layer'):
    #创建一个神经网络
    with tf.name_scope('layer-1'):
        with tf.name_scope('weights_1'):
            W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
            variable_summaries(W1)
        with tf.name_scope('biases_1'):
            b1 = tf.Variable(tf.zeros([500]) + 0.1)
            variable_summaries(b1)
        with tf.name_scope('wx_plus_b_1'):
            L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
        with tf.name_scope('drop1'):
            L1_drop = tf.nn.dropout(L1, keep_prob)

    with tf.name_scope('layer-2'):
        with tf.name_scope('weights_2'):
            W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1))
            variable_summaries(W2)
        with tf.name_scope('biases_2'):
            b2 = tf.Variable(tf.zeros([300]) + 0.1)
            variable_summaries(b2)
        with tf.name_scope('wx_plus_b_2'):
            L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
        with tf.name_scope('drop2'):
            L2_drop = tf.nn.dropout(L2, keep_prob)

    with tf.name_scope('layer-3'):
        with tf.name_scope('weights_3'):
            W3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1))
            variable_summaries(W3)
        with tf.name_scope('biases_3'):
            b3 = tf.Variable(tf.zeros([10]) + 0.1)
            variable_summaries(b3)
        with tf.name_scope('prediction'):
            prediction = tf.nn.softmax(tf.matmul(L2_drop, W3) + b3)

with tf.name_scope('loss'):
    # 二次代价函数
    # loss = tf.reduce_mean(tf.square(y - prediction))
    # 交叉熵代价函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    tf.summary.scalar('loss', loss)
# 梯度下降法
#train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
# AdamOptimizer优化
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)


# 初始化
init = tf.global_variables_initializer()
sess.run(init)

# 结果存放在一个布尔型变量中
with tf.name_scope('Accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

# 产生metadata文件
# if tf.gfile.Exists(DIR + 'projector/projector/metadata.tsv'):
#     tf.gfile.DeleteRecursively(DIR + 'projector/projector/metadata.tsv')
with open(DIR + 'projector/projector/metadata.tsv', 'w') as f:
    labels = sess.run(tf.argmax(mnist.test.labels[:], 1))
    for i in range(image_num):
        f.write(str(labels[i]) + '\n')

# 合并所有的summary
merged = tf.summary.merge_all()

projector_writer = tf.summary.FileWriter(DIR + 'projector/projector', sess.graph)
saver = tf.train.Saver()
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = embedding.name
embed.metadata_path = DIR + 'projector/projector/metadata.tsv'
embed.sprite.image_path = DIR + 'projector/data/mnist_10k_sprite.png'
embed.sprite.single_image_dim.extend([28, 28])
projector.visualize_embeddings(projector_writer, config)

# with tf.Session() as sess:
#     sess.run(init)
#     writer = tf.summary.FileWriter('logs/', sess.graph)
#     for epoch in range(20):
#         sess.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))
#         for i in range(n_batch):
#             batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#             summary, _ = sess.run([merged, train_step],feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
#             writer.add_summary(summary, i)
#
#         learning_rate = sess.run(lr)
#         test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
#         print("Iter " + str(epoch) + " Testing Accuracy " + str(test_acc) + " Learning Rate " + str(learning_rate))


for i in range(max_steps):
    # 每个批次100
    batch_xs, batch_ys = mnist.train.next_batch(100)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0}, options=run_options, run_metadata=run_metadata)
    projector_writer.add_run_metadata(run_metadata, 'step%03d' % i)
    projector_writer.add_summary(summary, i)
    if i % 100 ==0:
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        print("Iter " + str(i) + " Testing Accuracy= " + str(test_acc))

saver.save(sess, DIR + 'projector/projector/a_model.ckpt', global_step=max_steps)
projector_writer.close()
sess.close()









