import tensorflow as tf

'''
Shape error
'''
# 下面程序会报出错误：Dimensions must be equal, but are 2 and 4 for 'add' (op: 'Add') with input shapes: [4,2], [4].
def shape_error_01():
    def some_method(data):
        a = data[:, 0:2]
        print(a.get_shape())
        c = data[:, 1]
        print(c.get_shape())
        s = (a + c)
        return tf.sqrt(tf.matmul(s, tf.transpose(s)))

    with tf.Session() as sess:
        fake_data = tf.constant([
            [5.0, 3.0, 7.1],
            [2.3, 4.1, 4.8],
            [2.8, 4.2, 5.6],
            [2.9, 8.3, 7.3]
        ])
        print(sess.run(some_method(fake_data)))
        pass
    pass

def shape_error_01_repair():
    # 下面程序通过修改shape，修复了上面的bug
    def some_method(data):
      a = data[:,0:2]
      print(a.get_shape())
      c = data[:,1:3]
      print(c.get_shape())
      s = (a + c)
      return tf.sqrt(tf.matmul(s, tf.transpose(s)))

    with tf.Session() as sess:
      fake_data = tf.constant([
          [5.0, 3.0, 7.1],
          [2.3, 4.1, 4.8],
          [2.8, 4.2, 5.6],
          [2.9, 8.3, 7.3]
        ])
      print(sess.run(some_method(fake_data)))

# 几种修复shape error的方式
def shape_error_02_repair():
    x = tf.constant([[3, 2], [4, 5], [6, 7]])
    print("x.shape", x.shape)
    expanded = tf.expand_dims(x, 1)
    print( "expanded.shape", expanded.shape)
    sliced = tf.slice(x, [0, 1], [2, 1])
    print("sliced.shape", sliced.shape)

    with tf.Session() as sess:
        print("expanded: ", expanded.eval())
        print("sliced: ", sliced.eval())


'''
TensorFlow debugger
'''
def study_tenforflow_debugger():
    import tensorflow as tf
    from tensorflow.python import debug as tf_debug

    def some_method(a, b):
        b = tf.cast(b, tf.float32)
        s = (a / b)
        s2 = tf.matmul(s, tf.transpose(s))
        return tf.sqrt(s2)

    with tf.Session() as sess:
        fake_a = [[5.0, 3.0, 7.1], [2.3, 4.1, 4.8]]
        fake_b = [[2, 0, 5], [2, 8, 7]]
        a = tf.placeholder(tf.float32, shape=[2, 3])
        b = tf.placeholder(tf.int32, shape=[2, 3])
        k = some_method(a, b)

        # Note: won't work without the ui_type="readline" argument because
        # Datalab is not an interactive terminal and doesn't support the default "curses" ui_type.
        # If you are running this a standalone program, omit the ui_type parameter and add --debug
        # when invoking the TensorFlow program
        #      --debug (e.g: python debugger.py --debug )
        sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="readline")
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        print(sess.run(k, feed_dict={a: fake_a, b: fake_b}))


'''
tf.Print()
'''
def study_tensorflow_Print():
    def some_method(a, b):
        b = tf.cast(b, tf.float32)
        s = (a / b)
        print_ab = tf.Print(s, [a, b])
        s = tf.where(tf.is_nan(s), print_ab, s)
        return tf.sqrt(tf.matmul(s, tf.transpose(s)))

    with tf.Session() as sess:
        fake_a = tf.constant([
            [5.0, 3.0, 7.1],
            [2.3, 4.1, 4.8],
        ])
        fake_b = tf.constant([
            [2, 0, 5],
            [2, 8, 7]
        ])

        print(sess.run(some_method(fake_a, fake_b)))
        pass
    pass





if __name__ == '__main__':



    pass

