'''
Note: I assume that you already know how to build tensorflow models and train some simple networks.
Here I only give the multi-gpu part so that you can transfer it to any of your network by some small
modicifations.
There are two samples. They are similar so you can learn multi-gpu better.
'''

def average_gradients(grads):#grads:[[grad0, grad1,..], [grad0,grad1,..]..]
  averaged_grads = []
  for grads_per_var in zip(*grads):
    grads = []
    for grad in grads_per_var:
      expanded_grad = tf.expanded_dim(grad,0)
      grads.append(expanded_grad)
    grads = tf.concat_v2(grads, 0)
    grads = tf.reduce_mean(grads, 0)
    averaged_grads.append(grads)
  return averaged_grads

#sample 1. This sample can bu used based on single-gpu.py in this repo.
def multi_gpu_model(num_gpus=1):
  grads = []
  for i in range(num_gpus):
    with tf.device("/gpu:%d"%i):
      with tf.name_scope("tower_%d"%i):
        model = Model(is_training, config, scope)
        tf.add_to_collection("train_model", model)
        grads.append(model.grad) 
        tf.add_to_collection("loss",model.loss)
        tf.add_to_collection("predict", model.predict)
        tf.add_to_collection("merge_summary", model.merge_summary)
  with tf.device("cpu:0"):
    averaged_gradients = average_gradients(grads)
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    train_op=opt.apply_gradients(zip(average_gradients,tf.trainable_variables()))
  return train_op
def generate_feed_dic(model, feed_dict, batch_generator):
  x, y = batch_generator.next_batch()
  feed_dict[model.x] = x
  feed_dict[model.y] = y
def run_epoch(session, data_set, scope, train_op=None, is_training=True):
  batch_generator = BatchGenerator(data_set, batch_size)
  ...
  ...
  if is_training and train_op is not None:
    models = tf.get_collection("train_model")
    feed_dic = {}
    for model in models:
      generate_feed_dic(model, feed_dic, batch_generator)
    losses = tf.get_collection("loss", scope)
    ...
    ...
#main process
data_process()
with tf.name_scope("train") as train_scope:
  train_op = multi_gpu_model(..)
with tf.name_scope("test") as test_scope:
  model = Model(...)
saver = tf.train.Saver()
with tf.Session() as sess:
  writer = tf.summary.FileWriter(...)
  ...
  run_epoch(...,train_scope)
  run_epoch(...,test_scope)


#sample 2
def train_multi_gpu(self, flag, train_queue=None, val_queue=None):
    save_cfg(os.path.join(self.sample_folder, 'cfg.yaml'))
    with tf.device('/cpu:0'):
      tower_grads = []
     with tf.variable_scope(tf.get_variable_scope()):
        for gpu_id in range(cfg.CONST.NUM_GPUS):
	        with tf.device('/gpu:%d' % gpu_id):
	          print('tower:%d...'% gpu_id)
	          with tf.name_scope('tower_%d' % gpu_id) as scope:
              model = NetClass(cfg)#initiate net
              model.build_model()#build model 
              if(gpu_id==0):
                learning_rate=model.learning_rate#we only set lr for one time
              gen=model.loss
              grads = tf.train.AdamOptimizer(learning_rate=learning_rate).compute_gradients(loss)
              tower_grads.append(grads)
              tf.get_variable_scope().reuse_variables()#we reuse variableds here after we initiate one model 
              tf.add_to_collection("train_model", model)#add to collection
      print('build model on gpu tower done.')
      opt= tf.train.AdamOptimizer(learning_rate=learning_rate)
      apply_gradient_op = opt.apply_gradients(average_gradients(tower_grads))
      print('reduce model on cpu done.')
      self.sess.run(tf.global_variables_initializer())
      self.sess.run(tf.local_variables_initializer())
      for model in tf.get_collection("train_model"):
        self.init_net(model)# Do some initiation. Note we should only initiate variables for one time because they are shared
      self.writer = tf.summary.FileWriter(self.log_folder, self.sess.graph)
      for it in range(self.start_iter, self.max_iter):
	      iteration_start_time = time.time()
        models=tf.get_collection("train_model")
        feed_dict={}
        for model in models:
          image, label= train_queue.get()
          model.set_feed(image, label, it)
          is_training=True
          feed_dict[model.img_in]=image
          feed_dict[model.gt_label]=label
          feed_dict[model.is_training]=is_training
        self.net=models[0]# use the first model to print loss and produce validation resutls
        _ = self.sess.run(apply_gradient_op, feed_dict)
        self.net.print_loss(self.sess)

