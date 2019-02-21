import tensorflow as tf
def model(numYs, k, l, s, thetas, alphas, isdiscrete, user_a, penalty):
  
  # user_a = tf.reshape(user_a, [len(k)])
  is_discrete = tf.convert_to_tensor(isdiscrete, dtype = tf.float64)

  user_alphas = tf.convert_to_tensor(user_a, dtype=tf.float64)
  numYs = 2

  active = is_discrete + (1-is_discrete) * tf.cast(tf.greater_equal(s, user_alphas-0.0001), dtype=tf.float64)
  active = tf.stop_gradient(active)
  # scale the s_ values to be between 0 and 1.
  s_ = tf.clip_by_value((s - user_alphas)/(1-user_alphas+0.00001), 0, 1) # tf.stop_gradient(s * active)
  l = tf.clip_by_value(l * active, -1, 1)

  numbins = tf.constant(10, dtype=tf.float64)

  # numbins = 10
  sbin_widths = (1-tf.zeros_like(user_alphas))/numbins
  #sbin_widths = np.ones_like(user_alphas)/numbins
  
  sbins = tf.einsum('i,j->ij', sbin_widths, tf.cast(tf.range(0,numbins+1), dtype=tf.float64)) # +tf.expand_dims(user_alphas,1)
  sbins = tf.transpose(sbins)

#             s_smooth = tf.round((s_-user_alphas)*numbins/(1-user_alphas))*sbin_widths + user_alphas
#             s_ = (1-is_discrete) * s_smooth + is_discrete * s_

  ys = tf.constant([-1,1], dtype=tf.float64) if numYs == 2 else tf.range(0,numYs)

#adding new
  def cont_pots_binary(thetas, s, y, l):
      if(penalty%10 == 1):
          return (thetas*s-alphas)*y*l
      elif(penalty%10 == 2):
          return y*l*thetas*tf.clip_by_value(s-alphas,0,1)
      elif(penalty%10 == 3):
          return y*l*thetas*s
      elif(penalty%10 == 4):
          return y*l*thetas*tf.sigmoid(s-alphas)
      elif(penalty%10 == 5):
        # if labels(l,y) agree use a potential of theta*s
        # else if they disagree and l is non-zero use alphas*(1-s)
        # zero in all other cases.
        # the expression here might be incorrect, make sure it implements
        # as per the above intention.
          return thetas*s*tf.cast(tf.equal(y,l),dtype=tf.float64) + \
                  alphas*(-s)*l*tf.cast(tf.not_equal(y,l), dtype=tf.float64)
#                 elif (penalty%10 == 6):
#                   # Gaussian distribution with a variance of 1.
#                   pos = tf.cast(tf.equal(y,l),dtype=tf.float64)
#                   return -(s-thetas)*(s-thetas)*pos/2 - (1-pos)*l*(s-alphas)*(s-alphas)/2
      return 0

  def equal_sign(y, l):
    eq = tf.cast(tf.equal(y,l), dtype=tf.float64)
    return eq + (1-eq)*-1

  def cont_pot(s, y, l):
    if numYs == 2:
      return cont_pots_binary(thetas, s, y, l)
    return cont_pots_binary(thetas[y], s, equal_sign(y, k), l)

  def dis_pot(y,l):
      return y*thetas*l if numYs == 2 else thetas[y]*l*equal_sign(y,k) 

  def pot(s, y, l):
      return (1-is_discrete) * cont_pot(s, y,l) + is_discrete * dis_pot(y,l)

  def msg(s, y, l):
      return (1 +  (1-is_discrete) * tf.reduce_sum(tf.exp(pot(s, y,l)), axis=0)*sbin_widths 
                +  (is_discrete) *  tf.exp(dis_pot(y, l)))
    
  def msgActive(s, y, l):
    return is_discrete*(msg(s,y,l)-1) + (1-is_discrete)* tf.reduce_sum(tf.exp(pot(s, y,l))*tf.cast(tf.greater(s,alphas),tf.float64), axis=0)*sbin_widths

  z_y = tf.map_fn(lambda y: tf.reduce_prod(msg(sbins, y,k)), ys)
  Z_ = tf.reduce_sum(z_y)

  log_pt = tf.map_fn(lambda y: tf.reduce_sum(pot(s_, y,l), axis=1), ys)
  pt_1 = tf.reduce_logsumexp(log_pt, axis=1) - tf.log(Z_)

  LF_label = (1+k)/2 if numYs == 2 else k
  per_lf = tf.gather(z_y, tf.cast(LF_label,tf.int32))
  prec_factor = tf.squeeze(msgActive(sbins,k,k) / (msg(sbins, k,k)))

  per_lf_z = tf.reduce_sum(tf.map_fn(lambda y: msgActive(sbins, y,k)/msg(sbins, y,k)*tf.reduce_prod(msg(sbins, y,k)), ys), axis=0)

#             per_lf_prob = prec_factor * per_lf / Z_
  per_lf_prob = prec_factor * per_lf/per_lf_z

 
  marginals_new = tf.expand_dims(tf.nn.softmax(log_pt, axis=0), 2)
  marginals = marginals_new

  loss_new = tf.negative(tf.reduce_sum(pt_1))
  return loss_new, per_lf_prob, marginals

def precision_loss(a_t, n_t, per_lf_prob):
   ptheta_ = a_t * n_t * tf.log(per_lf_prob) + (1-a_t) * n_t * tf.log(1-per_lf_prob)  
   return tf.negative(tf.reduce_sum(ptheta_))

 
numYs = 3
NoOfLFs = 10
batch_size = 32
LFs = [] 
k = tf.convert_to_tensor([lf.label() for lf in LFs], dtype=tf.float64)
isdiscreet = [lf.isdiscreet() for lf in LFs]
user_a = [lf.threshold() for lf in LFs]

# per_lf_prob, loss, marginals = model(numYs, k, thetas, alphas, isdiscreet, user_a)
