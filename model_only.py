import tensorflow as tf

def allocate_params(NoOfLFs, numYs, penalty, th, af):
    alphas = tf.get_variable('alphas', [NoOfLFs], initializer=af, dtype=tf.float64)
    if(penalty in [103, 113]):
        alphas = tf.stop_gradient(alphas*0)
    theta_dim = 1 if penalty % 10 != 7 and numYs == 2 else numYs
    thetas = tf.get_variable('thetas', [theta_dim, NoOfLFs], initializer=th, dtype=tf.float64)
    return thetas, alphas

def model(numYs, k, l, s, thetas, alpha_vars, isdiscrete, user_a, penalty, alpha_max_arg=None, s_thresholds_precision=None):
  # s_thresholds_precision is of shape [numLfs, numAlphaThresholds] and specifies the thresholds at which precision constraints are applied.
  
  # user_a = tf.reshape(user_a, [len(k)])
  is_discrete = tf.convert_to_tensor(isdiscrete, dtype = tf.float64)

  if penalty % 10 == 6:
        user_alphas = tf.zeros_like(alpha_vars)
  else:
      user_alphas = tf.convert_to_tensor(user_a, dtype=tf.float64)
  numYs = 2

  active = is_discrete + (1-is_discrete) * tf.cast(tf.greater_equal(s, user_alphas-0.0001), dtype=tf.float64)
  active = tf.stop_gradient(active)
  
  scaleS=False
  numbins = tf.constant(10, dtype=tf.float64)
  # scale the s_ values to be between 0 and 1.
  if scaleS:
      s_ = tf.clip_by_value((s - user_alphas)/(1-user_alphas+0.00001), 0, 1)
      start_alpha = tf.zeros_like(user_alphas)
  else:
      s_ = tf.stop_gradient(s * active)
      start_alpha = user_alphas
        
  l = tf.clip_by_value(l * active, -1, 1)

  # numbins = 10
  
  #sbin_widths = np.ones_like(user_alphas)/numbins
  sbin_widths = (1-start_alpha)/numbins
  sbins = tf.einsum('i,j->ij', sbin_widths, tf.cast(tf.range(0,numbins+1), dtype=tf.float64)) +tf.expand_dims(start_alpha, 1) 
  sbins = tf.transpose(sbins)
  
  
  if (penalty//10 % 10 == 2):
      if alpha_max_arg is not None:
          alpha_max = tf.convert_to_tensor(alpha_max_arg, dtype = tf.float64)
      else:
          alpha_max = tf.ones_like(alphas)

      alphas = tf.minimum(alpha_vars, alpha_max)
  else:
    alphas = alpha_vars
    
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
      elif (penalty%10 == 6):
          # Gaussian distribution with a variance of 1.
          pos = (y*l+1)/2
          return (1-thetas*thetas*(s-1)*(s-1)*pos/2 - (1-pos)*alphas*alphas*(s)*(s)/2)*tf.abs(l)
      return 0

  def equal_sign(y, k):
    y = discrete(y)
    k = discrete(k)
    eq = tf.cast(tf.equal(y,k), dtype=tf.float64)
    return eq + (1-eq)*-1

  def discrete(y):
      yy = (1+y)/2 if numYs == 2 else y 
      return tf.cast(yy, tf.int32)
    
  def cont_pot(s, y, l):
    if numYs == 2:
      return cont_pots_binary(thetas, s, y, l)
    return cont_pots_binary(thetas[discrete(y)], s, equal_sign(y, k), l)

  def dis_pot(y,l):
      return y*thetas*l if numYs == 2 and penalty%10 != 7 else thetas[discrete(y)]*l*equal_sign(y,k) 

  def pot(s, y, l):
      return (1-is_discrete) * cont_pot(s, y,l) + is_discrete * dis_pot(y,l)

  def msg(s, y, l):
      return (1 +  (1-is_discrete) * tf.reduce_sum(tf.exp(pot(s, y,l)), axis=0)*sbin_widths 
                +  (is_discrete) *  tf.exp(dis_pot(y, l)))
  
  def logmsg(s, y, l):
      return tf.log(msg(s,y,l))
    
  def msgActive(s, y, l, s_thresholds_precisions):
    if (penalty % 10 == 2 or penalty % 10 == 4) and s_thresholds_precisions is None:
        return is_discrete*(msg(s,y,l)-1) + (1-is_discrete)* tf.reduce_sum(tf.exp(pot(s, y,l))*tf.cast(tf.greater(s,alphas),tf.float64), axis=0)*sbin_widths
    elif s_thresholds_precisions is not None:
        return is_discrete*(msg(s,y,l)-1) + (1-is_discrete)* tf.reduce_sum(tf.exp(pot(s, y,l))*tf.cast(tf.greater(s,s_thresholds_precisions),tf.float64), axis=0)*sbin_widths
    else:
        return msg(s, y, l)-1
   
  def logmsgActive(s, y, l, s_thresholds_precisions):
        return tf.log(msgActive(s, y, l, s_thresholds_precisions))
    
  logz_y = tf.map_fn(lambda y: tf.reduce_sum(logmsg(sbins, y,k)), ys)
  logZ_ = tf.reduce_logsumexp(logz_y)

  log_pt = tf.map_fn(lambda y: tf.reduce_sum(pot(s_, y,l), axis=1), ys)
  pt_1 = tf.reduce_logsumexp(log_pt, axis=1) - logZ_
  
  LF_label = tf.squeeze((1+k)/2 if numYs == 2 else k)
    
  per_lf_logz_y = tf.map_fn(lambda y: logmsgActive(sbins, y,k, s_thresholds_precision)-logmsg(sbins, y,k)+tf.reduce_sum(logmsg(sbins, y,k)), ys)

  per_lf_logz_y = tf.transpose(tf.squeeze(per_lf_logz_y))
  # prec_factor = tf.squeeze(per_lf_z_y[LF_label] / (msg(sbins, k,k)))

  per_lf_logz = tf.squeeze(tf.reduce_logsumexp(per_lf_logz_y, axis=1))
  # tmp = logmsgActive(sbins,k,k, s_thresholds_precision) -logmsg(sbins,k,k) + tf.reduce_sum(logmsg(sbins, k,k)) #
  def batch_gather(x,idx):
        cat_idx = tf.concat([tf.expand_dims(tf.range(0, tf.shape(x)[0]),1), tf.expand_dims(idx,1)], axis=1)
        return tf.gather_nd(x, cat_idx)
    
  tmp = batch_gather(per_lf_logz_y, tf.cast(LF_label, tf.int32))
  tmp = tf.squeeze(tmp)
  
  per_lf_logprob = tmp - logZ_
  
  if penalty//10 % 10 == 5:
      per_lf_logprob = tmp - per_lf_logz
    
  marginals_new = tf.expand_dims(tf.nn.softmax(log_pt, axis=0), 2)
  marginals = marginals_new
    
  per_lf_recall = None
  if alpha_max_arg is not None and (penalty//10 % 10 == 3):
     # recall constraints.
     per_lf_recall = msgActive(sbins, k,k, alpha_max_arg)/msg(sbins, k,k)
     
  loss_new = tf.negative(tf.reduce_sum(pt_1))
  return loss_new, per_lf_logprob, marginals, per_lf_recall,  tmp - logZ_

def precision_loss(precisions, n_t, per_lf_logprob, penalty):
   # precisions: [numLFs, numAlphaThresholds]
   if (penalty//10) % 10 == 4:
       print("Using softplus precision constraints")
       return tf.reduce_sum(tf.nn.softplus(n_t*precisions - n_t*tf.exp(per_lf_logprob)))
   else:
       print("Using binomial precision constraints")
       ptheta_ = precisions * n_t * per_lf_logprob + (1-precisions) * n_t * tf.log(tf.maximum(1- tf.exp(per_lf_logprob), 1e-10))
       return tf.negative(tf.reduce_sum(ptheta_))
  
       
   # return tf.constant([0])

def recall_loss(recalls, n_t, per_lf_recall, isdiscrete):
    is_discrete = tf.convert_to_tensor(isdiscrete, dtype = tf.float64)
    return tf.reduce_sum(n_t*is_discrete*tf.nn.softplus(recall-per_lf_recall))
# numYs = 3
# NoOfLFs = 10
# batch_size = 32
# LFs = [] 
# k = tf.convert_to_tensor([lf.label() for lf in LFs], dtype=tf.float64)
# isdiscreet = [lf.isdiscreet() for lf in LFs]
# user_a = [lf.threshold() for lf in LFs]

# per_lf_prob, loss, marginals = model(numYs, k, thetas, alphas, isdiscreet, user_a)
