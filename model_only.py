import tensorflow as tf

# The model is controlled by values of the penalty integer which is a three digit decimal
# I kept this only for legacy reasons.  The correct way to do this would be via a config class.
# First digit of penalty, that is, penalty % 10 controls the potential types
# second digit of penalty controls what precision and recall constraints we add.
# This digit controls the top-level method applied which for this model is 1.


def allocate_params(NoOfLFs, numYs, penalty, th, af):
    alphas = tf.get_variable('alphas', [NoOfLFs], initializer=af, dtype=tf.float64)
    if(penalty in [103, 113]):
        alphas = tf.stop_gradient(alphas*0)
    theta_dim = 1 if penalty % 10 != 7 and numYs == 2 else numYs
    thetas = tf.get_variable('thetas', [theta_dim, NoOfLFs], initializer=th, dtype=tf.float64)
    return thetas, alphas

#if binary, convert -1/+1 to 0/1. Else return as it is
def discrete(y, numYs=2):
      yy = (1+y)/2 if numYs == 2 else y 
      return tf.cast(yy, tf.int32)
    
def equal_sign(y, k, numYs=2):
    y = discrete(y,numYs)
    kk = discrete(k,numYs)
    eq = tf.cast(tf.equal(y,kk), dtype=tf.float64)
    return (eq + (1-eq)*-1)*tf.cast(tf.not_equal(k,0),tf.float64)

def batch_gather(x,idx):
    # x : [numRows, numCols]
    # idx = [numRows]
    # collects the column of x for each row as specified in the idx. 
    cat_idx = tf.concat([tf.expand_dims(tf.range(0, tf.shape(x)[0]),1), tf.expand_dims(idx,1)], axis=1)
    return tf.gather_nd(x, cat_idx)


# this is the main routine that defines the graphical model over the l and s variables.
# It can handle both continuous and discrete LFs, and both binary and multi-class labeling tasks.
def model(numYs, k, l, s, thetas, alpha_vars, isdiscrete, user_a, penalty, alpha_max_arg=None, s_thresholds_precision=None):
  # k : [numLFs] fixed label of LF, values are [-1, +1] for binary and [1...numYs] for multiclass. LF for class C1 is -1 for rest
  # l : [numLFs] the labels as output by each LF for the current batch, values are [0,class]
  # s : [numLFs] the scores as output by each LF for the current batch.  
  # user_a : [numLFs] fixed user-specified thresholds on the s values.
  # s_thresholds_precision is of shape [numLfs, numAlphaThresholds] and specifies the thresholds at which precision constraints are applied.
  
  # user_a = tf.reshape(user_a, [len(k)])
  is_discrete = tf.convert_to_tensor(isdiscrete, dtype = tf.float64) #convert boolean to 0/1

  if penalty % 10 == 6:
        # do not threshold on user-alphas.
        user_alphas = tf.zeros_like(alpha_vars)
  else:
      user_alphas = tf.convert_to_tensor(user_a, dtype=tf.float64)

  #Q:should it not be is_discrete*tf.abs(l) + ...
  # This part of the code was because we did not want to change the LFs to support user-provided
  # thresholds on alpha. 
  active = is_discrete + (1-is_discrete) * tf.cast(tf.greater_equal(s, user_alphas-0.0001), dtype=tf.float64) 
  #do not back propoagate on the previous update on active 
  active = tf.stop_gradient(active) 
  
  scaleS=False 
  numbins = tf.constant(10, dtype=tf.float64)
  # scale the s_ values to be between 0 and 1.
  if scaleS:
      # Q:should the expression not be tf.clip_by_value((s - user_alphas)/(tf.max(s)-user_alphas+0.00001), 0, 1). Does this not assume max of s ==1 already?
      # Yes, it assume s == 1.  We should change as per your suggestion.
      s_ = tf.clip_by_value((s - user_alphas)/(1-user_alphas+0.00001), 0, 1)
      start_alpha = tf.zeros_like(user_alphas)
  else:
      s_ = tf.stop_gradient(s * active)
      start_alpha = user_alphas
  
  #Q:will l ever become -1 if it were multiclass?
  # No.
  l = tf.clip_by_value(l * active, -1, 1)

  # numbins = 10
  
  #sbin_widths = np.ones_like(user_alphas)/numbins
  # These bins over s-values is to approximate the integration over s as required.
  # Q:Why are s values never used for computing sbins (of dimension # LFs x numbins) ? Q: Should it not be tf.range(0,numbins)?
  sbin_widths = (1-start_alpha)/numbins
  sbins = tf.einsum('i,j->ij', sbin_widths, tf.cast(tf.range(0,numbins+1), dtype=tf.float64)) +tf.expand_dims(start_alpha, 1) 
  sbins = tf.transpose(sbins)
  
  
  if (penalty//10 % 10 == 2):
      # this was an experiment where user is allowed to set the maximum value of alpha so that the recall per LF is at least that alpha.
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
      #Q: an earlier version correct?
      # sorry, don't get this question.
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
        # Q: should it not be alphas*(1-s)*tf.abs(l)? Yes, changed.
          return thetas*s*tf.cast(tf.equal(y,l),dtype=tf.float64) + \
                  alphas*(-s)*tf.abs(l)*tf.cast(tf.not_equal(y,l), dtype=tf.float64)
      elif (penalty%10 == 6):
          # Gaussian distribution with a variance of 1.
          #pos = 1 if l==y, and pos = 0 otherwise
          #Q: Mixture of two Gaussians, each with variance 1, but what are the two means? Something seems to be wrong.
          # Means is 0 and 1 as I had explained to Pr Ganesh and Raghav.
          pos = (y*l+1)/2
          return (1-thetas*thetas*(s-1)*(s-1)*pos/2 - (1-pos)*alphas*alphas*(s)*(s)/2)*tf.abs(l)
      return 0
  
    
  def cont_pot(s, y, l):
    if numYs == 2:
      return cont_pots_binary(thetas, s, y, l)
    #Q: Not passing numYs to discrete. So wont discrete create problem for numYs=3 by mapping 1 to 1 and 2 also 1?
    # l is either 0 or 1 for numY>2 because of the clipping. The class label is in k.
    return cont_pots_binary(thetas[discrete(y)], s, equal_sign(y, k), l)

  def dis_pot(y,l):
      return y*thetas*l if numYs == 2 and penalty%10 != 7 else thetas[discrete(y)]*equal_sign(y,l) 

  def pot(s, y, l):
      return (1-is_discrete) * cont_pot(s, y,l) + is_discrete * dis_pot(y,l)

  def msg(s, y, l):
      return (1 +  (1-is_discrete) * tf.reduce_sum(tf.exp(pot(s, y,l)), axis=0)*sbin_widths 
                +  (is_discrete) *  tf.exp(dis_pot(y, l)))
  
  def logmsg(s, y, l):
      #TODO: msgs should be computed directly in the log-space.
      return tf.log(msg(s,y,l))
    
  def msgActive(s, y, l, s_thresholds_precisions):
    # this calculates the message from the LFs over the cases when l is non-zero and s is in the active range for continuous LFs.
    if (penalty % 10 == 2 or penalty % 10 == 4) and s_thresholds_precisions is None:
        # In both these cases there is a threshold "alpha" on s.
        return is_discrete*(msg(s,y,l)-1) + (1-is_discrete)* tf.reduce_sum(tf.exp(pot(s, y,l))*tf.cast(tf.greater(s,alphas),tf.float64), axis=0)*sbin_widths
    elif s_thresholds_precisions is not None:
        return is_discrete*(msg(s,y,l)-1) + (1-is_discrete)* tf.reduce_sum(tf.exp(pot(s, y,l))*tf.cast(tf.greater(s,s_thresholds_precisions),tf.float64), axis=0)*sbin_widths
    else:
        return msg(s, y, l)-1
   
  def logmsgActive(s, y, l, s_thresholds_precisions):
        #TODO: msgs should be computed directly in the log-space.
        return tf.log(msgActive(s, y, l, s_thresholds_precisions))
    
  logz_y = tf.map_fn(lambda y: tf.reduce_sum(logmsg(sbins, y,k)), ys)
  logZ_ = tf.reduce_logsumexp(logz_y)

  log_pt = tf.map_fn(lambda y: tf.reduce_sum(pot(s_, y,l), axis=1), ys)
  pt_1 = tf.reduce_logsumexp(log_pt, axis=1) - logZ_
  
  LF_label = tf.squeeze((1+k)/2 if numYs == 2 else k)

  # This calculates the logZ corresponding to the case where the l is active and for each y.
  per_lf_logz_y = tf.map_fn(lambda y: logmsgActive(sbins, y,k, s_thresholds_precision)-logmsg(sbins, y,k)+tf.reduce_sum(logmsg(sbins, y,k)), ys)

  per_lf_logz_y = tf.transpose(tf.squeeze(per_lf_logz_y))
  # prec_factor = tf.squeeze(per_lf_z_y[LF_label] / (msg(sbins, k,k)))

  per_lf_logz = tf.squeeze(tf.reduce_logsumexp(per_lf_logz_y, axis=1))
  # tmp = logmsgActive(sbins,k,k, s_thresholds_precision) -logmsg(sbins,k,k) + tf.reduce_sum(logmsg(sbins, k,k)) #
  
    
  tmp = batch_gather(per_lf_logz_y, tf.cast(LF_label, tf.int32))
  tmp = tf.squeeze(tmp)

  
  if penalty//10 % 10 == 5:
      # computing the precision constraint logPr(l_j = y | l_j is active)
      per_lf_logprob = tmp - per_lf_logz
  else:
      # computing the accuracy constraint using logPr(l_j = y)
      per_lf_logprob = tmp - logZ_
      
  marginals_new = tf.expand_dims(tf.nn.softmax(log_pt, axis=0), 2)
  marginals = marginals_new
    
  per_lf_recall = None
  if alpha_max_arg is not None and (penalty//10 % 10 == 3):
     # recall constraints.
     per_lf_recall = msgActive(sbins, k,k, alpha_max_arg)/msg(sbins, k,k)
     
  loss_new = tf.negative(tf.reduce_sum(pt_1))
  return loss_new, per_lf_logprob, marginals, per_lf_recall,  pot(s_,1,l)

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
