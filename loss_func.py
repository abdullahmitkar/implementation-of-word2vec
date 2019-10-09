import tensorflow as tf


def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =



    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """

    """
    - How would I calculate A? 
    -> 'A' is the cross product of a center word with each context word.
        Reference for the wording used: https://www.youtube.com/watch?v=ERibwqs9p38,
        Stanford University School of Engineering Lecture 2: Word2Vec
        
        Hence A can be simply calculated as a scalar multiplication of each word in Uo with Vc.
        
        that gives us multiplication of each value in the matrix and then we sum in up across the rows 
        and hence axis = 1 
        
        Based on the above parameters, Uo = true_w and Vc = inputs   
    """

    scalar_multiplication = tf.multiply(inputs, true_w);
    sum_of_rows = tf.reduce_sum(scalar_multiplication, axis=1);
    batch_size = inputs.get_shape()[0];
    uo_vc = tf.reshape(sum_of_rows, [batch_size, 1]);

    A = tf.log(tf.exp(uo_vc) + 1e-10);

    """
    
    - How would I calculate B?
    -> 'B' is the summation of all the cross products of center words with context words
    
        Hence can be calculated as matrix multiplication of two matrices i.e.,[ Uo(T)(dot)Vc ]
           
    """

    Uo_T = tf.transpose(true_w);

    dot_product = tf.matmul(inputs, Uo_T)
    exp_dot_product = tf.exp(dot_product);
    summation_exp_dot_product = tf.reduce_sum(exp_dot_product, axis=1);
    log_summation_exp_dot_product = tf.log(summation_exp_dot_product + 7e-10);

    B = tf.reshape(log_summation_exp_dot_product, [batch_size, 1]);

    return tf.subtract(B, A)


def get_delta_scoring_function(parameter, inputs, weights, biases, labels, sample, unigram_prob):
    # Since this was a redundant task, a function is created to implement this

    k = sample.shape[0]

    vocab_size = biases.shape[0];

    p_weights = tf.nn.embedding_lookup(weights, parameter)

    q_hat_transpose_q_w_for_p = tf.matmul(p_weights, tf.transpose(inputs))

    p_biases = tf.nn.embedding_lookup(biases, parameter)
    s_f_p = q_hat_transpose_q_w_for_p + p_biases;

    pn_param = tf.nn.embedding_lookup(unigram_prob, parameter)
    k_pn_param = tf.scalar_mul(k, pn_param)
    log_k_pn_param = tf.log(k_pn_param)

    delta_s_f_param = s_f_p - log_k_pn_param;
    return delta_s_f_param;


def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    print("Begin NCE Loss")

    # Jh(theta) = log(sigmoid(scoring_function(labels) - log(k*P(labels))))         Ph(D=1|w, theta)
    #            + log(1-sigmoid(scoring_function(samples) - log(k*P(samples)))     Ph(D=0|w, theta)

    # scoring_function(S_theta_w_h) = inputs * weight_labels + bias_labels for ph_d_equals_1
    # scoring_function(S_theta_w_h) = inputs * weight_samples + bias_samples for ph_d_equals_0
    # Calculating Ph_D_equals_one = log(sigmoid(scoring_function(labels)))

    # inputs = q_hat_h[From paper]

    # q_w_labels = weight representation of labels
    # q_w_sample = weight representation of samples

    vocab_size = biases.shape[0];
    biases = tf.reshape(biases, [vocab_size, 1])

    unigram_prob = tf.reshape(tf.convert_to_tensor(unigram_prob), [vocab_size, 1])

    batch_size = labels.shape[0]
    labels = tf.reshape(labels, [batch_size]);

    delta_s_f_labels = get_delta_scoring_function(labels, inputs, weights, biases, labels, sample, unigram_prob);

    sigmoid_labels = tf.sigmoid(delta_s_f_labels)
    log_sigmoid_labels = tf.log(sigmoid_labels + 1e-10);

    delta_s_f_samples = get_delta_scoring_function(sample, inputs, weights, biases, labels, sample, unigram_prob);
    sigmoid_samples = tf.sigmoid(delta_s_f_samples)

    log_one_minus_sigmoid_samples = tf.log((tf.ones(sigmoid_samples.shape) - sigmoid_samples) + 1e-10);
    log_one_minus_sigmoid_samples = tf.reduce_sum(log_one_minus_sigmoid_samples, 0, keepdims=True);

    # J = log_sigmoid_labels + log_one_minus_sigmoid_samples
    J = log_sigmoid_labels + log_one_minus_sigmoid_samples

    J = tf.scalar_mul(-1, J)
    print("J returned")
    return J
