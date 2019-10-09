As per the README supplied by the professor, the implementation consists of 4 parts:
  a) generate batch for skip-gram model (word2vec_basic.py)
  b) implement two loss functions to train word embeddings (loss_func.py)
  c) tune the parameters for word embeddings 
  d) apply learned word embeddings to word analogy task (word_analogy.py)
  
  ==================================================================================
  
	a) GENERATION OF BATCH
		- File Name - word2vec_basic.py
		- Method Name - generate_batch
		- Parameters - data, batch_size, num_skips, skip_window
		- Outputs - batch, labels
		- Description : This method is used to create a batch for training data. The process consists of creating pairs  of (context, target words). This can be done be selecting a window of size (skip_window * 2 + 1) with context word in center and words to be predicted around it.
	
		- Pseudocode:
			i. Calculate start and end of window
			ii. Iterate over all the words and select window size data in order
			iii. From the window, select center word as context word
			iv. Iteratively for the next num_skips count, add context word, and target word to data set 
			v. The target word is selected as word with distance 1,2,3... num_skips left and right to the context word
			vi. A flag, flip is used to select words left and right of context word. After every selection flip is toggled and the next time the word is chosen from other side.
			vi. Every time num_skips words are found in a window, the next window is created and the steps are reiterated.
	
   ==================================================================================
   
	b)i) CROSS ENTROPY
		- File Name - loss_func.py
		- Method Name - cross_entropy_loss
		- Parameters - inputs, true_w
		- Outputs - loss_value
		- Description : This method is used to calculate cross_entropy loss for training the model. 		
		- Explanation : 
			Let A = log(exp({u_o}^T v_c))
			and B = log(\sum{exp({u_w}^T v_c)})
			
			so loss = B - A
			
			- How would I calculate A? 
			-> 'A' is the cross product of a center word with each context word.
			Reference for the wording used: https://www.youtube.com/watch?v=ERibwqs9p38,
			Stanford University School of Engineering Lecture 2: Word2Vec
			
			Hence A can be simply calculated as a scalar multiplication of each word in Uo with Vc.
			That gives us multiplication of each value in the matrix and then we sum in up across the rows. 
			Based on the above parameters, Uo = true_w and Vc = inputs.
			
			
			- How would I calculate B?
			-> 'B' is the summation of all the cross products of center words with context words
    
			Hence can be calculated as matrix multiplication of two matrices i.e.,[ Uo(T)(dot)Vc ]
			
		- Pseudocode:
			a) Calculate 'A'
				i) Perform scalar multiplication of inputs(Uo) and true_w (Vc)
				ii) Sum the resultant matrix across rows. That gives us the dot product of Uo.Vc.
				iii) Calculate exponent of the resultant and then take log. This gives us 'A' 
			
			b) Calculate 'B'
				i) Calculate dot product of [ Uo(T) * Vc ]
				ii) Calculate exp of the resultant
				iii) Calculate summation of the resultant
				iv) Calculate log of the resultant above. [Adding noise (7e-10, random) for safe log calculation]
			
			c) loss = B - A
			d) This value will be minimised by the learning algorithm
	
	-----------------------------------------------------------------------------------------
	
	b)ii) NOISE CONTRATIVE ESTIMATIONS
		- File Name - loss_func.py
		- Method Name - nce_loss
		- Parameters - inputs, weights, biases, labels, sample, unigram_prob
		- Outputs - loss_value
		- Description : This method is used to calculate noise contrastive loss for training the model. 		
		- Explanation : 
			We have 
			Jh(theta) = Ph(D=1|w, theta) + Ph(D=0|w, theta)
			Therefore, Jh(theta) = log(sigmoid(scoring_function(labels) - log(k*P(labels)))) + log(1-sigmoid(scoring_function(samples) - log(k*P(samples)))
			
			(Reference from paper) scoring function: 
		
			Embedding lookup (Tensorflow): It enables us to do a lookup in the embedding matrix for a given word
				- embedding_lookup(word) -> Embedding(embedding represnetation of a given word)
				
		- Pseudocode : 
			a) Calculate Ph(D=1|W)
				i) Calculate weights(labels) by doing embedding lookup in weight embedding
				ii) Calculate (qh(T).q(W))  [ qh = inputs ; qw = labels_weights (Result from step(i) ]    --- (Reference from paper) 
				iii) Calculate biases(labels) by doing embedding lookup in the bias embedding
				iv) Scoring function (labels) = step(ii) + step(iii) [ qh(T).q(W) + biases ]
				v) Calculate probabilities(labels) by doing embedding lookup in unigram_prob embedding
				vi) Perform scalar multiplication of above with k( No of samples) 
				vii) Take log of the above result
				viii) Subtract the result of step (vii) from step(iv) [ detla scoring function (labels) = scoring function (labels) - log(k. probability of (labels))]
				ix) Calculate sigmoid of (delta of scoring function (labels))
				x) Calculate log of above sigmoid value.
				xi) let this value be represented by (a).
				
			b) Calculate Ph(D=0|W)
				i) Perform steps (i) - (viii), with paramters as (sample). We will end up with delta scoring function (samples)
				ii) ..
				... ..
				viii) ..
				ix) Calculate sigmoid of above result (sigmoid(delta of scoring function (samples)))
				x) Calculate 1 - result of above. (1 - sigmoid(delta of scoring function (samples)))
				xi) Calculate log of above result. log(1 - sigmoid(delta of scoring function (samples)))
				xii) Add the values of the above resultant matix.
				xiii) Let this value be represented by (b) 
			
			c) Add the result of (a) and (b)
			d) Perform scalar multiplication with -1.
			e) This value will be minimised by the learning algorithm.
			
		
		
		
		- File Name - loss_func.py
		- Method Name - get_delta_scoring_function
		- Parameters - parameter, inputs, weights, biases, labels, sample, unigram_prob
		- Outputs - delta_scoring_function
		- Description : This method is used to calculate delta scoring function for labels or samples.
		- Explanation : Auxilary method
		- Pseudocode : 
			i) Calculate weights(parameter) by doing embedding lookup in weight embedding
			ii) Calculate (qh(T).q(W))  [ qh = inputs ; qw = parameter_weights (Result from step(i) ]    --- (Reference from paper) 
			iii) Calculate biases(parameter) by doing embedding lookup in the bias embedding
			iv) Scoring function (parameter) = step(ii) + step(iii) [ qh(T).q(W) + biases ]
			v) Calculate probabilities(parameter) by doing embedding lookup in unigram_prob embedding
			vi) Perform scalar multiplication of above with k( No of samples) 
			vii) Take log of the above result
			viii) Subtract the result of step (vii) from step(iv) [ detla scoring function (parameter) = scoring function (parameter) - log(k. probability of (parameter))]
			
			
	==========================================================================================
	
	
	d) WORD ANALOGY TASK:
		File Name - word_analogy.py
		- Method Name -  NA
		- Parameters - nce.model or cross_entropy_model , word_analogy_dev.txt
		- Outputs - word_analogy_nce.txt or word_analogy_cross_entropy.txt
 		- Description : This method is used to find the relation between pair of words.
		- Explanation : 
			+ The input word_analogy_dev.txt contains words to predict the most and least illustrative pair using models created.
			+ The pattern of the file is [examples]||[choices]
			+ Each line of this file is divided into "examples" and "choices" by "||"
		- Pseudocode: 
			a) We split the input into 2 parts. First part = left part of the double pipes "||"
			b) For each pair in the list formed by comma-separated output of the previous step:
				+ We put their difference of the vector in a list we maintain
			c) We then calculate the mean of the difference to represent the examples i.e., the left part of the "||"
			d) For the Second part = right part.
			e) For each pair in the list formed by comma-separated output of the previous step:
				+ We calcuate the differnce of their vectors.
				+ We then find the cosine of the above differnce with the mean of vectors (from examples) - step(c)
				+ This step is called the cosine similarity between two vectors and this is what is used to calculate max diff.
				+ This banks of the fact, that if two vectors are similar their cosine similarity i.e., cosine of the two vectors would be close to zero.
				+ If cosine_similarity ~ 0; means similar
				+ The lesser this value; means the more illustrative a pair is.
				+ Conversely, the more this value; means the least illustrative a pair is.
			f) The above ideology is used to calculate the least and most illustrative pair and hence the output of the model for the test case given.
		
				
	==========================================================================================

	c) HYPERPARAMETER TUNING
	
		The parameter candidates available for tuning are:
			max num steps, batch size,skip window, num skips, num sampled
		
	
	
	===========================================================================================
	
	Due to large size of model files, they are uploaded on the google drive.
	
	Cross Entropy: https://drive.google.com/open?id=1szL00nwzrI71j2enZis2tL3MMIBaAfkq
	
	Noise Constrastive Estimation: https://drive.google.com/open?id=1sq3rxq3-uILA8GsTbVcI3uNizg_FE65D
	
		