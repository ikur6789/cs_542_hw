#!/usr/bin/env python
# coding: utf-8

# # Ian Kurzrock

# ## Assignment 6

# #### Joint probability (of co-occurrence) is most important in science
# #### and engineering in general, and in machine learning in particular.
# #### Refer to what was said about the GLCM (grey level co-occurrence matrix)
# #### of an image for example.
# #### The purpose of this assignment is to implement the computation
# #### of many information-theoretic measures associated with such joint probability,
# #### using TensorFlow functions.
# ####   ...
# #### Refer to modules 4 and 5 again, if necessary. 

# # ....

# ##### Recall that  log base 2 of x  =  (log base 2 of e) * (log base e of x)

# ##### So,    log base 2 of x =  1.44269 * log base e of x

# ###  .....................

# # ............

# #### 1. The usual import(s)

# In[1]:


import numpy
import tensorflow as tf


# #### 2. The usual Session object

# In[2]:


session = tf.Session()


# #### 3. Define a TensorFlow placeholder to hold the joint probability J.
# #####    Sizes are not fixed in advance, so that the below functions
# #####    may be used on any such joint probability.

# In[3]:


J = tf.placeholder("float", shape=[None])


# ####  The actual 3 joint probability tables must be the following:

# In[4]:


p1q1 = [[0.125, 0.125], [0.5, 0.25]]
p2q2 = [[0.08,0.01,0.10], [0.10,0.20,0.01], [0.06,0.08,0.20],[0.03,0.08,0.05]]
p3q3 = [[0.,1.], [0.,0.]]


# #### 4. Define the joint entropy function for any joint probability table J,
# ####     using appropriate TensorFlow functions.
# ####     This entropy must be base 2.
# ####     It should never produce NaN  ot Inf.

# In[5]:


def joint_entropy(J):
    sh = tf.shape(J)
    Q = tf.fill(sh, 1e-10)
    return -tf.reduce_sum(tf.multiply(J, tf.log(J+Q)*1.44269))


# #### 5. Show the joint entropy for p1q1, p2q2, p3q3 above.

# In[6]:


print('Joint entropy of p1q1: ', session.run(joint_entropy(p1q1)))
print('Joint entropy of p2q2: ', session.run(joint_entropy(p2q2)))
print('Joint entropy of p3q3: ', session.run(joint_entropy(p3q3)))


# #### 6. Define the function that extracts the probability vector
# ####     of the first component in the probability table (marginalization).

# In[7]:


def first(J): #p(X)
    
    ten = tf.shape(J)
    rows = ten[0]
    
    s = tf.fill([rows],0) #filling an rows length array with 0's
    
    s = tf.reduce_sum(J, 1) #reduce sum adds tensor elements one by one
                            #ex [3,4,5] [1,2,3] reduce_sum = [4,6,8]
    return s


# #### 7. Show the probability vector for the first component of p1q1, p2q2, p3q3.

# In[8]:


print('First component of joint p1q1: ', session.run(first(p1q1)))
print('First component of joint p2q2: ', session.run(first(p2q2)))
print('First component of joint p3q3: ', session.run(first(p3q3)))


# #### 8. Define the function that extracts the probability vector
# ####     of the second component in the probability table (marginalization).

# In[9]:


def second(J): #p(Y)
    
    ten = tf.shape(J)
    cols = ten[1]
    
    s = tf.fill([cols],0) 
    
    s = tf.reduce_sum(J, 0) 
    
    return s


# #### 9. Show the probability vector for the second component of p1q1, p2q2, p3q3.

# In[10]:


print('Second component of joint p1q1: ', session.run(second(p1q1)))
print('Second component of joint p2q2: ', session.run(second(p2q2)))
print('Second component of joint p3q3: ', session.run(second(p3q3)))


# #### 10. Define the entropy function of the first component 
# ####     for any joint probability table J, using appropriate TensorFlow functions.
# ####     This entropy must be base 2.
# ####     It should never produce NaN  ot Inf.

# In[11]:


def entropy_first(J): #H(X)
    px = first(J)
    return joint_entropy(px)


# #### 11. Show the entropy for the first component of p1q1, p2q2, p3q3.

# In[12]:


print('First entropy of p1q1: ', session.run(entropy_first(p1q1)))
print('First entropy of p2q2: ', session.run(entropy_first(p2q2)))
print('First entropy of p3q3: ', session.run(entropy_first(p3q3)))


# #### 12. Define the entropy function of the second component 
# ####     for any joint probability table J, using appropriate TensorFlow functions.
# ####     This entropy must be base 2.
# ####     It should never produce NaN  ot Inf.

# In[13]:


def entropy_second(J): #H(Y)
    px = second(J)
    return joint_entropy(px)


# #### 13. Show the entropy for the second component of p1q1, p2q2, p3q3.

# In[14]:


print('Second entropy of p1q1: ', session.run(entropy_second(p1q1)))
print('Second entropy of p2q2: ', session.run(entropy_second(p2q2)))
print('Second entropy of p3q3: ', session.run(entropy_second(p3q3)))


# #### 14. Define the conditional entropy function of the first component 
# ####       if (with respect to) second component, for any joint probability table J, 
# ####       using appropriate TensorFlow functions.
# ####     Should be easy, if you recall your entropy formulas.

# In[15]:


def conditional_entropy_first_if_second(J): #H(X|Y)
    return tf.subtract(joint_entropy(J), entropy_second(J))


# #### 15. Show the conditional entropy H(first | second) for the joint p1q1, p2q2, p3q3.

# In[16]:


print('entropy first if second of p1q1: ', session.run(conditional_entropy_first_if_second(p1q1)))
print('entropy first if second of p2q2: ', session.run(conditional_entropy_first_if_second(p2q2)))
print('entropy first if second of p3q3: ', session.run(conditional_entropy_first_if_second(p3q3)))


# #### 16. Define the conditional entropy function of the second component 
# ####       if (with respect to) first component, for any joint probability table J, 
# ####       using appropriate TensorFlow functions.

# In[17]:


def conditional_entropy_second_if_first(J): #H(Y|X)
    return tf.subtract(joint_entropy(J), entropy_first(J))


# #### 17. Show the conditional entropy H(second | first) for the joint p1q1, p2q2, p3q3.

# In[18]:


print('entropy second if first of p1q1: ', session.run(conditional_entropy_second_if_first(p1q1)))
print('entropy second if first of p2q2: ', session.run(conditional_entropy_second_if_first(p2q2)))
print('entropy second if first of p3q3: ', session.run(conditional_entropy_second_if_first(p3q3)))


# ### ....

# #### 18. Define the mutual information function between the first component 
# ####       and the second component, for any joint probability table J, 
# ####       using appropriate TensorFlow functions.
# ####     Should be easy, if you recall your entropy formulas.

# In[19]:


def mutual_information_first_and_second(J): #I(X,Y)
    return tf.subtract(entropy_first(J), conditional_entropy_first_if_second(J))


# #### 19. Show the mutual information I(first;second) for the joint p1q1, p2q2, p3q3.

# In[20]:


print('mutual information first and second of p1q1: ', session.run(mutual_information_first_and_second(p1q1)))
print('mutual information first and second of p2q2: ', session.run(mutual_information_first_and_second(p2q2)))
print('mutual information first and second of p3q3: ', session.run(mutual_information_first_and_second(p3q3)))


# #### 20. Define the mutual information function between the second component 
# ####       and the first component, for any joint probability table J, 
# ####       using appropriate TensorFlow functions.

# In[21]:


def mutual_information_second_and_first(J): #I(Y,X)
    return tf.subtract(entropy_second(J), conditional_entropy_second_if_first(J))


# #### 21. Show the mutual information I(second; first) for the joint p1q1, p2q2, p3q3.
# ####    .......
# ###  Since mutual information I(X;Y) is symmetric, the results must be identical to the above.

# In[22]:


print('mutual information second and first of p1q1: ', session.run(mutual_information_second_and_first(p1q1)))
print('mutual information second and first of p2q2: ', session.run(mutual_information_second_and_first(p2q2)))
print('mutual information second and first of p3q3: ', session.run(mutual_information_second_and_first(p3q3)))


# ## .....

# ### KL (and other divergences, distances) apply only to distributions
# ### of the same length (on the same space of outcomes)

# #### 22. Define the Kullback-Leibler (relative entropy) function  between the first component 
# ####     and the second component for any joint probability table J, using appropriate TensorFlow functions.
# ####     This entropy must be base 2.
# ####     It should never produce NaN  ot Inf.

# In[23]:


def kullback_leibler_first_and_second(J):  #D(X||Y)  
    sh = tf.shape(J)
    Q = tf.fill([sh[1]], 1e-10)
    return -tf.reduce_sum(tf.multiply(first(J), tf.log(second(J)+Q)*1.44269))


# #### 23. Show the Kullback-Leibler divergence KL(first;second) for the joint p1q1 and p3q3.
# ####       p2q2 cannot be used.

# In[24]:


print('KL first||second of p1q1: ', session.run(kullback_leibler_first_and_second(p1q1)))
print('KL first||second of p3q3: ', session.run(kullback_leibler_first_and_second(p3q3)))


# #### 24. Define the Kullback-Leibler (relative entropy) function  between the second component 
# ####     and the first component for any joint probability table J, using appropriate TensorFlow functions.
# ####     This entropy must be base 2.
# ####     It should never produce NaN  ot Inf.

# In[25]:


def kullback_leibler_second_and_first(J): #D(Y||X)
    sh = tf.shape(J)
    Q = tf.fill([sh[0]], 1e-10)
    return -tf.reduce_sum(tf.multiply(second(J), tf.log(first(J)+Q)*1.44269))


# #### 25. Show the Kullback-Leibler divergence KL(second;first) for the joint p1q1 and p3q3.
# ####       p2q2 cannot be used.
# ####   ....
# ###    KL(X;Y) is not symmetric, so you would not always get the same results as above.

# In[26]:


print('KL second||first of p1q1: ', session.run(kullback_leibler_second_and_first(p1q1)))
print('KL second||first of p3q3: ', session.run(kullback_leibler_second_and_first(p3q3)))


# # ................

# ###   There is a true metric (true distance) defined for probability distributions of the same
# ###    length, shown in my write-ups.
# ### ...
# ####   26. Define this true metric function between the components of any joint probability table J, 
# ####   using the conditional entropy functions you defined above.

# In[27]:


def distance_first_second(J): #d(X,Y)
    return tf.add(conditional_entropy_first_if_second(J), conditional_entropy_second_if_first(J))


# #### 27. Show the distance distance(first,second) for the joint p1q1 and p3q3.
# ####       p2q2 cannot be used.
# ####   ....

# In[28]:


print('Distance(first,second) of p1q1: ', session.run(distance_first_second(p1q1)))
print('Distance(first,second)of p3q3: ', session.run(distance_first_second(p3q3)))

