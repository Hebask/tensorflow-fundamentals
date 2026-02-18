import tensorflow as tf
print(tf.__version__)

#1) Creating a vector, scalar, matrix and tensor with any value using tf.constant()
scalar = tf.constant(8)
vector = tf.constant([1 , 2 , 3])
matrix = tf.constant([[1 ,2] , 
                      [3, 4]])

tensor = tf.constant(
                      [
                          [
                              [10 , 9 , 8] , 
                              [1 ,2 , 4], 
                              [8 , 4 , 2]
                          ]
                          ,
                          [
                              [10 , 9 , 8] , 
                              [1 ,2 , 4], 
                              [8 , 4 , 2]
                          ]
                      ]
                     )
scalar , vector , matrix , tensor 

#2) Finding the shape, rank and size
print(f'The shape of the tensor: {tensor.shape}')
print(f'The rank of the tensor: {tf.size(tensor)}')
print(f'The rank of the tensor: {tf.rank(tensor)}')

#3) Create two tensors containing random values between 0 and 1 with shape [5, 300].
# Set seed
tf.random.set_seed(42)
# Setting up the shape 
tensor_1 = tf.random.uniform(shape=(5, 300))
tensor_2 = tf.random.uniform(shape=(5, 300))
tensor_1, tensor_1.shape, tensor_2, tensor_2.shape

#4) Multiply the two tensors created in 3 using matrix multiplication.
print('Shape before the matrix multiplication')
print(tensor_1.shape , '\n')
print(tensor_2.shape , '\n')
resultant_vector = tf.linalg.matmul(tensor_1 , tf.transpose(tensor_2))
print(f'After multiplying the resultant vector is: {resultant_vector} \n')
print(f'The resultant vector shape: {resultant_vector.shape}')

#5) Multiply the two tensors created in 3 using dot product.
tf.tensordot(tensor_1 , tf.transpose(tensor_2) , axes=1)

#6) Create a tensor with random values between 0 and 1 with shape [224, 224, 3].
big_tensor = tf.random.uniform(shape = [224 , 224 , 3], minval = 0 , maxval = 1)
print(f'Maximum value: {tf.reduce_max(big_tensor)}')
print(f'Minimum value: {tf.reduce_min(big_tensor)}')

#7) Find the min and max values of the tensor created in 6.
minimum = tf.math.reduce_min(big_tensor)
print(f'The minimum value is: {minimum}')
maximum = tf.math.reduce_max(big_tensor)
print(f'The maximum value is: {maximum}')

#8) Created a tensor with random values of shape [1, 224, 224, 3] then squeeze it to change the shape to [224, 224, 3].
unsqueezed_tensor = tf.random.Generator.from_seed(42)
unsqueezed_tensor = unsqueezed_tensor.normal(shape = (1 , 224 , 224 , 3))
print(f'The shape before squeezing: {unsqueezed_tensor.shape}')
squeezed_tensor = tf.squeeze(unsqueezed_tensor)
print(f'The shape after squeezing: {squeezed_tensor.shape}')

#9) Create a tensor with shape [10] using own choice of values, then find the index which has the maximum value.
ten_tensor = tf.constant([0 ,1 ,2 ,3, 4, 9, 10, 0.1 , -000.1 , 6])
max_value = tf.math.argmax(ten_tensor)
print(f'The indices where the value is maximum: {max_value}')
print(f'The maximum value was: {ten_tensor[max_value]}')

#10) One-hot encode the tensor you created in 9.
# Throw's an error since I had floating numbers in my tensor, casting to int solved! 
tf.cast(ten_tensor , dtype=tf.int32)
# One hot encoding the tensor of shape 10 
tf.one_hot(tf.cast(ten_tensor , dtype=tf.int32) , depth = 10)
