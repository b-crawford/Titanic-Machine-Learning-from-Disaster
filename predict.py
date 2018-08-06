from clean_data import clean_data
import tensorflow as tf
import random
import pandas as pd


df = clean_data('train.csv')

df = pd.get_dummies(df, columns=['Pclass', 'Sex','Deck','Embarked','Title','Survived'], drop_first=False)

indicies = range(df.shape[0])
random.shuffle(indicies)

train = df.iloc[indicies[:600],:]
valid = df.iloc[indicies[600:],:]

x_train = train.drop(['PassengerId','Name','Ticket','Cabin','CabinStr','Survived_0','Survived_1'], axis=1)
y_train = train[['Survived_0','Survived_1']]

x_valid = valid.drop(['PassengerId','Name','Ticket','Cabin','CabinStr','Survived_0','Survived_1'], axis=1)
y_valid= valid[['Survived_0','Survived_1']]



# Neural network

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)


x = tf.placeholder(tf.float32, shape = [None,27])
y = tf.placeholder(tf.float32, shape=[None, 2])


# Layer 1:
W_fc1 = weight_variable([27, 128])
b_fc1 = bias_variable([128])

activ_1 = tf.matmul(x, W_fc1) + b_fc1
hidden_1 = tf.nn.relu(activ_1)

# Layer 2:
W_fc2 = weight_variable([128, 128])
b_fc2 = bias_variable([128])

activ_2 = tf.matmul(hidden_1, W_fc2) + b_fc2
hidden_2 = tf.nn.relu(activ_2)

# Output layer:
W_fc3 = weight_variable([128, 2])
b_fc3 = bias_variable([2])

activ_3 = tf.matmul(hidden_2, W_fc3) + b_fc3
output = tf.nn.sigmoid(activ_3)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


epochs = 10000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(epochs):
        train_step.run(feed_dict={x: x_train, y: y_train})
    print("Accuracy (NN):", accuracy.eval({x: x_valid, y: y_valid}))