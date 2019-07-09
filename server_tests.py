from flask import Flask, render_template, g, current_app
from flask_socketio import SocketIO, emit
import numpy as np
import tensorflow as tf

# retrieve saved data
d = np.load('saved_data/training_data.npz')
m = d['m']
f = d['f']
lowest_limit = d['lowest_limit']
highest_limit = d['highest_limit']
train_data = d['train_data']
# test batch generation example
max_len = 16
batch_size = 320
step = 1
input_rows = m.shape[0] + f.shape[0]
output_rows = m.shape[0]
num_units = 256
learning_rate = 0.001
epochs = 5000
temperature = 0.5

print('--- --- --- before')

def makeSequencerColumn(p):
    m = np.zeros((128))
    '''
    print("lowest_limit: ", lowest_limit)
    print("highest_limit: ", highest_limit)
    print("len(p): ", len(p))
    '''
    tmpIDX = np.append(np.zeros(lowest_limit), p)
    tmpIDX = np.append(tmpIDX, np.zeros(128-highest_limit))
    m[ tmpIDX > 0 ] = 120
    return m
# end makeSequencerMatrix

def rnn(x, weight, bias, input_rows):
    '''
     define rnn cell and prediction
    '''
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, input_rows])
    x = tf.split(x, max_len, 0)
    
    cell = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0)
    outputs, states = tf.contrib.rnn.static_rnn(cell, x, dtype=tf.float32)
    prediction = tf.matmul(outputs[-1], weight) + bias
    return prediction
# end rnn

def sample(predicted_in):
    # keep only positive
    predicted = np.zeros(len(predicted_in))
    passes = np.where(predicted_in >= 0.0)[0]
    next_event = np.zeros( (len(predicted),1) )
    if len(passes) > 4:
        # get the 4 most possible events
        predicted[ passes ] = predicted_in[ passes ]
        passes = predicted.argsort()[-4:][::1]
    elif len(passes) > 0:
        passes = passes[0:np.min([4, len(passes)])]
    
    next_event[passes] = 1
    return next_event
# end sample

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('send features', namespace='/test')
def test_message(features):
    print('--- --- --- test message')

    f1 = features['f1']
    f2 = features['f2']
    x_client = np.array([f1, f2])
    with app.app_context():
        predicted = sess.run([prediction], feed_dict = {x:current_app.seed})
        predicted = np.asarray(predicted[0]).astype('float64')[0]
        predicted_output = sample(predicted)
        remove_fist_event = current_app.seed[:,1:,:]
        new_input = np.append( x_client , predicted_output)
        current_app.seed = np.append(remove_fist_event, np.reshape(new_input, [1, 1, input_rows]), axis=1)
        # make final column
        m = makeSequencerColumn(predicted_output)
        emit('send column', {'column': list(m)})

@socketio.on('my broadcast event', namespace='/test')
def test_message(message):
    emit('my response', {'data': message['data']}, broadcast=True)

@socketio.on('connect', namespace='/test')
def test_connect():
    print('--- --- --- connect')
    emit('my response', {'data': 'Connected'})

@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected')

@socketio.on('initialise', namespace='/test')
def initialise_variables():
    print('--- --- --- initialise')
    session['seed']

if __name__ == '__main__':
    print('--- --- --- main')

    # initialise before sending connection message
    tf.reset_default_graph()

    x = tf.placeholder("float", [None, max_len, input_rows])
    y = tf.placeholder("float", [None, output_rows])
    weight = tf.Variable(tf.random_normal([num_units, output_rows]))
    bias = tf.Variable(tf.random_normal([output_rows]))

    prediction = rnn(x, weight, bias, input_rows)
    dist = tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y)
    cost = tf.reduce_mean(dist)
    
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, 'saved_model/file.ckpt')
    # init_op = tf.global_variables_initializer()
    # sess.run(init_op)

    # GENERATE
    # generate seed
    with app.app_context():
        current_app.seed = train_data[:1:]
        current_app.composition = np.array(current_app.seed[0,:,2:]).transpose()
        print('in init --- seed: ', current_app.seed)

    socketio.run(app, host='0.0.0.0', port=8886)
    app.run(threaded=True)
