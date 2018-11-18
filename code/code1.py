import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
#from ann_visualizer.visualize import ann_viz
# 
from sklearn.preprocessing import  StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from IPython.display import display, Math, Latex

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(np.asarray(X_tr).reshape(-1,1))
X_test_scaled  = scaler.transform(np.asarray(X_te).reshape(-1,1))


input_dim = np.asarray(X_train_scaled).shape[1]
encoding_dim = 6

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="tanh",activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="tanh")(encoder)
encoder = Dense(int(2), activation="tanh")(encoder)
decoder = Dense(int(encoding_dim/ 2), activation='tanh')(encoder)
decoder = Dense(int(encoding_dim), activation='tanh')(decoder)
decoder = Dense(input_dim, activation='tanh')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.summary()


import datetime
nb_epoch = 100
batch_size = 16
autoencoder.compile(optimizer='adam', loss='mse' )

t_ini = datetime.datetime.now()
history = autoencoder.fit(np.asarray(X_train_scaled ), np.asarray(X_train_scaled ),#.reshape(16, 1),
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_split=0.1,
                        verbose=0
                        )

t_fin = datetime.datetime.now()
print('Time to run the model: {} Sec.'.format((t_fin - t_ini).total_seconds()))



df_history = pd.DataFrame(history.history)



predictions = autoencoder.predict(np.asarray(X_test_scaled ).reshape(-1,1))

mse = np.mean(np.power(X_test_scaled  - predictions, 2), axis=1)
dates = np.arange(0,3863, 1)
df_error = pd.DataFrame({'reconstruction_error': mse.reshape(-1), 'Label': np.asarray(y_test).reshape(-1)},index=dates)
df_error.describe()


outliers = df_error.index[df_error.reconstruction_error < 0.4].tolist()
outliers




# plot the data

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1)
fig.autofmt_xdate()

#total_re1=np.asarray(total_re) *  std_y_train +  mean_y_train
#test_values = np.asarray(total_re)*np.std(Y)+np.mean(Y)#.reshape(-1)
#test_dates = np.arange(60)
X_te=np.asarray(X_te).reshape(-1,)
#test_series = pd.Series(Y[:160], index=test_dates)
plot_test, = ax.plot(X_te, label='true')

plot_test, = ax.plot(X_te, label='true')
#xfmt = mdates.DateFormatter('%b %d %H')
#ax.xaxis.set_major_formatter(xfmt)


#plt.scatter(range(len(X_te)), df_error.reconstruction_error, color="#7A68A6")
plt.plot(outliers,df_error.reconstruction_error[df_error.reconstruction_error>0.4], 'ro') 

plt.axis()

# ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d %H')
plt.title('Outlier detection(scaled)')
#plt.legend(handles=[plot_predicted, plot_test])



#plt.fill_between(test_xs_scaled[:, 0], lower[:, 0], upper[:, 0], color='yellow', alpha=0.5)
plt.show()

import pickle
pickle.dump(outliers,open('outliers.p', 'wb'))
pickle.dump(df_error.reconstruction_error,open('df_error.reconstruction_error.p', 'wb'))
