import matplotlib.pyplot as plt
import pickle

CORPUS = 'LEQ_20_NOMOD'
EXPERIMENT = 'LRD'

for i in range(1,10):
	srnn_file = './eval/' + 'SRNN' + '/' + str(2**i) + '/' + CORPUS + '/' + EXPERIMENT + '/' + 'avg_acc_by_index.p'
	lstm_file = './eval/' + 'LSTM' + '/' + str(2**i) + '/' + CORPUS + '/' + EXPERIMENT + '/' + 'avg_acc_by_index.p'
	gru_file = './eval/' + 'GRU' + '/' + str(2**i) + '/' + CORPUS + '/' + EXPERIMENT + '/' + 'avg_acc_by_index.p'
	outfile = './results/' + CORPUS + '/' + EXPERIMENT + '/' + str(2**i) + '.png'
	SRNN = pickle.load(open(srnn_file, 'rb'))
	LSTM = pickle.load(open(lstm_file, 'rb'))
	GRU = pickle.load(open(gru_file, 'rb'))
	print(len(SRNN), len(LSTM), len(GRU))
	print("======= {} Neurons =======".format(2**i))
	print("SRNN Acc: " + str(SRNN[-1]))
	print("LSTM Acc: " + str(LSTM[-1]))
	print("GRU Acc: " + str(GRU[-1]))
	
	fig = plt.figure()
	ax = plt.subplot()
	ax.plot(SRNN, 'r', label='SRNN')
	ax.plot(LSTM, 'b', label='LSTM')
	ax.plot(GRU, 'g', label='GRU')
	plt.hlines(0.25, 0.0, len(SRNN), colors='black', linestyles='dashed', label='Baseline')
	plt.title('# Hidden Neurons: ' + str(2**i))
	ax.set_xlabel('Character Index')
	ax.set_ylabel('Accuracy')
	# Shrink current axis's height by 10% on the bottom
	box = ax.get_position()
	ax.set_position([box.x0, box.y0 + box.height * 0.15,
                 box.width, box.height * 0.85])
	# Put a legend below x axis label
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=False, ncol=4)
	#legend((SRNN, LSTM, GRU), ('SRNN', 'LSTM', 'GRU'))
	plt.axis([0, len(SRNN), 0.2, 1.0])
	plt.savefig(outfile)
	#plt.show()