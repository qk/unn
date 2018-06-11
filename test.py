from utils import exp_lr, Frame
import sys, subprocess
from argparse import ArgumentParser
import re

import numpy as np
from matplotlib import pyplot as pp
import os
import time
from multiprocessing import cpu_count
from data_loader import MolecularDataPreprocessor, split_sets

def load_simple(filename="data/benzene.npz", skip=10e3, N=None, sizes=[10e3, 2e3, 8e3], verbose=True):
	N = N or sum(sizes)
	skip, N = int(np.round(skip)), int(np.round(N))
	data_raw = Frame(np.load(filename))
	data_raw['index'] = np.arange(len(data_raw))
	data_raw = data_raw[skip:]
	# leak information about your testset, decrease your rmse!
	data_raw = data_raw[np.random.choice(np.arange(len(data_raw)), size=N, replace=False)]
	if verbose: print(data_raw.keys(), len(data_raw["E"]))
	preproc = MolecularDataPreprocessor(expand_method=None, target_key="E", dont_norm='R Z E index'.split())
	sets = split_sets(data_raw['D E R Z index'.split()].copy(), sizes, start=0)
	preproc = preproc.fit(sets[0])
	sets = map(preproc.transform, sets)
	sets = [Frame(s.items()) for s in sets]
	
	if verbose:
		print([len(s) for s in sets], 'sets')
		for k,v in sets[0].items():
			for s in sets:
				print(k, s[k].shape, s[k].mean(), s[k].std())
	
	return preproc, sets

from networks.dropout_he_dtnn import HeteroscedasticDTNN as henet
from networks.dropout_ho_dtnn import HomoscedasticDTNN as honet
from networks.density_dtnn import DensityDTNN as densnet
from networks.density_dtnn import FullDensityDTNN as fulldensnet
from networks.density_dtnn import ReDensityDTNN as redensnet
def test(estimator, data=None, pretrained=False):
	# test("dohe", "data/benzene.npz")
	preproc, (tr, va, te) = load_simple(
		filename=data,
		sizes=[50e3, 2e3, 10e3],
		verbose=False
	)
	if re.match(r'^doh[oe]\d*$', estimator):
		p100 = re.match(r'\D+(\d+)', estimator)
		p = int(p100.group(1))/100 if p100 is not None else 0.8
		print(estimator,'with p', p)
		net = honet if 'doho' in estimator else henet
		print(net, "(selected dropout variance estimator class)")
		nn = net(preproc.scaling_params, tr[0], len(tr), mu_max=10, keep_prob=p, length_scale=20, n_predictions=50, tf_config=None, name=estimator)
	elif estimator == 'density':
		nn = densnet(preproc.scaling_params, tr[0], mu_max=10, tf_config=None, name=estimator)
	elif estimator == 'density_full':
		nn = fulldensnet(preproc.scaling_params, tr[0], mu_max=10, tf_config=None, name=estimator)
	elif estimator == 'density_re':
		nn = redensnet(preproc.scaling_params, tr[0], mu_max=10, tf_config=None, name=estimator)
	else:
		raise ValueError('estimator not defined: {}'.format(estimator))
	if pretrained:
		nn.load()
	else:
		# lr = cycle_lr(1e-3, 1e-5, 5, 200*len(tr)/25, decay="exp")
		lr = exp_lr(1e-3, 1e-4, 3000, floor=1e-4)
		# lr = 1e-3
		tstart = time.time()
		restart = ("rmse", 1.5, 100)
		nn.fit(tr, va, 25, 4000, lr=lr, silent=True, stop_at={"rmse":0.05}, restart_check=restart)
		# nn.fit(tr, va, 25, 3000, lr=lr, silent=True, stop_at={"rmse":0.05})
		# nn.fit(tr, va, 250, 100, lr=lr, plot_loss='rmse', silent=False)
		# nn.fit(tr, va, 25, 10/60, lr=lr, plot_loss='rmse', silent=False)
		# nn.fit(tr, va, 25, 30/60, lr=lr, plot_loss='rmse', silent=False)
		# nn.fit(tr, va, 25, 10, lr=lr, plot_loss='rmse', silent=False)
		sec = time.time() - tstart
		print("fit in {} hours".format(sec/60/60))
		nn.save()

	pr = nn.predict(va)
	print(((va['E']-pr['E'])**2).mean()**0.5, "va rmse")
	pr = nn.predict(te)
	print(((te['E']-pr['E'])**2).mean()**0.5, "te rmse")
	pp.hist(pr["U"], bins=100)
	pp.title("U {}".format(estimator))
	pp.savefig(os.path.join("figs", estimator + ".jpg"))

	nn.close()

if __name__ == "__main__":
	parser = ArgumentParser(description='train a network and print the testset rmse')
	parser.add_argument(
		'estimator', type=str,
		help='net id'
	)
	parser.add_argument(
		'-d', dest='data', type=str,
		help='data directory'
	)
	parser.add_argument(
		'-g', dest='gpu', type=int,
		help='gpu to train on'
	)
	parser.add_argument(
		'--debug', dest='debug', action="store_const",
		const=True, default=False,
		help='wether to drop into debugger on exception'
	)
	parser.add_argument(
		'--pretrained', dest='pretrained', action="store_const",
		const=True, default=False,
		help='load model from ./trained_models/{savename}/session'
	)
	args = vars(parser.parse_args())
	if args['gpu'] is None:
		est_id = args.pop("estimator")
		test_args = dict((k,args[k]) for k in "data pretrained".split())
		try:
			test(est_id, **test_args)
			print("all done")
		except:
			import traceback
			traceback.print_exc()
			if args['debug']:
				import pdb
				pdb.post_mortem()
	else:
		# launch in subprocess with only selected gpu visible to tensorflow
		print('gpu option present, launching in subprocess; chosen gpu', args['gpu'])
		print('modified run command:')
		# strip gpu selection from command (it will be applied to the env variable)
		command = sys.argv
		i = command.index('-g')
		command = ["python3"] + command[:i] + command[i+2:]
		print(" ", " ".join(command))
		env = os.environ.copy() # copy current environment
		env["CUDA_VISIBLE_DEVICES"] = str(args.pop('gpu'))
		proc = subprocess.Popen(
			command, stdout=sys.stdout, stderr=sys.stderr, env=env
		)
		proc.wait()
