import numpy as np
import scipy as sp
from joblib import Parallel, delayed
from matplotlib import pyplot as pp
from multiprocessing import cpu_count
from scipy import spatial
import scipy.cluster
import scipy.stats
from utils import *
import os
import sys
import tensorflow as tf
import time
# commandline
import sys, subprocess
from argparse import ArgumentParser

n_cores = cpu_count()

def only(D, keys):
	data_dict = {}
	for k in keys:
		data_dict[k] = D[k].copy()
	return data_dict

def split_sets(D, sizes, start=0, shuffle=False):
	"""
	splits dictionary of arrays into subsets according to size
	it discards points before start, shuffles the remaining points (optional), then splits them.
	D should be passed through only(), which returns copies of the selected arrays
	"""
	D = dict(**D) # in case D was a Frame, because it doesn't allow arrays of different lengths
	sizes = [int(s) for s in sizes]
	start = int(start)

	any_key = list(D.keys())[0]
	N = len(D[any_key])
	if -1 in sizes:
		indices = [i for i,s in enumerate(sizes) if s == -1]
		c = len(indices)
		remaining = N - start - sum(sizes) - c
		for i in indices:
			sizes[i] = remaining//c
		for i in range(1, 1 + remaining - c*(remaining//c)):
			sizes[indices[-i]] += 1
		if sum(sizes) != N-start:
			print(sum(sizes), N-start, sizes)
			raise RuntimeError('calculated dynamic sizes don\'t add up correctly')
	print(sizes)

	end = start + sum(sizes)
	if end > N:
		raise ValueError("ERROR: Not enough data for all sets: %s -> %i > %i = N"
				% (str(sizes), sum([trN, vaN, teN]), N))

	I = np.arange(N)[start:end]
	if shuffle:
		print("You really shouldn't use shuffle unless you're certain your points cover all possible configurations. it leaks information into the validation and testsets. The resulting model overfits to your dataset.")
		I = np.random.permutation(I)
	for k in D.keys():
		D[k] = D[k][I]

	start = 0
	sets = [{} for _ in range(len(sizes))]
	for n,splitD in zip(sizes,sets):
		for k in D.keys():
			splitD[k] = D[k][start:start+n]
		start += n

	return sets

# def split_sets_old(D, trN, vaN, teN, start=0, shuffle=False):
	# # D should be passed through only(), which returns a copies of the selected arrays

	# any_key = list(D.keys())[0]
	# N = len(D[any_key])

	# end = start + sum([trN, vaN, teN])
	# if end > N:
		# raise ValueError("ERROR: Not enough data for all sets: %i+%i+%i = %i > %i = N"
				# % (trN, vaN, teN, sum([trN, vaN, teN]), N))

	# I = np.arange(N)[start:end]
	# if shuffle:
		# print("You really shouldn't use shuffle. it leaks information into the validation and testsets. The resulting model overfits to your training data.")
		# I = np.random.permutation(I)
	# for k in D.keys():
		# D[k] = D[k][I]

	# trD, vaD, teD = {}, {}, {}
	# for k in D.keys():
		# trD[k] = D[k][:trN]
		# vaD[k] = D[k][trN:trN+vaN]
		# # teD[k] = np.concatenate((D[k][trN:trN+vaN][teI], D[k][trN+vaN:trN+vaN+teN]), axis=0)
		# teD[k] = D[k][trN+vaN:trN+vaN+teN]

	# return trD, vaD, teD

def subsample(D, size, replace=False):
	any_key = list(D.keys())[0]
	N = len(D[any_key])
	if size > N: raise ValueError("subsample size (%i) is greater than data set size (%i)" % (size, N))
	sample = np.random.choice(np.arange(N), size=size, replace=replace)
	sampleD = {}
	for k in D.keys():
		sampleD[k] = D[k][sample].copy()
	return sampleD

def _expand_grid_gauss(D, mu_min, delta_mu, sigma, Mu):
	G = np.stack([np.exp(-((D - mu)**2)/(2*(sigma**2))) for mu in Mu], -1)
	return G
def _expand_prefitted_gauss(D, G):
	G = np.stack([np.exp(-((D - mu)**2)/(2*sigma2)) for mu, sigma2 in G], -1)
	return G
def _expand_clip_one_hot(D,l,delta):
	# D[D<l] = 0.0
	# D[D>l+delta*2] = 0.0
	D[D!=0] = (D[D!=0] - l)/delta
	D[D!=0] = 1/(1+D[D!=0]**2)
	# linear
	# D[D!=0] = -(D[D!=0]-l)
	return D
def _expand_grid_clip(D, L, delta):
	G = np.stack([clip_one_hot(D.copy(), l, delta) for l in L], -1)
	return G

class MolecularDataPreprocessor:
	def __init__(self, scale=True, width=0.2, spacing=0.2, eps=1e-5, selection_matrices_key=None, expand_method="gauss", target_key=None, dont_norm="E Z S index".split()):
		self.scale = scale
		self.expand_method = expand_method
		self.selection_matrices_key = selection_matrices_key
		self.target_key = target_key
		self.dont_norm = dont_norm
		self.expanded_dists_scaling_params_are_fitted = False
		self.width = width
		self.spacing = spacing
		self.eps = eps

	def fit(self, D):
		self.keys = list(D.keys())
		scale_dims = set(self.keys) if self.scale else set()
		self.scale_dims = scale_dims - set(['selection_matrices'])

		msg, is_ok = self.check(D)
		if not is_ok:
			print(msg, "=> fitting skipped")
			return self

		if self.expand_method is None:
			pass
		else:
			if not "D" in self.keys:
				raise RuntimeError("expand_method is {} but there are no interatomic distances 'D' in keys {}".format(self.expand_method, str(self.keys)))
			if self.expand_method == 'grid_gauss':
				# gaussian expansion of interatomic distances
				# sigma and delta_mu may have to be 0.1, 0.1
				delta_mu = self.spacing # gap between gaussians, in angstrom
				sigma = self.width # width of gaussians
				print(sigma, delta_mu, "width of and gap between gaussians")
				# mu_min, mu_max = -1.0, D['D'].max()*1.5
				mu_min, mu_max = -1.0, 10
				#  mu_max in paper:
				#  10A for benzene and GDB7
				#  15A for toluene, malonaldehyde, and salicylic acid
				#  20A for GDB9
				# grid steps: 0 <= k <= mu_max/delta_mu
				K = np.linspace(0, mu_max/delta_mu, int(np.ceil(mu_max/delta_mu))) # not exact, but should be fine
				# gaussian centers on the grid
				Mu = [mu_min + k*delta_mu for k in K]
				self.expand_params = {
					"delta_mu": delta_mu,
					"sigma": sigma,
					"mu_min": mu_min,
					"mu_max": mu_max,
					"Mu": Mu
				}
			elif self.expand_method == 'fitted_gauss':
				G = np.load("tmp/G.npy")
				G_new = []
				for mu,sigma in G:
					G_new += [[mu+delta*np.sqrt(sigma), sigma] for delta in [-2,-1,0,1,2]]
				G = np.array(G_new)
				print(G, "expansions around fitted gaussian G")
				self.expand_params = {"G": G}
			elif self.expand_method == 'clip':
				B = np.load("tmp/B.npy")
				print(B, "clip regions B")
				self.expand_params = {"B": B}
			elif self.expand_method == 'grid_clip':
				delta = 0.2
				L = np.arange(-1,D.max()+delta,delta)
				self.expand_params = {"delta":delta, "L":L}

		# selection matrices
		if self.selection_matrices_key is not None:
			if self.selection_matrices_key not in self.keys:
				print("WARNING:", self.selection_matrices_key, "is not included in keys, transformation into selection matrices S will be skipped.")
			else:
				print(self.selection_matrices_key, "will be converted to selection matrices")
				self.selection_matrices_max_i = np.max(D[self.selection_matrices_key])+1
				self.keys += ["selection_matrices"]

		# scaling
		self.scaling_params = {}
		for k in self.scale_dims:
			# compute scaling factors
			sigma = D[k].std(axis=0)
			if len(sigma.shape) > 0:
				sigma[sigma==0.0] = 1.0
			elif sigma == 0:
				sigma = 1.0
			mu = D[k].mean(axis=0)
			self.scaling_params[k] = [mu, sigma]

		return self

	def transform(self, D, permute_indices=False):
		'''
		this will try to process data in place. be careful to not reuse it's output.
		'''

		msg, is_ok = self.check(D)
		if not is_ok:
			print(msg, "=> transformation skipped")
			return D

		# expand interatomic distances
		if self.expand_method is None:
			pass
		else:
			if not "D" in self.keys:
				print("expand_method is", self.expand_method, "but there are no interatomic distances 'D' in keys", self.keys)
				print("Aborting.")
				return
			R = D["D"] # interatomic distances
			print(R.max(), "max interatomic distance")
			if self.expand_method == 'grid_gauss':
				args = [self.expand_params[k] for k in ["mu_min", "delta_mu", "sigma", "Mu"]]
				R = np.concatenate(
					Parallel(n_cores//2, 'threading')(
						delayed(_expand_grid_gauss)(*([P]+args))
							for P in np.array_split(R, R.shape[0]//2000+1, axis=0)),
					axis=0)
			elif self.expand_method == 'fitted_gauss':
				R = np.concatenate(
					Parallel(n_cores//2, 'threading')(
						delayed(_expand_prefitted_gauss)(P, self.expand_params["G"])
							for P in np.array_split(R, R.shape[0]//2000+1, axis=0)),
					axis=0)
			elif self.expand_method == 'clip':
				R = np.concatenate(
					Parallel(n_cores//2, 'threading')(
						delayed(_expand_clip_one_hot)(P, self.expand_params["B"])
							for P in np.array_split(R, R.shape[0]//2000+1, axis=0)),
					axis=0)
			elif self.expand_method == 'grid_clip':
				delta, L = [self.expand_params[k] for k in ["delta", "L"]]
				R = np.concatenate(
					Parallel(n_cores//2, 'threading')(
						delayed(_expand_grid_clip)(P, L, delta)
							for P in np.array_split(R, R.shape[0]//2000+1, axis=0)),
					axis=0)
			print("filtering values lower than", self.eps)
			D["D"] = R * (R >= self.eps)
			del R

			# update scaling parameters (do it here, because it is computationally expensive. otherwise you'd have to expand the training set twice.)
			if not self.expanded_dists_scaling_params_are_fitted:
				self.expanded_dists_scaling_params_are_fitted = True
				if "D" not in self.dont_norm:
					sigma = D["D"].std(axis=0)
					if len(sigma.shape) > 0:
						sigma[sigma==0.0] = 1.0
					elif sigma == 0:
						sigma = 1.0
					mu = D["D"].mean(axis=0)
					self.scaling_params["D"] = [mu, sigma]

		# transform Z into selection matrices
		if self.selection_matrices_key is not None:
			if self.selection_matrices_key not in self.keys:
				print("WARNING:", self.selection_matrices_key, "is not included in keys, transformation into selection matrices S will be skipped.")
			else:
				print(self.selection_matrices_key, "will be converted to selection matrices")
				intS = D[self.selection_matrices_key] # integer keys (not checked btw.)
				D["selection_matrices"] = np.eye(self.selection_matrices_max_i).T[intS] # (points, atoms, atom types)
				print(D["selection_matrices"].shape, "selection_matrices")
				# then do S.dot(C_mc) to get molecule representation C_nac

		# apply scaling
		for k in self.scale_dims:
			mu, sigma = self.scaling_params[k]
			# print(k, 'with ~checksums for mean, std', np.mean(mu), np.mean(sigma), mu.shape, sigma.shape, end=" ")
			if D[k].dtype != float:
				D[k] = D[k].astype(float)
			if k not in self.dont_norm:
				D[k] -= mu
				D[k] /= sigma
				# print('applied')
			else:
				# print('not applied!')
				pass

		# permute atom indices
		if permute_indices:
			print("WARNING: Permuting atom indices is experimental, recheck if it works correctly.")
			print("DTNN is insensitive to atom index permutation, so test error should not change.")
			print("Also, this is a very simple implementation that assumes you have exactly two types")
			print("of atoms with equal atom counts (C6H6 f.i.), and that the distance matrix is")
			print("bipartitioned s.t. each half of the matrix indices belongs to only one atom type.")
			print("(So first half only for C-atoms, second half only for H-atoms, f.i. .)")
			q,a,*_ = D["D"].shape
			ix = np.hstack([
					np.random.permutation(np.arange(0, a//2)),
					np.random.permutation(np.arange(a//2, a))
				])
			D["D"] = D["D"][:,ix,:][:,:,ix]

		if self.target_key is not None:
			if "T" not in D:
				D["T"] = D[self.target_key]
			else:
				raise ValueError("Cannot set target, data dict already has array key 'T'")
		return D

	def fit_transform(self, D):
		return self.fit(D).transform(D)

	def check(self, D):
		# if non empty
		msg = ""
		if sum(v.shape[0] for v in D.values()) == 0:
			msg += "D arrays are empty"
		return msg, len(msg) == 0


def prepare_data(parsed_data_filename, out_folder=None, skip=10e3, N=None, sizes=(10e3, 2e3, 8e3)):
	'''
	splits data into training, validation and testsets
	saves them in trained_models/{outfolder}/
	parameters
	----------
	parsed_data_filename -- .npz file with data after first stage of preprocessing
	out_folder -- where to store the processed tr/va/te{i}.npz files
	skip -- number of datapoints to discard, starting from point index 0
	N -- number of points to randomly draw from the data. useful when using wildcards in the sizes parameter. defaults to sum(sizes).
	sizes -- (n_tr, n_va, n_te), how much data goes into which split. -1 can be used as a wildcard and will be replaced by the remaining points. points will be divided evenly across multiple wildcards.
	gaussgrid -- wether to expand distances on a gaussian grid
	verbose -- wether to print debugging messages
	'''
	if len(sizes) != 3 and len(sizes) != 2:
		raise ValueError("sizes must be a tuple of 2 or 3 values")
	data_raw = Frame(np.load(parsed_data_filename))
	if N is None:
		if -1 not in sizes:
			N = sum(sizes)
		else:
			N = len(data_raw) - skip
	skip, N = int(np.round(skip)), int(np.round(N))
	print(N, "data points total")
	print(data_raw.keys(), "keys")
	data_raw["index"] = np.arange(len(data_raw))
	data_raw = data_raw[skip:]
	data_raw['T'] = data_raw['E'] # generic key for target value (required)
	# leaks information about the testset, but we're assuming the dataset samples all possible states sufficiently
	data_raw = data_raw[np.random.choice(np.arange(len(data_raw)), size=N, replace=False)]
	print(data_raw.keys(), len(data_raw["E"]))
	sets = split_sets(data_raw['T R Z E index'.split()].copy(), sizes, start=0)
	if len(sets) == 3:
		tr, va, te = sets
	elif len(sets) == 2:
		tr, va = sets
		te = None
	print(len(tr["R"]), len(va["R"]), len(te["R"]) if te is not None else None)
	# the preprocessor doesn't do much here, but will yield mean and std for each 
	# array, which are required for later
	preproc = MolecularDataPreprocessor(expand_method=None, eps=None, target_key="E")
	preproc.fit(tr)

	# extract extra training set in case additonal calibration is needed
	# this sample is produced in the same way as the training set
	N = len(te)
	I = np.random.choice(np.arange(N), size=min(2000,N//5), replace=False)
	I = (np.arange(N)[:,None] == I[None,:]).max(axis=1)
	print(len(I), len(te), len(te[I]), len(te[~I]))
	tr_k = te[I]
	te = te[~I]

	# save it all to disk for transparency, don't apply any transformation
	folder = out_folder
	os.makedirs(folder, exist_ok=True)
	np.savez(os.path.join(folder, "tr.npz"), **tr)
	np.savez(os.path.join(folder, "tr_k.npz"), **tr_k)
	np.savez(os.path.join(folder, "va.npz"), **va)
	if te is not None:
		np.savez(os.path.join(folder, "te.npz"), **te)
	print(np.load(os.path.join(folder, 'tr.npz')).keys(), "keys remaining after transformation")

	np.savez(os.path.join(folder, "tr_mean_std.npz"), **preproc.scaling_params)
	return

if __name__ == "__main__":
	parser = ArgumentParser(description='split the data and store those splits in the output directory')
	parser.add_argument(
		'parsed_data_filename', type=str,
		help='.npz file containing dataset'
	)
	parser.add_argument(
		'sizes', nargs='+', type=float,
		help='sizes'
	)
	parser.add_argument(
		'-d', dest='out_folder', type=str,
		help='output directory'
	)
	parser.add_argument(
		'-a', dest='skip', type=float,
		default=0,
		help='wether to calibrate the interval width'
	)
	parser.add_argument(
		'--debug', dest='debug', action="store_const",
		const=True, default=False,
		help='wether to drop into debugger on exception'
	)
	args = vars(parser.parse_args())
	# print(args)
	datapath = args.pop("parsed_data_filename")
	debug = args.pop("debug")
	if not debug:
		prepare_data(datapath, **args)
	else: 
		try:
			prepare_data(datapath, **args)
		except:
			import traceback, pdb
			traceback.print_exc()
			pdb.post_mortem()
	print("all done")
