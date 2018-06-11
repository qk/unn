# from uncertainty.cluster import ClusterUncertainty
from uncertainty.dummy import DummyUncertainty
from uncertainty.ensemble import EnsembleUncertainty
# from uncertainty.gradient_boosting import GradientBoostingUncertainty
from uncertainty.knn import KNNUncertainty
# from uncertainty.quantile_forest import QuantileForestUncertainty
from uncertainty.wrapper import UncertaintyWrapper
from numpy import ceil, load
from networks.shims import params
from utils import Frame
import os
import scipy
import scipy.stats

from networks.shims import DensityShim, MixtureShim, DTNNShim
from networks.shims import HomoscedasticShim60 as HoShim60
from networks.shims import HomoscedasticShim70 as HoShim70
from networks.shims import HomoscedasticShim80 as HoShim80
from networks.shims import HomoscedasticShim90 as HoShim90
from networks.shims import HeteroscedasticShim60 as HeShim60
from networks.shims import HeteroscedasticShim70 as HeShim70
from networks.shims import HeteroscedasticShim80 as HeShim80
from networks.shims import HeteroscedasticShim90 as HeShim90
from networks.shims import FullDensityShim
from networks.shims import SteppedDensityShim
from networks.shims import HalfDensityShim
from networks.shims import HalfMixtureShim
from networks.shims import FullMixtureShim
from networks.shims import SteppedMixtureShim
from networks.shims import ReMixtureShim

def create_instances(data_dir="trained_models/benzene/", folder_prefix="benzene/", n_jobs=1, only_est=None, gpus=(1,), mu_max=10):
	'''
	returns a list of (name,constructor) tuples of uncertainty estimators to train 
	and predict with in the evaluation loop.
	'''
	fp = folder_prefix
	if fp[:len("trained_models")] == "trained_models":
		# all models are implied to be saved under trained_models/, no need to 
		# explicitely prefix it
		fp = fp[len("trained_models/"):]
	instances = []
	dtnn = lambda: DTNNShim(name=os.path.join(fp,"dtnn"))
	normal_ppf = scipy.stats.norm.ppf
	t_ppf = lambda df: lambda x: scipy.stats.t.ppf(x, df=df)
	t50_ppf = t_ppf(50)
	ensemble_members = 24
	days = ceil(ensemble_members/sum(gpus))*72/24
	print("ensemble training will take approx.", days, "days")
	tr = Frame(**load(os.path.join(data_dir, "tr.npz")))
	params["tr_mean_std"] = Frame(**load(os.path.join(data_dir, "tr_mean_std.npz")))
	params["point"] = tr[0]
	params["n_tr_points"] = len(tr)
	params["mu_max"] = mu_max

	# basic baseline, nothing should perform worse than these
	instances += [("dummy-const", lambda: DummyUncertainty(dtnn, method="const"))]
	# these rely on the points being sorted (from one trajectory)
	instances += [("dummy-f'", lambda: DummyUncertainty(dtnn, method="f'"))]
	instances += [("dummy-f''", lambda: DummyUncertainty(dtnn, method="f''"))]
	instances += [("dummy-1/f'", lambda: DummyUncertainty(dtnn, method="1/f'"))]
	instances += [("dummy-1/f''", lambda: DummyUncertainty(dtnn, method="1/f''"))]

	# dropout variance ~ gal2015
	instances += [('doho60', lambda: UncertaintyWrapper(HoShim60, ppf=t50_ppf, name=os.path.join(fp, 'doho60')))]
	instances += [('doho70', lambda: UncertaintyWrapper(HoShim70, ppf=t50_ppf, name=os.path.join(fp, 'doho70')))]
	instances += [('doho80', lambda: UncertaintyWrapper(HoShim80, ppf=t50_ppf, name=os.path.join(fp, 'doho80')))]
	instances += [('doho90', lambda: UncertaintyWrapper(HoShim90, ppf=t50_ppf, name=os.path.join(fp, 'doho90')))]

	instances += [('dohe60', lambda: UncertaintyWrapper(HeShim60, ppf=t50_ppf, name=os.path.join(fp, 'dohe60')))]
	instances += [('dohe70', lambda: UncertaintyWrapper(HeShim70, ppf=t50_ppf, name=os.path.join(fp, 'dohe70')))]
	instances += [('dohe80', lambda: UncertaintyWrapper(HeShim80, ppf=t50_ppf, name=os.path.join(fp, 'dohe80')))]
	instances += [('dohe90', lambda: UncertaintyWrapper(HeShim90, ppf=t50_ppf, name=os.path.join(fp, 'dohe90')))]

	# conditional density estimation [bishop1995]
	instances += [("density", lambda: UncertaintyWrapper(DensityShim, ppf=normal_ppf, name=os.path.join(fp, "density")))]
	instances += [("density_full", lambda: UncertaintyWrapper(FullDensityShim, ppf=normal_ppf, name=os.path.join(fp, "density_full")))]
	instances += [("density_stepped", lambda: UncertaintyWrapper(SteppedDensityShim, ppf=normal_ppf, name=os.path.join(fp, "density_stepped")))]
	instances += [("density_half", lambda: UncertaintyWrapper(HalfDensityShim, ppf=normal_ppf, name=os.path.join(fp, "density_half")))]

	instances += [("mixture", lambda: UncertaintyWrapper(MixtureShim, ppf=normal_ppf, name=os.path.join(fp, "mixture")))]
	instances += [("mixture_half", lambda: UncertaintyWrapper(HalfMixtureShim, ppf=normal_ppf, name=os.path.join(fp, "mixture_half")))]
	instances += [("mixture_full", lambda: UncertaintyWrapper(FullMixtureShim, ppf=normal_ppf, name=os.path.join(fp, "mixture_full")))]
	instances += [("mixture_stepped", lambda: UncertaintyWrapper(SteppedMixtureShim, ppf=normal_ppf, name=os.path.join(fp, "mixture_stepped")))]
	instances += [("mixture_re", lambda: UncertaintyWrapper(ReMixtureShim, ppf=normal_ppf, name=os.path.join(fp, "mixture_re")))]

	# print("skipping quantile regression")
	# if False:
		# instances += [("quantile-regression", lambda: GradientBoostingUncertainty(DTNNShim, keys="DY", n_estimators=100, max_depth=300, learning_rate=.1, min_samples_leaf=9, min_samples_split=9))]
		# est = lambda: QuantileForestUncertainty(DTNNShim, keys="DY", n_estimators=500, max_depth=30, n_jobs=n_jobs)
			# # let's try it how the org. authors suggested as well (e.i. not include prY)
		# instances += [("quantile-regression-no-y", lambda: GradientBoostingUncertainty(DTNNShim, keys="D", n_estimators=100, max_depth=300, learning_rate=.1, min_samples_leaf=9, min_samples_split=9))]
		# instances += [("quantile-forest-no-y", lambda: QuantileForestUncertainty(DTNNShim, keys="D", n_estimators=500, max_depth=30, n_jobs=n_jobs))]
		# fuzzy c-means clustering [pevec2013 shrestha2006]
	# LOCAL NEIGHBORHOOD
	# instances += [("fuzzy-cmeans", lambda: ClusterUncertainty(dtnn, method="cmeans", coverage=cp, n_jobs=n_jobs))]
	# instances += [("kmeans", lambda: ClusterUncertainty(dtnn, method="kmeans", coverage=cp, n_jobs=n_jobs))]
	# mahalanobis distance to data set center [toplak2014]
	# instances += [("kmeans-mahalanobis", lambda: ClusterUncertainty(dtnn, method="kmeans-mahalanobis", coverage=cp, n_jobs=n_jobs))]
	# density based estimate [bosnic2008]
	# instances += [("gaussian-clusters", lambda: ClusterUncertainty(dtnn, method="density", coverage=cp, n_jobs=n_jobs))]
	# CONFIVE (Variance of Error) [briesemeister2012]
	# (self, network_shim, method="label_var", sets="va", keys="", neighbors=20, n_jobs=1, silent=False):
	# quantile will have to be provided at evaluation time.
	# it makes sense to generate quantiles when the desired coverage is known.
	instances += [("knn-mse", lambda: KNNUncertainty(dtnn, method="mse", sets='va', keys='Y', neighbors=8, n_jobs=n_jobs))]
	instances += [("knn-mseq", lambda: KNNUncertainty(dtnn, method="mse", sets='tr', keys='', neighbors=10, n_jobs=n_jobs))]
	instances += [("knn-dev", lambda: KNNUncertainty(dtnn, method="dev", sets='va', keys='Y', neighbors=716, n_jobs=n_jobs))]
	instances += [("knn-avgdist", lambda: KNNUncertainty(dtnn, method="avgdist", sets='tr', keys='Y', neighbors=34, n_jobs=n_jobs))]
	instances += [("knn-vary", lambda: KNNUncertainty(dtnn, method="label_var", sets='va', neighbors=2653,  n_jobs=n_jobs))]
	# instances += [("knn-mahalanobis", lambda: KNNUncertainty(dtnn, method="mahalanobis", n_jobs=n_jobs))]
	# instances += [("knn-absolute-deviation-from-avg-label", lambda: KNNUncertainty(dtnn, method="absDev", n_jobs=n_jobs))]
	# instances += [("knn-mae", lambda: KNNUncertainty(dtnn, method="mae", n_jobs=n_jobs))]
	# instances += [("knn-rmse", lambda: KNNUncertainty(dtnn, method="rmse", n_jobs=n_jobs))]
		# avgDist [briesemeister2012]
	# instances += [("knn-avg-dist", lambda: KNNUncertainty(dtnn, method="avgDist", n_jobs=n_jobs))]

	# ENSEMBLES
	# bagging by example pairs and residuals [tibshirani1995]
	instances += [("ensemble-pairs", lambda: EnsembleUncertainty(DTNNShim, m=ensemble_members, method="pairs", balancing=False, folder=os.path.join(fp, "ensemble-pairs/"), gpus=gpus))]
	instances += [("ensemble-pairs-12", lambda: EnsembleUncertainty(DTNNShim, m=12, method="pairs", balancing=False, folder=os.path.join(fp, "ensemble-pairs-12/"), gpus=gpus))]
	instances += [("ensemble-pairs-6", lambda: EnsembleUncertainty(DTNNShim, m=6, method="pairs", balancing=False, folder=os.path.join(fp, "ensemble-pairs-6/"), gpus=gpus))]
	instances += [("ensemble-pairs-3", lambda: EnsembleUncertainty(DTNNShim, m=3, method="pairs", balancing=False, folder=os.path.join(fp, "ensemble-pairs-3/"), gpus=gpus))]
	instances += [("ensemble-residuals", lambda: EnsembleUncertainty(DTNNShim, m=ensemble_members, method="residuals", balancing=False, folder=os.path.join(fp, "ensemble-residuals/"), gpus=gpus))]
	instances += [("ensemble-residuals-12", lambda: EnsembleUncertainty(DTNNShim, m=12, method="residuals", balancing=False, folder=os.path.join(fp, "ensemble-residuals-12/"), gpus=gpus))]
	instances += [("ensemble-residuals-6", lambda: EnsembleUncertainty(DTNNShim, m=6, method="residuals", balancing=False, folder=os.path.join(fp, "ensemble-residuals-6/"), gpus=gpus))]
	instances += [("ensemble-residuals-3", lambda: EnsembleUncertainty(DTNNShim, m=3, method="residuals", balancing=False, folder=os.path.join(fp, "ensemble-residuals-3/"), gpus=gpus))]

	# balancing doesn't yield better estimates, unfortunately
	instances += [("ensemble-pairs-balanced", lambda: EnsembleUncertainty(DTNNShim, m=ensemble_members, k=ensemble_members//6, method="pairs", balancing=True, folder=os.path.join(fp, "ensemble-pairs-balanced/"), gpus=gpus))]
	instances += [("ensemble-residuals-balanced", lambda: EnsembleUncertainty(DTNNShim, m=ensemble_members, k=ensemble_members//6, method="residuals", balancing=True, folder=os.path.join(fp, "ensemble-residuals-balanced/"), gpus=gpus))]

	if only_est is not None:
		estimators = dict(instances)
		if only_est not in estimators.keys():
			raise ValueError("No such estimator instance: '{}'\nChoose from \n  {}".format(only_est, "\n  ".join(sorted(estimators.keys()))))
		else:
			return [(only_est, estimators[only_est])]
	return instances
