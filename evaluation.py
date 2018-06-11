import numpy as np
import scipy as sp
import scipy.stats
import os
import re
import time
from utils import Frame
from uncertainty_estimator_instances import create_instances
from scores import *
from plots import *
from matplotlib import pyplot as pp
from data_loader import prepare_data
# commandline
import sys, subprocess
from argparse import ArgumentParser
import sys, subprocess
# plot
from matplotlib import pyplot as pp
from acronyms import translate

# def evaluate_all(uests, *args, **keywords):
	# ''' evaluates all uncertainty estimators in list uests '''
	# predict_kws = dict(
		# (k,keywords[k])
		# for k in "data calibrate coverage pretrained".split()
	# )
	# score_kws = dict(
		# (k,keywords[k])
		# for k in "coverage refs".split()
	# )
	# failed = []
	# scores = []
	# for name,est in uests:
		# try:
			# pr = predict(est, **predict_kws)
		# except Exception as e:
			# failed += ["predict", name, e]

		# try:
			# scores += [(name, score(pr, **score_kws))]
		# except Exception as e:
			# failed += ["score", name, e]

		# try:
			# plot(pr, os.path.join(data, plots, name))
		# except Exception as e:
			# failed += ["plot", name, e]

	# print(failed) # just to be save
	# print("\n".join(("{} {} {}".format(*line) for line in failed)))
	# return scores

def predict(est, data_dir=None, calibrate=False, coverage=0.95, pretrained=False, savename=None):
	''' predicts  '''
	print(est, "(loaded estimator)")
	print("loading datasets")
	if not isinstance(data_dir, str): # directory containing tr, va, te{i} files
		raise ValueError("data must a path to directory containing tr.npz va.npz te.npz files")

	# load data from file
	sets = dict()
	for name in "tr va te de special".split(): 
		filename = os.path.join(data_dir, "{}.npz".format(name))
		if os.path.exists(filename):
			print("loading file", filename)
			sets[name] = Frame(**np.load(filename))

	# fit
	if pretrained:
		print("pretrained model, training skipped")
	else:
		print("fitting...")
		tstart = time.time()
		est.fit(sets['tr'], sets['va'])
		mins = (time.time() - tstart)/60
		print("fit in {}h{}m".format(int(mins/60),int(mins%60)))
		if calibrate:
			prY, lo, hi = est.predict(sets['va'], coverage=coverage)
			factor = est.calibrate(sets['va']['T'], prY, (hi-lo)/2) # factor is a global scaling factor, but pointwise scaling will be used instead

	# predict
	print("predicting and saving...")
	has_index = 'index' in sets['tr'] and 'index' in sets['va']
	# predict and save reference sets
	prs = dict()
	for dname,d in sets.items():
		print("predicting", dname.upper(), "for coverage", coverage)
		prY, minY, maxY = est.predict(d, coverage=coverage)
		pr = dict(T=d["T"], Y=prY, minY=minY, maxY=maxY)
		print(((pr['T'] - pr['Y'])**2).mean()**0.5, 'rmse')
		prs[dname] = Frame(pr)
		if has_index: pr["index"] = d["index"].copy()
		path = os.path.join(data_dir, savename)
		os.makedirs(path, exist_ok=True)
		np.savez(os.path.join(path, "pr_{}.npz".format(dname)), **pr)
		if calibrate: # save calibrated predictions, if desired
			prY, minY, maxY = est.scale(pr['T'], pr['Y'], pr['minY'], pr['maxY'], coverage=coverage)
			pr = dict(pr.items(), Y=prY, minY=minY, maxY=maxY)
			dname_cp = "{}_{:02d}cp".format(dname, int(100*coverage))
			prs[dname_cp] = Frame(pr)
			np.savez(os.path.join(path, "pr_{}.npz".format(dname_cp)), **pr)

	print("predict done")
	return prs

def score(pr, coverage=0.95, refs=()):
	# load reference uncertainties (tr and va sets)
	if len(refs) > 0:
		refU = []
		for ref in refs: # load reference data
			if isinstance(ref, str): ref = np.load(ref)
			refU += [np.abs(ref["maxY"] - ref["minY"])/2]
		refU = np.concatenate(refU)
	# load actual testset predictions
	if isinstance(pr, str): pr = np.load(pr)
	Y, prY, lo, hi = [pr[k] for k in "T Y minY maxY".split()]
	prU = np.abs(hi-lo)/2 # assuming minY=Y-U, maxY=Y+U
	R = Y - prY # residuals
	absR = np.abs(R)

	# print("recheck matsumoto2016 if wilcoxon tests are used correctly")
	f1, prec, recall = f1_precision_recall(absR, prU, delta_err=0.1, delta_u=0.1)
	pearson_corr_coeff, pearson_pvalue = pearsonr(Y, prY, prU)
	results = dict(
		rmse=rmse(Y, prY),
		pearson=pearson_corr_coeff,
		pearson_p = pearson_pvalue,
		mpiw=mpiw(lo, hi),
		nmpiw=nmpiw(Y, lo, hi),
		rmpiw=rmpiw(Y, prY, lo, hi),
		coverage=coverage_percentage(Y, prY, lo, hi),
		ace=ace(Y, prY, lo, hi, coverage=coverage),
		cwc=cwc(Y, prY, lo, hi, eta=1.0, coverage=coverage),
		cwc2=cwc2(Y, prY, lo, hi, coverage=coverage),
		cwc3=cwc3(Y, prY, lo, hi, eta=1.0, coverage=coverage),
		winkler_score=winkler_score(Y, prY, lo, hi, coverage=coverage),
		cec=cec(Y, prY, prU),
		capi=capi(Y, prY, prU, ptop=0.2),
		# wilcoxon_rank_sum=wilcoxon_rank_sum(absR, prU),
		# wilcoxon_sign_rank=wilcoxon_sign_rank(absR, prU),
		f1=f1,
		precision=prec,
		recall=recall,
	)
	if len(refs) > 0:
		results['ncs'] = ncs(prU, refU)

	return results

def scoretable(scores, decimals=4, est_keys=None, score_keys=None):
	actual_est_keys = list(scores.keys())
	if est_keys is None:
		est_keys = actual_est_keys
	else:
		if not all([e in actual_est_keys for e in est_keys]):
			print("est_keys {} missing in dictionary, ignored.".format(
				[e for e in est_keys if e not in actual_est_keys]
			))
		est_keys = [e for e in est_keys if e in actual_est_keys]
	actual_score_keys = list(sorted(k for k in scores[est_keys[0]].keys()))
	if score_keys is None:
		score_keys = actual_score_keys
	else:
		if not all([k in actual_score_keys for k in score_keys]):
			print("est_keys {} missing in dictionary, ignored.".format(
				[k for k in score_keys if k not in actual_score_keys]
			))
		score_keys = [k for k in score_keys if k in actual_score_keys]
	# header
	T = [[""] + est_keys]
	lower_is_better = "rmse mpiw nmpiw rmpiw ace cwc cwc2 cwc3 winkler_score pearson_p".split()
	# body
	for k in score_keys:
		line = [k]
		values = np.array([scores[e][k] for e in est_keys])
		ismax = values == np.nanmax(values)
		ismin = np.abs(values) == np.nanmin(np.abs(values)) # lower ~= closer to zero
		for i,score in enumerate(values):
			score = np.round(score, decimals)
			score = "{: .{prec}f}".format(score, prec=decimals)
			score = re.sub(r'\.?0*$', '', score)
			is_loss = k in lower_is_better
			if k != 'coverage' and ((is_loss and ismin[i]) or (not is_loss and ismax[i])):
					score = "\\bfseries {}".format(score)
			else:
				score = "          {}".format(score)
			line += [score]
		T += [line]
	return T
	
def table_transpose(table):
	rows, cols = len(table), len(table[0])
	return [[table[r][c] for r in range(rows)] for c in range(cols)]

def table2tex(table, translate=translate):
	rows, cols = len(table), len(table[0])
	if translate is not None:
		dont_translate = ['rmse', 'mae']
		to_latex = lambda title: translate(title, latex=True) if title.lower() not in dont_translate else title.upper()
		table = [
			[to_latex(entry) for entry in row]
			for row in table
		]
	colwidths = [max(len(str(table[r][c])) for r in range(rows)) for c in range(cols)]
	table[0] = [
		# empty \thead{}'s cause "misplaced \omit" errors
		# "\\thead{{{}}}".format("\\\\ ".join(entry.split())) if len(entry)>0 else ""
		"\\thead{{{}}}".format(entry) if len(entry)>0 else ""
		for entry in table[0]
	]
	lines = [" & ".join(
			"{:{align}{width}s}".format(
				str(s),
				align=">" if i == 0 else "<",
				width=w
			) for i,w,s in zip(range(len(line)),colwidths,line)
		) for line in table
	]
	alignment = "\n".join(["r"] + ["S[table-format=-1.4]" for _ in range(cols-1)])
	lines = [line + "\\\\" for line in lines]
	lines[0] += " \\midrule"
	# lines = [line + "\n" for line in lines]
	text = "\n".join(lines)
	# text = "".join(lines)
	text = re.subn(r'\\', r'\\\\', text)[0]
	with open("tex/_template_table.tex", "r") as templatef:
		template = templatef.read()
	template = re.sub(r'%cols', alignment, template)
	template = re.sub(r'%table', text, template)
	return template

def scores2tex(scores, decimals=4, transpose=False):
	'''
	scores -- list of (estimator_name,scores_dict) pairs
	'''
	table = scoretable(scores, decimals=decimals)
	if transpose:
		table = table_transpose(table)
	return table2tex(table)

def _save(folder, basename, exts=("pdf",)):
	''' helper for plot() function '''
	name = os.path.join(folder, basename)

	# save folder/basename.ext
	for ext in exts:
		pp.savefig('{}.{}'.format(name, ext), bbox_inches='tight', dpi=300)

	# save folder/preview/basename.png
	preview_folder = os.path.join(folder, "preview")
	os.makedirs(preview_folder, exist_ok=True)
	name = os.path.join(preview_folder, basename)
	pp.savefig("{}.png".format(name), bbox_inches='tight', dpi=100)

def plot(pr, folder):
	'''
	generate all plots in {folder}/{plotname}.{ext} and additionally in 
	{folder}/{preview}/{plotname}.png for a quick preview
	'''
	# load actual testset predictions
	if isinstance(pr, str): pr = np.load(pr)
	Y, prY, lo, hi = [pr[k] for k in "T Y minY maxY".split()]
	prU = np.abs(hi-lo)/2 # assuming minY=Y-U, maxY=Y+U
	R = Y - prY # residuals
	absR = np.abs(R)

	os.makedirs(folder, exist_ok=True)

	# plots need different figure sizes, so this'll be somewhat unwieldy
	pp.figure(figsize=(4,4))
	plot_error_over_uncertainty_cdf(absR, prU, which="mae") # rmse gives same minimum
	_save(folder, 'mae_over_uncertainty_cdf')

	pp.figure(figsize=(4,4))
	plot_abs_error_over_interval_width(Y, prY, prU)
	_save(folder, 'abs_error_over_interval_width')

	# not implemented
	# pp.figure(figsize=(4,4))
	# plot_prediction_error_over_normalized_confidence_score()
	# _save(folder, 'prediction_error_over_normalized_confidence_score')

	pp.figure(figsize=(4,4))
	plot_coverage_alignment(Y, prY, prU, ppf=sp.stats.norm.ppf, max_iter=20)
	_save(folder, 'coverage_alignment')

	pp.figure(figsize=(12,4))
	plot_region(Y, prY, lo, hi, start=0, length=400, step=1)
	_save(folder, 'region')

	print("unimplemented: error precentile regions, requires subsampling check")
	# pp.figure(figsize=(4,4))
	# plot_error_percentile_regions(Y, prY, percentiles=[0.05, 0.50, 0.95], measure="rmse", offset=0)
	# _save(folder, 'plot_error_percentile_regions')

	pp.figure(figsize=(4,4))
	plot_rmse_over_thresholds(Y, prY, prU, force=False)
	_save(folder, 'rmse_over_thresholds')

	pp.figure(figsize=(4,4))
	plot_coverage_over_thresholds(Y, prY, lo, hi, force=False)
	_save(folder, 'coverage_over_thresholds')

	pp.figure(figsize=(12,4))
	plot_sorted_confidence_error(Y, prY, lo, hi, zoom=True, center='U')
	_save(folder, 'sorted_confidence_error')

	# plot_text(text, loc="nw"):
	# best_threshold(absE, prU, delta_err=1.0, p_budget=1.0, p_err=0.2, p_fn=0.0, start=1, method="svm", plots=True):
	# plot_training(filename, loss='loss', desc=True):
	return

if __name__ == "__main__":
	parser = ArgumentParser(description='interface for the predict method. trains the specified uncertainty estimator and generates predictions for the testset. invalid estimator-ids will yield an error message that includes an autoupdated list of available estimators. will save models as ./trained_models/{modelname}/session, where modelname usually is "{dataset}/{estimator_id}". current working directory should be that of this file (evaluation.py).')
	parser.add_argument(
		'estimator', type=str,
		help='uncertainty estimator id'
	)
	parser.add_argument(
		'-d', dest='data_dir', type=str,
		help='data directory'
	)
	# parser.add_argument(
		# '-c', dest='calibrate', action='store_const',
		# const=True, default=False,
		# help='wether to calibrate the interval width'
	# )
	parser.add_argument(
		'-p', dest='coverage', type=float, default=0.95,
		help='desired coverage percentage for estimators that support it, and for interval width scaling. translates negative values to None.'
	)
	parser.add_argument(
		'-m', "--mu-max", dest='mu_max', type=int, default=10,
		help='mu_max for gaussian grid expansion of interatomic distances. lower numbers result in better gpu utilization. set to 10 for benzene, 15 for everything else.'
	)
	parser.add_argument(
		'-s', dest='savename', type=str,
		help='predictions will be saved as {data_dir}/pr_{savename}.npz. savename defaults to the estimator-id.'
	)
	parser.add_argument(
		'-j', dest='n_jobs', type=int, default=1,
		help='number of parallel jobs (doesn\'t apply to all estimators))'
	)
	parser.add_argument(
		'-g', dest='gpu', type=int,
		help='gpu to train on'
	)
	parser.add_argument(
		'--ens-gpu', dest='gpus', nargs="+", type=int,
		help='how many ensemble instances to allocate to which gpu'
	)
	parser.add_argument(
		'--pretrained', dest='pretrained', action="store_const",
		const=True, default=False,
		help='skip training and load model from ./trained_models/{savename}/session'
	)
	parser.add_argument(
		'--debug', dest='debug', action="store_const",
		const=True, default=False,
		help='wether to drop into debugger on exception'
	)
	args = vars(parser.parse_args())
	if args['gpu'] is None:
		est_id = args.pop("estimator")
		# create_instances(folder_prefix="benzene/", n_jobs=1, only_est=None):
		ci_args = dict(folder_prefix=args['data_dir'])
		for a in "data_dir n_jobs gpus".split():
			if args[a] is not None:
				ci_args[a] = args[a]
		name, est = create_instances(only_est=est_id, **ci_args)[0]
		# predict(est, data=None, calibrate=False, coverage=0.95, savename=None):
		predict_args = dict()
		for a in "data_dir coverage pretrained savename".split():
			if args[a] is not None:
				predict_args[a] = args[a]
		# set default savename
		if "savename" not in predict_args:
			predict_args["savename"] = est_id
		if predict_args["coverage"] <= 0:
			predict_args["coverage"] = None
		if not args['debug']:
			predict(est(), **predict_args)
		else: 
			try:
				predict(est(), **predict_args)
			except:
				import traceback, pdb
				traceback.print_exc()
				pdb.post_mortem()
		print("all done")
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
