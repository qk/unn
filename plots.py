import numpy as np
from matplotlib import pyplot as pp
import time
from utils import fit_shapes, ecdf
from acronyms import translate
# best_threshold
from sklearn.metrics import classification_report, f1_score, auc, precision_score, precision_recall_curve, recall_score, accuracy_score
from sklearn.svm import LinearSVC

__all__ = [
	'plot_error_over_uncertainty_cdf',
	'plot_text',
	'plot_abs_error_over_interval_width',
	'plot_prediction_error_over_normalized_confidence_score',
	'plot_coverage_alignment',
	'plot_text',
	'plot_region',
	'plot_error_percentile_regions',
	'plot_convolved_error',
	'plot_rmse_over_thresholds',
	'plot_coverage_over_thresholds',
	'plot_sorted_confidence_error',
	'plot_training',
	'best_threshold',
	'plot_abs_error_over_uncertainty',
	'plot_error_over_confident_instances',
]


def best_threshold(absE, prU, delta_err=1.0, p_budget=1.0, p_err=0.2, p_fn=0.0, start=1, method="svm", plots=True):
	E, prU = fit_shapes(absE, prU)
	Y = (E > delta_err)
	pos = Y.sum()
	neg = len(Y)-pos
	N = len(Y)
	
	defined_methods = "svm error_fraction min_mae min_rmse fn".split(" ")
	if method not in defined_methods:
		raise ValueError("method '%s' is not defined. Must be one of %s" % (method, str(defined_methods)))
	else:
		print("using method", method, "with parameter", dict(zip(defined_methods, (p_budget, p_err, start, p_fn)))[method])
	
	if method == "svm":
		# True: high error cases (critical), False: low error cases
		est = LinearSVC(class_weight={True:N/pos, False:(N/neg)*p_budget})
		est.fit(prU[:,None], Y)
		prY = est.predict(prU[:,None])
		w,b = est.coef_[0], est.intercept_
		threshold = np.squeeze(-b/w) # w*x+b > 0 => positive label
		if plots:
			ax = pp.subplot2grid((1,2),(0,0))
			n = 200
			X = np.linspace(0.0, 1, 200000)
			Z = est.predict(X[:,None])
			ax.plot(X, Z)
			emp_thresh = X[Z.tolist().index(1)]
			pp.gca().set_xticks([emp_thresh], minor=True)
			pp.gca().set_xticklabels([str(emp_thresh)], rotation='vertical', minor=True)
			pp.grid(which='minor')
	
	elif method == "error_fraction":
		asc_uncertainty = np.argsort(prU)
		fractions = np.cumsum(E[asc_uncertainty])/E.sum()
		threshold = prU[asc_uncertainty][(fractions > p_err).argmax()]
		if plots:
			print(threshold, fractions[(fractions > p_err).argmax()])
			ax = pp.subplot2grid((1,2),(0,0))
			ax.plot(prU[asc_uncertainty], fractions, "k,")
			ax.set_xticks([threshold], minor=True)
			ax.set_xticklabels([str(threshold)], rotation='vertical', minor=True)
			pp.grid(which='minor')
			pp.xlabel("uncertainty")
			pp.ylabel("cumulative error [%]")
	
	elif method in "min_mae min_rmse".split(" "):
		# minimize mean error for "confident" points
		asc_uncertainty = np.argsort(prU)
		if method == "min_mae":
			meanE = np.cumsum(E[asc_uncertainty])/np.arange(1,len(Y)+1)
		else:
			meanE = (np.cumsum(E[asc_uncertainty]**2)/np.arange(1,len(Y)+1))**0.5
		imin = meanE[start:].argmin() + start
		threshold = prU[asc_uncertainty][imin]
		if plots:
			print(threshold, meanE[imin], imin)
			ax = pp.subplot2grid((1,2),(0,0))
			ax.semilogx(prU[asc_uncertainty], meanE, "k")
			ax.set_xticks([threshold], minor=True)
			ax.set_xticklabels([str(threshold)], rotation='vertical', minor=True)
			pp.grid(which='minor')
			pp.xlabel("uncertainty")
			pp.ylabel("%s" % str(method[4:]))
	
	elif method == "fn":
		asc_uncertainty = np.argsort(prU)
		fn = np.cumsum(Y[asc_uncertainty])/Y.sum() # high error (critical) cases
		threshold = prU[asc_uncertainty][(fn > p_fn).argmax()]
		if plots:
			print(threshold, fn[(fn > p_fn).argmax()])
			ax = pp.subplot2grid((1,2),(0,0))
			ax.plot(prU[asc_uncertainty], fn, "k,")
			ax.set_xticks([threshold], minor=True)
			ax.set_xticklabels([str(threshold)], rotation='vertical', minor=True)
			pp.grid(which='minor')
			pp.xlabel("uncertainty")
			pp.ylabel("cumulative fn's [% total fn's]")

	if plots:
		ax = pp.subplot2grid((1,2), (0,1))
		crit = (prU > threshold)
		tp = crit & Y
		fp = crit & ~Y
		tn = ~crit & ~Y
		fn = ~crit & Y
		precision = tp.sum() / crit.sum()
		recall = tp.sum() / Y.sum()
		print("precision", precision, "recall", recall)
		print("fn/pos", fn.sum()/Y.sum())
		ax.semilogx(prU[tp], E[tp], "k,")
		ax.semilogx(prU[tn], E[tn], "k,")
		ax.semilogx(prU[fp], E[fp], "b,", label="fp (blue)")
		ax.semilogx(prU[fn], E[fn], "r,", label="fn (red)")
		ax.set_xticks([threshold], minor=True)
		ax.set_xticklabels([str(threshold)], rotation='vertical', minor=True)
		pp.grid(which='minor')
		pp.xlabel("uncertainty")
		pp.ylabel("error")
		pp.legend()

	return threshold

# def plot_error_over_uncertainty_cdf(absE, prU, which="mae"):
	# # maybe similar to what briesemeister did
	# E, prU = fit_shapes(absE, prU)
	# asc_uncertainty = np.argsort(prU)
	# if which == "mae":
		# meanE = np.cumsum(E[asc_uncertainty])/np.arange(1,len(E)+1)
	# elif which == "rmse":
		# meanE = (np.cumsum(E[asc_uncertainty]**2)/np.arange(1,len(E)+1))**0.5
	# imin = meanE[1:].argmin() + 1
	# threshold = prU[asc_uncertainty][imin]

	# print(threshold, meanE[imin], imin)
	# cdf, inv_cdf = ecdf(prU)
	# ax = pp.gca()
	# # ax.semilogx(prU[asc_uncertainty], meanE, "k")
	# ax.semilogx(cdf(prU[asc_uncertainty]), meanE, "k,")
	# # ticks = np.array(ax.get_xticks())
	# # ticks = ticks[(ticks >= cdf(prU.min())) & (ticks <= cdf(prU.max()))]
	# # # ticks = np.linspace(cdf(prU.min()), cdf(prU.max()), 5)
	# # ax.set_xticks(ticks)
	# # ax.set_xticklabels(["%.4f"%t for t in inv_cdf(ticks)])

	# # ax.plot(cdf(prU[asc_uncertainty]), meanE, "k,")
	# # ax.plot(prU[asc_uncertainty], meanE, "k")
	# ax.set_xticks([cdf(threshold)], minor=True)
	# ax.set_xticklabels(["\n"+str(threshold)], minor=True)
	# pp.grid(which='minor')
	# pp.xlabel("uncertainty\n(x-axis scale: log(ecdf))")
	# pp.ylabel("%s" % str(which))
	# return


def plot_abs_error_over_uncertainty(Y, prY, prU):
	E = np.abs(Y-prY)
	cdf_E, _ = ecdf(E)
	cdf_U, _ = ecdf(prU)
	# convert to uniform distributions, then plot
	pp.plot(cdf_U(prU), cdf_E(E), "k,", alpha=0.05)


def plot_error_over_confident_instances(Y,prY,prU,which='mae'):
	I = np.argsort(prU)
	if which == 'mae':
		E = np.abs(Y-prY)[I]
		E = np.cumsum(E)/np.arange(1,len(E)+1)
	elif which == 'rmse':
		E = ((Y-prY)**2)[I]
		E = np.cumsum(E)/np.arange(1,len(E)+1)
		E = E**0.5
	pp.plot(np.arange(len(E))/len(E), E, "k.", alpha=0.1, ms=1)
	pp.xlabel("Proportion of Instances (Sorted by Uncertainty)")
	pp.ylabel(translate(which, latex=False))


def plot_error_over_uncertainty_cdf(absE, prU, which="mae", skip_first=0.01):
	# maybe similar to what briesemeister did
	E, prU = fit_shapes(absE, prU)
	asc_uncertainty = np.argsort(prU)
	if len(np.unique(prU)) < len(prU):
		print("WARNING: duplicate values in uncertainty")
	if which == "mae":
		meanE = np.cumsum(E[asc_uncertainty])/np.arange(1,len(E)+1)
	elif which == "rmse":
		meanE = (np.cumsum(E[asc_uncertainty]**2)/np.arange(1,len(E)+1))**0.5
	if isinstance(skip_first, float):
		skip_first = int(len(meanE) * skip_first)
	# 876543210 index reversed
	# 012345678 index
	# 123456789 count/length
	imin = meanE[skip_first:].argmin() + skip_first
	threshold = prU[asc_uncertainty][imin]

	print(threshold, meanE[imin], imin)
	cdf_x, cdf_y = ecdf(prU, interpolate=False) # cdf_x == prU[asc_uncertainty]
	ax = pp.gca()
	ax.semilogx(cdf_y, meanE, "k,")

	# mark minimum-error region for the uncertainty threshold
	a = (cdf_x >= threshold).argmax()
	b = (cdf_x > threshold).argmax()
	value = cdf_y[imin]
	if a == b:
		ax.axvline(value)
	else:
		ax.axvspan(cdf_y[a],cdf_y[b], alpha=0.4)
	plot_text("min error at U = {:3.2e}".format(meanE[imin]))

	pp.xlabel("logscaled cdf(uncertainty)")
	pp.ylabel("%s" % str(which))
	return

def plot_text(text, loc="nw"):
	pp.text(0.01,0.99, text, transform=pp.gca().transAxes, verticalalignment="top")
	return


def plot_abs_error_over_interval_width(Y, prY, prU):
	# used in briesemeister2012
	Y, prY, prU = fit_shapes(Y, prY, prU)
	abs_error = np.abs(Y - prY)
	asc_uncertainty = np.argsort(prU)
	pp.plot(prU[asc_uncertainty], abs_error[asc_uncertainty], 'k,', alpha=0.4)
	pp.xlabel("uncertainty")
	pp.ylabel("abs. error")


def plot_prediction_error_over_normalized_confidence_score():
	# used in briesemeister2012
	pass


# plot desired vs actual coverage
def plot_coverage_alignment(Y, prY, prU, ppf, max_iter=20):
	Y, prY, prU = (np.squeeze(A) for A in (Y, prY, prU))

	# calc average misalignment factor for target ppf
	R = np.abs(Y-prY) # abs. residuals
	cdf, eppf = ecdf(R/prU)
	_,p = ecdf(R/prU, interpolate=False)
	factor = (eppf(p)/ppf((1+p)/2)).mean()

	percent = np.linspace(0,1,100)[1:]
	zscores = factor*ppf((1+percent)/2)

	coverage = ((prY - zscores[:,None]*prU <= Y) & (Y <= prY + zscores[:,None]*prU)).mean(axis=1)
	se = ((percent - coverage)**2).mean()**0.5

	pp.plot(percent, coverage, "r", alpha=0.7)
	pp.plot(percent, percent, "k:", alpha=0.5)
	pp.xlim([0,1])
	pp.ylim([0,1])
	pp.axis('equal')
	pp.xlabel("Nominal Coverage Proportion")
	pp.ylabel("Actual Coverage Proportion")
	# plot_text("Alignment Factor %f\nAlignment RMSE %f" % (factor, se))
	plot_text("Alignment RMSE %f" % (se))

	return factor, se


def plot_region(Y, prY, lower, upper, start=0, length=400, step=1):
	end = start+length
	I = np.arange(len(Y))[start:end:step]
	Y, prY, lower, upper = (A[start:end:step] for A in fit_shapes(Y, prY, lower, upper))
	mean = Y.mean(axis=0)
	CW = upper-lower
	
	pp.plot(I, Y-mean, "k.", alpha=1, lw=0, markersize=2) # true function
	pp.plot(I, prY-mean, "r") # est. function
	pp.fill_between(I, lower-mean, upper-mean, color="red", alpha=0.2) # CI
	pp.ylim([lower.min() - mean - (upper.max()-lower.min())/2, upper.max()-mean])
	
	ax = pp.gca().twinx()
	miny = (Y - lower).min()
	ax.fill_between(I, [miny]*len(I), CW, color="skyblue", alpha=0.3, zorder=-1) # cw
	ax.plot(I, (Y-lower), "k.", markersize=2)
	ax.plot(I, prY-lower, "r", markersize=0.1)
	ax.set_ylim([miny, CW.max()*4])
	ax.set_ylabel("interval width", color="skyblue")
	ax.tick_params("y", colors="skyblue")


def plot_error_percentile_regions(Y, prY, percentiles=[0.05, 0.50, 0.95], measure="rmse", offset=0):
	Y, prY = fit_shapes(Y, prY)
	mu = Y.mean()
	Y, prY = Y - mu, prY - mu
	window = 1000
	print(window/len(Y), "fraction of data per plot")
	# could leave out ()/window and np.sqrt() in sum_error calcs, but it's fast so why not do it properly
	if measure == "mae":
		error = np.abs(Y-prY)
		sum_error = np.convolve(error, np.ones(window), 'same') / window
	elif measure == "rmse":
		error = (Y-prY)**2
		sum_error = np.sqrt(np.convolve(error, np.ones(window), 'same') / window)
	else:
		print("measure must be in ['mae','rmse']")
	I = np.argsort(sum_error)
	# idxs = [I[int(np.round(p * (len(Y)-1)))] for p in percentiles]
	idxs = [I[np.searchsorted(sum_error[I], np.percentile(sum_error, p*100))] for p in percentiles]
	I = np.arange(len(Y)) + offset

	for i,hint in enumerate(["%s-percentile"%str(100*p) for p in percentiles]):
		ax = pp.subplot2grid((4,1), (i,0))
		start, end = max(idxs[i]-window//2, 0), min(idxs[i]+window//2, len(Y))
		ax.plot(I[start:end], Y[start:end], "k:", alpha=0.5, lw=1)
		ax.plot(I[start:end], prY[start:end], "r", alpha=0.5, lw=1)
		ax.set_xlim([I[start], I[end]])
		if i == 0:
			ax.set_title("%s %s-Percentile %s"%(str([p*100 for p in percentiles]), measure.upper(), "Regions"))
		if i == 1:
			ax.set_ylabel("Centered Label")
		if i == 3:
			ax.set_xlabel("Point Indices")
		ax.text(0.005,0.99, hint, transform=pp.gca().transAxes, verticalalignment="top")
	ax = pp.subplot2grid((4,1), (3,0))
	ax.fill_between(I, [0]*len(I), sum_error, color="lightgrey")
	ax.set_ylabel("Convolved %s"%measure.upper())
	ax.set_xlim([offset, offset+len(I)])


def plot_convolved_error(Y, prY, measure="rmse", offset=0):
	Y, prY = fit_shapes(Y, prY)
	mu = Y.mean()
	Y, prY = Y - mu, prY - mu
	window = 1000
	print(window/len(Y), "fraction of data per plot")
	# could leave out ()/window and np.sqrt() in sum_error calcs, but it's fast so why not do it properly
	if measure == "mae":
		error = np.abs(Y-prY)
		sum_error = np.convolve(error, np.ones(window), 'same') / window
	elif measure == "rmse":
		error = (Y-prY)**2
		sum_error = np.sqrt(np.convolve(error, np.ones(window), 'same') / window)
	else:
		print("measure must be in ['mae','rmse']")
	I = np.arange(len(Y)) + offset

	ax = pp.gca()
	ax.fill_between(I, np.zeros_like(I), sum_error, color="lightgrey")
	ax.set_ylabel("Convolved %s"%measure.upper())
	ax.set_xlabel("Time Index")
	ax.set_xlim([offset, offset+len(I)])


def plot_rmse_over_thresholds(Y, prY, prC, force=False):
	# ideal plot:
	#   rmse rises as a step function when a certain threshold is overstepped.
	#   #confident_points rises early without increasing rmse, significantly increases when the same threshold as above is overstepped.
	# generally, it would be nice to have many points with low rmse.
	Y, prY, prC = fit_shapes(Y, prY, prC)
	# for A in (Y, prY, prC): print(A.shape)
	if max(len(A.shape) for A in (Y, prY, prC)) > 2:
		print("WARNING:   Y has shape", Y.shape)
		print("WARNING: prY has shape", prY.shape)
		print("WARNING: prC has shape", prC.shape)
		if not force: return
		else: print("use force=True to still calc the rmse, but it could take long")

	percent = np.linspace(0,1,1000)
	thresholds = percent*prC.max(axis=0)
	I = (prC[None,:] < thresholds[:,None])
	
	ax1 = pp.gca()
	rmses = np.sqrt(((Y*I - prY*I)**2).mean(axis=1))
	pp.plot(percent, rmses, 'r')
	ax1.set_ylabel("rmse", color='r')
	ax1.tick_params('y', colors='r')
	pp.xlabel("threshold [%/100]")
	ax1.text(0.005,0.99, "max rmse %.4f"%rmses[-1], transform=pp.gca().transAxes, verticalalignment="top")

	ax2 = ax1.twinx()
	ax2.plot(percent, I.sum(axis=1)/I.shape[1], "b", alpha=0.6)
	ax2.set_ylabel("confident points [%/100]", color='b')
	ax2.tick_params('y', colors='b')
	# confident points: fraction of points predicted with required amount of confidence. max confidence = all points


def plot_coverage_over_thresholds(Y, prY, lower, upper, force=False):
	# ideal plot:
	#   coverage rises early to expected coverage (for normal distr.: band widths 1, 2, 3 => 68, 95, 99.7 coverage).
	#   #confident_points rises early, significantly increases when some threshold is overstepped.
	# generally, it would be nice to have many confident points with high coverage.
	Y, prY, lower, upper = fit_shapes(Y, prY, lower, upper)
	
	# TODO: calculate band widths from expected_coverage parameter
	if max(len(A.shape) for A in (Y, prY, lower, upper)) > 2:
		print("WARNING:   Y has shape", Y.shape)
		print("WARNING: prY has shape", prY.shape)
		print("WARNING: prC has shape", prC.shape)
		if not force: return
		else: print("use force=True to still calc the rmse, but it could take long")

	percent = np.linspace(0,1,200)

	# coverage: number of true values inside thresholds

	# relative thresholds, depend on confidence interval per point
	# lower_th = percent[:,None]*(lower - prY)
	# upper_th = percent[:,None]*(upper - prY)
	# lower_I = ((Y - prY)[None,:] >= lower_th)
	# upper_I = ((Y - prY)[None,:] <= upper_th)
	# I = lower_I & upper_I
	# coverage = I.sum(axis=1)/len(Y)
	# print(I.shape)

	# absolute, fixed threshold value
	lower_th = percent[:,None]*((lower - prY).mean())
	upper_th = percent[:,None]*((upper - prY).mean())
	lower_I = (Y - prY)[None,:] >= lower_th
	upper_I = (Y - prY)[None,:] <= upper_th
	I = lower_I & upper_I
	coverage = I.sum(axis=1)/len(Y)
	print(I.shape)

	# confident points: number of points, whose intervals are inside desired thresholds
	lower_I = (lower - prY)[None,:] >= lower_th
	upper_I = (upper - prY)[None,:] <= upper_th
	I = lower_I & upper_I
	confident_points = I.sum(axis=1)/len(Y)
	print(I.shape)
	
	ax1 = pp.gca()
	pp.plot(percent, coverage, 'r')
	ax1.set_ylabel("coverage", color='r')
	ax1.tick_params('y', colors='r')
	pp.xlabel("threshold [%%/100 of (%4.2f, %4.2f)]"%(lower_th[-1], upper_th[-1]))
	ax1.text(0.005,0.99, "max coverage %.4f"%coverage[-1], transform=pp.gca().transAxes, verticalalignment="top")
	# coverage: fraction of true values within the confidence interval

	ax2 = ax1.twinx()
	ax2.plot(percent, confident_points, "b", alpha=0.6)
	ax2.set_ylabel("confident points [%/100]", color='b')
	ax2.tick_params('y', colors='b')
	# confident points: fraction of points predicted with required amount of confidence. max confidence = all points

	ax1.set_ylim([-0.1,1.1])
	ax2.set_ylim([-0.1,1.1])


def plot_sorted_confidence_error(Y, prY, lower, upper, zoom=True, center='U', alpha=0.05):
	# used by Meinshausen2006, briesemeister2012, pevec2014
	# ideal plot:
	#   all points within intervals, intervals wrap tightly around points.
	Y, prY, lower, upper = fit_shapes(Y, prY, lower, upper)

	CW = (upper-lower) # confidence interval width
	ascendingCW = np.argsort(CW)
	I = np.arange(len(CW))/len(CW)
	
	if center == 'U':
		horizon = (upper + lower)/2
	elif center == 'prY':
		horizon = prY
	else:
		raise ValueError("'center' must be 'U' or 'prY'")

	pp.fill_between(I, (lower-horizon)[ascendingCW], (upper-horizon)[ascendingCW], facecolor="skyblue", lw=0)
	pp.plot(I, (Y-horizon)[ascendingCW], "k.", alpha=alpha, ms=1)
	pp.plot(I, [0]*len(I), "w-", lw=1, alpha=0.5)
	pp.plot(I, (lower-horizon)[ascendingCW], "w-", lw=1, alpha=0.5)
	pp.plot(I, (upper-horizon)[ascendingCW], "w-", lw=1, alpha=0.5)

	inside = 100 * ((lower <= Y) & (Y <= upper)).mean()
	pp.text(0.005,0.99, "%.1f%% Coverage (True Value Inside Interval), %.1f Mean Interval Width" % (inside, CW.mean()), transform=pp.gca().transAxes, verticalalignment="top")
	pp.ylabel("Signed Error" if center == 'prY' else "Centered Interval Bounds")
	pp.xlabel("Proportion of Instances (Sorted by Interval Width)")
	pp.xticks(np.linspace(0,1,11))
	pp.xlim([0,I[-1]])
	# pp.grid(axis="y")

	if zoom:
		R = Y - horizon
		ymax = max(np.abs([R[R<0].std(), R[R>0].std()]))
		pp.ylim([-6*ymax, 6*ymax])
	return inside, CW.mean()

def plot_training(filename, loss='loss', desc=True):
	T = np.load(filename)
	if loss not in T.keys():
		raise ValueError('No entry for loss: "{}". Choose from {}'.format(loss, str(T.keys())))
	if desc: # extract the descending loss subsequence
		display_rows = []
		vmin = np.inf
		J = T[loss]
		for i in range(len(J)):
			v = J[i]
			if v < vmin:
				vmin = v
				display_rows.append(i)
		display_rows = np.array(display_rows)
		pp.plot(T["epoch"][display_rows], T[loss][display_rows])
	else:
		pp.plot(T["epoch"], T[loss])
	ymax = 0.1 * (T[loss].max() - T[loss].min()) + T[loss].min()
	pp.ylim([0, ymax])
	pp.grid()
	pp.ylabel(loss)
	pp.xlabel('epoch')
	pp.title('{}validation {}'.format("minimal " if desc else "", loss))
	return

class ScrollGraph:
	def __init__(self, window=50, step=1):
		self.fig, self.ax = pp.subplots(1,1)
		pp.tight_layout()
		self.window = window
		self.step = step
	
	def apply(self):
		self.ax.relim()
		self.ax.autoscale_view()
		self.fig.canvas.draw()
	
	def draw(self, i, x, fs, args, keywords):
		if self.ax.lines:
			pp.title(len(self.results[0].get_paths()))
			for coll in (self.ax.collections):
				self.ax.collections.remove(coll)
			for res,f,a,k in zip(self.results, fs, args, keywords):
				if f == "plot":
					res[0].set_xdata(x)
					res[0].set_ydata(a[0])
				elif f == "fill_between":
					self.ax.fill_between(x, *a, **k)
		else:
			self.results = []
			for f,a,k in zip(fs, args, keywords):
				#print(type(x), (type(ai) for ai in a), [kw for kw in k.keys()])
				self.results += [getattr(self.ax, f)(x, *a, **k)]
		self.apply()
	
	def plot(self, X, Y, prY, lower, upper, prYs, start=0):
		fs = ["fill_between", "plot", "plot"]
		args = [
			[],
			["r"],
			["k:"]
		]
		keywords = [
			{"facecolor":"skyblue", "label":"CI"},
			{"alpha":0.4, "lw":1.5},
			{"alpha":0.4, "lw":1.5},
		]
		for i in range(start, len(X) - self.window, self.step):
			end = i+self.window
			ys = [
				[lower[i:end], upper[i:end]],
				[prY[i:end]],
				[Y[i:end]],
			]
			self.draw(i, X[i:end], fs, [ysi+ai for ysi,ai in zip(ys, args)], keywords)
			time.sleep(0.1)
# scrollgraph = ScrollGraph(window=500, step=4)
# scrollgraph.plot(np.arange(len(Y)), Y-mu, prY*sigma, (prY-2*prC)*sigma, (prY+2*prC)*sigma, prYs, start=0)
