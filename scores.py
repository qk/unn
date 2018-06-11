import numpy as np
import scipy as sp
import scipy.stats
from utils import fit_shapes
from sklearn.metrics import f1_score, precision_score, recall_score


__all__ = [
	"rmse",
	"pearsonr",
	"mpiw",
	"coverage_percentage",
	"ace",
	"nmpiw",
	"rmpiw",
	"cwc",
	"cwc2",
	"cwc3",
	"winkler_score",
	"cec",
	"capi",
	"wilcoxon_rank_sum",
	"wilcoxon_sign_rank",
	"ncs",
	"f1_precision_recall",
]

def rmse(Y, prY):
	return ((Y-prY)**2).mean()**0.5

def pearsonr(Y, prY, prU):
	'''
	compute the pearson correlation coefficient between the absolute errors and 
	uncertainty scores
	returns correlation coefficient and p-value
	briesemeister2012 set the uncertainty to be the width between the 10% and 90% quantiles
	'''
	return sp.stats.pearsonr(np.abs(Y - prY), prU)

def mpiw(lower, upper):
	''' mean prediction interval width, see khosravi2011 (36) '''
	return np.abs(upper - lower).mean()

def coverage_percentage(Y, prY, lower, upper):
	'''
	khosravi2011 (34) calls this PICP (prediction interval coverage probability)
	but simply coverage percentage as in [pdfgrep for ref at some point] seems clearer
	'''
	return ((lower <= Y) & (Y <= upper)).mean()

def ace(Y, prY, lower, upper, coverage=0.95):
	'''
	average coverage error, see shrivastava2015 (10)
	there desired (or nominal) coverage = 1 - PINC = 1 - alpha
	I've changed it to mu for consistency with the other methods
	'''
	cp = coverage_percentage(Y, prY, lower, upper)
	return cp - coverage

def nmpiw(Y, lower, upper):
	'''
	normalized (to label range) mean prediction interval width, see khosravi2011 (37)
	calculation of label_range is not specified in khosravi
	  shrivastava2015 (11) refer to it as (maximum - minimum)
	  pevec2014 say its the abs. difference of the 95% and 5% quantiles of the labels
	'''
	label_range = Y.max() - Y.min()
	# label_range = 2*(3*Y.std()) # should account for 99.7% of data
	return np.abs(upper - lower).mean()/label_range

def rmpiw(Y, prY, lower, upper, eps=1e-1):
	'''
	relative mean prediction interval width, see pereira2014 (RMPIL, section E)
	gives interval width normalized by observed error
	because if the error is large, the intervals should be large as well
	problematic if the absolute error is 0
	'''
	return ((upper - lower)/(np.abs(Y - prY)+eps)).mean()

def cwc(Y, prY, lower, upper, eta=1.0, coverage=0.95):
	'''
	coverage width based criterion, see khosravi2011 (38)
	if the coverage percentage is smaller than the desired level mu(=1-alpha), only report the normed prediction interval width
	otherwise penalize the violation of the coverage percentage smoothly
	khosravi2011 set eta=50, coverage=0.90
	shrivastava2015 (12) say eta should be in [10,100]
	'''
	cp = coverage_percentage(Y, prY, lower, upper)
	gamma = float(cp < coverage)
	nw = nmpiw(Y, lower, upper)
	return nw*(1.0 + gamma*np.exp(-eta*(cp - coverage)))

def cwc2(Y, prY, lower, upper, coverage=0.95):
	'''
	coverage width based criterion revision, see pereira2014 (CLC2, section E)
	removes control parameter eta
	doesn't penalize slightly underestimated coverage percentages as harshly,
	because pereira et al. find the original cwc prefers models with large intervals to models,
	where the intervals are narrow but coverage percentage is slightly underestimated
	'''
	rw = rmpiw(Y, prY, lower, upper)
	cp = coverage_percentage(Y, prY, lower, upper)
	# this measure seems inconsistent
	# if cp=coverage it gives exp(0) = 1.0, regardless of interval width (not optimal but ok)
	# if cp>coverage it gives exp(-rw*some_factor)
	#	for rw, lower is better, so ideally you'd get values closer to exp(0)
	#	and if cp ~= coverage, which is desirable, you'll get even closer to exp(0)
	#	so for cp>coverage, cwc2 is in [0,1) and larger is better
	# if cp<coverage it gives exp(rw*some_factor)
	#	so for cp<coverage, cwc2 is in (1, infty] and lower is better
	return np.exp(-rw*(cp - coverage))

def cwc3(Y, prY, lower, upper, eta=1.0, coverage=0.95):
	'''
	coverage width based criterion, as revised by shrivastava2015 (13)
	they criticize that in the original cwc,
	if mean interval width becomes 0, the entire cwc term becomes 0 irrespective of cp.
	hence they change the multiplication to an addition.
	'''
	cp = coverage_percentage(Y, prY, lower, upper)
	gamma = float(cp < coverage)
	nw = nmpiw(Y, lower, upper)
	return nw + gamma*np.exp(-eta*(cp - coverage))

# def winkler_score(Y, prY, lower, upper, alpha=0.05):
	# '''
	# as referenced in shrivastava2015 (15)
	# an interval is considered sharp if the winkler score has lower absolute value for
	# the desired confidence level
	# '''
	# theta = upper - lower
	# too_lo = Y < lower
	# too_hi = Y > upper
	# inside = ~(too_lo | too_hi)
	# # could be calculated 2.6x faster, but at precision and/or clarity loss
	# S = np.concatenate((
			# (-2*alpha*theta[inside]),
			# (-2*alpha*theta[too_lo] - 4*(lower[too_lo] - Y[too_lo])),
			# (-2*alpha*theta[too_hi] - 4*(Y[too_hi] - upper[too_hi]))
		# )).mean()
	# return S

def winkler_score(Y, prY, lower, upper, coverage=0.95):
	'''
	as defined by Gneiting and Raftery (2017) and Liu et al. (2017)
	an interval is considered sharp if the winkler score has lower absolute value 
	for the desired confidence level
	'''
	alpha = 1.0 - coverage
	width = upper - lower
	too_low = Y < lower
	too_high = Y > upper
	inside = ~(too_low | too_high)
	S = width.sum() + 2/alpha*((lower[too_low] - Y[too_low]).sum() + (Y[too_high] - upper[too_high]).sum())
	return S/len(Y)

def cec(Y, prY, prU):
	'''
	returns the pearson correlation normalized by the correlation obtained by a perfect
	confidence estimator, see briesemeister2012 (section "Evaluation").
	this measure doesn't factor in wether the error is within interval bounds.
	their motivation was to have the absolute error be smaller than the interval width.
	they set the confidence to be the width between the 10% and 90% prediction quantiles
	(aka an 80% confidence interval)
	'''
	Y, prY, prU = (np.squeeze(A) for A in (Y, prY, prU))
	abs_error = np.abs(Y - prY)
	r_actual, p_actual = sp.stats.pearsonr(abs_error, prU)
	r_perfect, p_perfect = sp.stats.pearsonr(np.sort(prU, axis=0), np.sort(abs_error, axis=0))
	# print(r_actual, p_actual)
	# print(r_perfect, p_perfect)
	if p_actual > 0.05 or p_perfect > 0.05:
		print("p values exceed significance threshold:", p_actual, p_perfect, "(actual, perfect)")
	return r_actual/r_perfect

def capi(Y, prY, prU, ptop=0.2):
	'''
	confidence associated prediction improvement, see briesemeister2012 (section "Evaluation")
	measures by what percentage the MSE is reduced if we consider only the ptop % predictions
	with the smallest confidence intervals
	'''
	Y, prY, prU = fit_shapes(Y, prY, prU)
	se = (Y - prY)**2
	ntop = int(np.ceil(len(Y)*ptop))
	top = np.argsort(prU)[:ntop]
	return 1 - se[top].mean()/se.mean()

def wilcoxon_rank_sum(x, y):
	'''
	see matsumoto2016 below (13)
	null hypothesis:
	  values in the paired vector variables x and y
	  come from independent random samples from continuous distributions with equal means
	  with 5% significance level
	'''
	x, y = fit_shapes(x, y)
	statistic, p_value = sp.stats.ranksums(x, y)
	if p_value > 0.05:
		print("p_value exceeds significance threshold:", p_value)
	return statistic, p_value

def wilcoxon_sign_rank(x, y):
	'''
	see matsumoto2016 below (13)
	null hypothesis:
	  values in the vector variable (x-y) come from a population with continuous distribution
	  with mean equal to 0
	  with 5% significance level
	'''
	x, y = fit_shapes(x, y)
	statistic, p_value = sp.stats.wilcoxon(x, y=y)
	if p_value > 0.05:
		print("p_value exceeds significance threshold:", p_value)
	return statistic, p_value

def ncs(u, refs):
	"""
	normalized confidence score (briesemeister2012)
	confidence is antiproportional to uncertainty!
	(briesemeister2012 use the tr set for reference, but using the validation set should be more meaningful)
	maybe useful to compare confidences of different methods for the same point or region

	implemented inversely to adapt to uncertainty scores.
	(ncs == 0.8) => 80% of references have higher uncertainty
	(higher is better)

	parameters
	  u: (float or flat ndarray) uncertainty to check
	  refs: (flat ndarray) reference uncertainties to check against
	returns
	  percent of reference points with uncertainty >= u
	  it's like 1-cdf_refs(u)
	"""
	# TODO: check if using this measure deteriorates the quality of the uncertainty estimate.
	# i could imagine a strong correlation between error and uncertainty to be a necessary criterion for this measure to be usable.
	if not isinstance(u, np.ndarray):
		u = np.array(u)
	p = (refs[:,None] >= u).mean(axis=0)
	if len(np.shape(p)) == 1 and np.shape(p)[0] == 1:
		return p[0]
	return p

# refs = knnu.predict(vaD)
# U = refs[2] - refs[1]
# test = knnu.predict(teD)
# teU = test[2] - test[1]
# print(ncs(teU[0], U))
# print(ncs(teU[1], U))
# ncs(teU, U)

def f1_precision_recall(absR, prU, delta_err=0.1, delta_u=0.1):
	# f1 measure for identifying high error points
	# given thresholds delta_y and delta_u for abs. error and uncertainty
	absR, prU = fit_shapes(absR, prU)
	Y = (absR > delta_err)
	prY = (prU > delta_u)
	f1 = f1_score(Y, prY)
	recall = recall_score(Y, prY)
	precision = precision_score(Y, prY)
	# print(classification_report(Y, prY))
	# print(f1)
	return f1, precision, recall
