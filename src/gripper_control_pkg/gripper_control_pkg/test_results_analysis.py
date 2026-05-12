import math

n = 32
succes = 24

mean = succes/n

Z = 0.95

print(f"mean = {mean}")
print("")

print("Wald interval:")
print(f"n*mean = {n*mean}")
print(f"n*(1-mean) = {n*(1-mean)}")
print(f"The population mean is {mean}±{Z*math.sqrt( (mean*(1-mean)) / n ):.3f} with a {Z*100}% certainty")
print("")



from scipy.stats import beta
import numpy as np

k = 24
n = 32
alpha = 0.05
p_u, p_o = beta.ppf([alpha / 2, 1 - alpha / 2], [k, k + 1], [n - k + 1, n - k])
if np.isnan(p_o):
    p_o = 1
if np.isnan(p_u):
    p_u = 0

print("Clopper-Pearson interval:")
print(f"mean is {p_u:.4f} < {mean} < {p_o:.2f} witn a {(1-alpha)*100} percent confidence")



import scipy

print("Binomial test")
print(f"{scipy.stats.binomtest(24, 32, 1.0/2, alternative="greater")}")