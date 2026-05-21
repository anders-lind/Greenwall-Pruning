import math
import scipy.stats as stats

n = 16
succes = 16
alpha = 0.05

k = succes
Z = stats.norm.ppf(1 - alpha / 2)
sample_mean = succes/n

print(f"Sample mean = {sample_mean:.2f}")
print("")



### Wald Interval ###
import math

print("Wald interval:")

margin = (Z/math.sqrt(n) )* math.sqrt((sample_mean*(1-sample_mean)))
print(f"The population mean is {sample_mean-margin:.2f} < {sample_mean:.2f} < {sample_mean+margin:.2f} with a {(1-alpha)*100:.2f}% confidence")
print("")



### Adjusted Wald Interval ###
print("Adjusted Wald interval:")

adjusted_mean = (succes+2)/(n+4)
adjusted_n = n+4

# print(f"n*mean = {n*mean}")
# print(f"n*(1-mean) = {n*(1-mean)}")
margin = (Z/math.sqrt(adjusted_n) )* math.sqrt((adjusted_mean*(1-adjusted_mean)))
print(f"The population mean is {adjusted_mean-margin:.2f} < {adjusted_mean:.2f} < {adjusted_mean+margin:.2f} with a {(1-alpha)*100:.2f}% confidence")
print("")



### Clopper-Pearson interval ###
from scipy.stats import beta
import numpy as np


p_u, p_o = beta.ppf([alpha / 2, 1 - alpha / 2], [k, k + 1], [n - k + 1, n - k])
if np.isnan(p_o):
    p_o = 1
if np.isnan(p_u):
    p_u = 0

print("Clopper-Pearson interval:")
print(f"The population mean is {p_u:.2} < {sample_mean:.2f} < {p_o:.2f} witn a {(1-alpha)*100:.2f}% confidence")
print("")


### Wilson Score interval ###
# Website version
mean = (sample_mean+(Z*Z)/(2*n)) / (1+(Z*Z)/n)
margin = Z / (1+(Z*Z)/n) * math.sqrt( (sample_mean*(1-sample_mean)) / n + (Z*Z)/(4*(n*n)) )

# Source 1 version
# mean = sample_mean+((Z*Z)/(2*n))
# margin = Z * math.sqrt( ((sample_mean*(1-sample_mean)) + (Z*Z)/(4*n))/n ) / (1+Z*Z/n)

# source 2 version
lower = (sample_mean + (Z*Z)/(2*n) - Z*math.sqrt( (sample_mean*(1-sample_mean)) / (n) + (Z*Z)/(4*n*n) ) ) / (1 + (Z*Z)/n)
upper = (sample_mean + (Z*Z)/(2*n) + Z*math.sqrt( (sample_mean*(1-sample_mean)) / (n) + (Z*Z)/(4*n*n) ) ) / (1 + (Z*Z)/n)

print()
print(f"lower = {lower:.2f}")
print(f"upper = {upper:.2f}")
print()


print(f"Z = {Z}")
print(f"mean = {mean}")

print("Wilson Score Interval:")
print(f"The population mean is {mean-margin:.2} < {mean:.2f} < {mean+margin:.2f} with a {(1-alpha)*100:.2f}% confidence")
print("")


### Scipy test ###
import scipy

print("Binomial test")
print(f"{scipy.stats.binomtest(24, 32, 1.0/2, alternative="greater")}")