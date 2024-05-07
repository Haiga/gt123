from scipy.stats import ttest_rel
import numpy as np

vec1 = [0, 1, 2, 3]*8
vec2 = [2, 1, 0, 1]*8
vec3 = [1.1, 1, 1, 1]*8

print(np.mean(vec1))
print(np.mean(vec2))
print(np.mean(vec3))

print(ttest_rel(vec1, vec2))
print(ttest_rel(vec1, vec3))
# print(ttest_rel(vec1, vec2))
