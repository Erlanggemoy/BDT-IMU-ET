from functools import lru_cache
import numpy as np

@lru_cache(maxsize=1)
def _model():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    
    degree = 2
    poly_features = PolynomialFeatures(degree=degree)
            
    x_time      = np.array([ 9, 10, 11, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50]).reshape(-1, 1)
    y_age       = np.array([20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85])
    
    x_time_poly = poly_features.fit_transform(x_time)
    model_time = LinearRegression()
    model_time.fit(x_time_poly, y_age)
    return poly_features, model_time

def estimate_cognitive_age(seconds):
    poly_features, model_time = _model()
    
    new_x_time = np.array([seconds]).reshape(-1, 1)
    new_x_time_poly = poly_features.transform(new_x_time)
    cognitive_age = model_time.predict(new_x_time_poly)
    cognitive_age = model_time.predict(new_x_time_poly)
    cognitive_age = float(np.asarray(cognitive_age).item())   # ambil scalar sekali di sini
    
    if cognitive_age < 20:
        cognitive_age = 20
    elif cognitive_age > 85:
        cognitive_age = 90
            
    cognitive_age = int(cognitive_age)
    
    return cognitive_age

def average_cognitive_age(ages):
    return int(sum(ages) / len(ages))

def cognitive_fitness(age_cog, age_real):
    if age_cog <= age_real:
        return 100
    return 100 - (age_cog - age_real)