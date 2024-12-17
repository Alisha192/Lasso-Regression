import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error

# Генерация данных
np.random.seed(42)
n_samples = 100
X = np.sort(np.random.rand(n_samples) * 1)  # Входные данные
y = (X10 + X6 + 1) * (1 + np.random.randn(n_samples))  # Истинная функция с шумом

# Разделение данных на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Преобразование входных данных в формат [n_samples, 1] для sklearn
X_train = X_train[:, np.newaxis]
X_test = X_test[:, np.newaxis]

# Параметры для полиномиальных признаков
degree = 3  # Максимальная степень полинома

# 1. Обычная полиномиальная регрессия (без регуляризации)
poly_reg_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
poly_reg_model.fit(X_train, y_train)

# 2. Lasso-регрессия (с подбором оптимального alpha)
lasso_pipeline = make_pipeline(PolynomialFeatures(degree), Lasso(max_iter=10000))
lasso_param_grid = {'lasso__alpha': np.logspace(-4, 0, 50)}  # Значения alpha от 0.0001 до 1
lasso_grid_search = GridSearchCV(lasso_pipeline, lasso_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
lasso_grid_search.fit(X_train, y_train)
best_lasso_model = lasso_grid_search.best_estimator_
best_lasso_alpha = lasso_grid_search.best_params_['lasso__alpha']

# 3. Ridge-регрессия (с подбором оптимального alpha)
ridge_pipeline = make_pipeline(PolynomialFeatures(degree), Ridge(max_iter=10000))
ridge_param_grid = {'ridge__alpha': np.logspace(-4, 0, 50)}  # Значения alpha от 0.0001 до 1
ridge_grid_search = GridSearchCV(ridge_pipeline, ridge_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
ridge_grid_search.fit(X_train, y_train)
best_ridge_model = ridge_grid_search.best_estimator_
best_ridge_alpha = ridge_grid_search.best_params_['ridge__alpha']

# Функция для преобразования коэффициентов в строку полинома
def polynomial_to_string(coefs):
    terms = []
    for i, coef in enumerate(coefs):
        if np.isclose(coef, 0):
            continue
        if i == 0:
            terms.append(f"{coef:.3f}")
        elif i == 1:
            terms.append(f"{coef:.3f}x")
        else:
            terms.append(f"{coef:.3f}x^{i}")
    return " + ".join(terms)

# Вывод коэффициентов и полиномов
print("\nКоэффициенты и полиномы моделей:")

print("\nОбычная полиномиальная регрессия (без регуляризации):")
coefficients_poly = poly_reg_model.named_steps['linearregression'].coef_
print(f"Коэффициенты: {coefficients_poly}")
print(f"Полином: {polynomial_to_string(coefficients_poly)}\n")

print("\nLasso-регрессия (с регуляризацией):")
coefficients_lasso = best_lasso_model.named_steps['lasso'].coef_
print(f"Коэффициенты: {coefficients_lasso}")
print(f"Полином: {polynomial_to_string(coefficients_lasso)}\n")

print("\nRidge-регрессия (с регуляризацией):")
coefficients_ridge = best_ridge_model.named_steps['ridge'].coef_
print(f"Коэффициенты: {coefficients_ridge}")
print(f"Полином: {polynomial_to_string(coefficients_ridge)}\n")

# Визуализация
X_plot = np.linspace(0, 1, 500)[:, np.newaxis]
y_plot_poly = poly_reg_model.predict(X_plot)
y_plot_lasso = best_lasso_model.predict(X_plot)
y_plot_ridge = best_ridge_model.predict(X_plot)

plt.figure(figsize=(14, 8))
plt.scatter(X, y, color='black', label='Исходные данные')
plt.plot(X_plot, y_plot_poly, label='Полиномиальная регрессия', color='blue')
plt.plot(X_plot, y_plot_lasso, label='Lasso-регрессия', color='red')
plt.plot(X_plot, y_plot_ridge, label='Ridge-регрессия', color='green')
plt.legend()
plt.title('Сравнение моделей')
plt.xlabel('X')
plt.ylabel('y')
plt.grid(True)
plt.show()
