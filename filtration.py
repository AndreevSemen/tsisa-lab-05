import matplotlib.pyplot as plt
import numpy as np
import sys


def omega_criterion(function):
    sum = 0.
    for k in range(1, function.size):
        sum += np.power(function[k] - function[k - 1], 2)

    return np.sqrt(sum)


def delta_criterion(average_func, noised_func):
    sum = 0.
    for k in range(1, average_func.size):
        sum += np.power(average_func[k] - noised_func[k], 2)

    return np.sqrt(np.divide(sum, average_func.size))


def norm_vector(vector):
    sum = np.sum(vector)
    return np.divide(vector, sum)


def weighted_moving_average(function, M, alpha):
    filtered_function = np.copy(function)
    for k in range(M, filtered_function.size - M):
        sum = 0.
        for j in range(k - M, k + M):
            sum += np.multiply(np.power(filtered_function[j], 2), alpha[j + M + 1 - k])
        filtered_function[k] = np.sqrt(sum)

    return filtered_function


def random_alpha_vector(M):
    rng = np.random.default_rng()

    r = 2*M + 1
    vector = np.zeros(shape=r)
    vector[M] = rng.uniform(0, 1)

    for m in range(0, M):
        vector[m] = vector[r - m - 1] = 0.5*rng.uniform(0, np.sum(vector[m+1:r-m]))

    vector = norm_vector(vector)

    return vector


def get_noised_function(function, noise_amplitude) :
    random_generator = np.random.default_rng()

    noised_function = np.copy(function)
    for index in range(noised_function.size):
        random_piece = random_generator.uniform(-noise_amplitude/2., +noise_amplitude/2.)
        noised_function[index] = noised_function[index] + random_piece

    return noised_function


def get_filtered_function(function, M):
    best_lambda = 0
    best_alpha = None
    min_J_value = sys.float_info.max

    for lambda_index in range(0, 11):
        lam = np.divide(lambda_index, 10)
        J = lambda filtered, noised:     (lam) * omega_criterion(filtered) + \
                                     (1 - lam) * delta_criterion(filtered, noised)

        N = int(np.divide(np.log(1-0.95), np.log(1 - np.divide(0.01, np.pi))))

        for i in range(0, N):
            alpha = random_alpha_vector(M)
            filtered_function = weighted_moving_average(function, M, alpha)

            J_value = J(filtered_function, function)
            if np.less(J_value, min_J_value):
                min_J_value = J_value
                best_lambda = lam
                best_alpha = alpha

    print("Best lambda and alpha : ", best_lambda, best_alpha)

    return weighted_moving_average(function, alpha=best_alpha, M=M)


if __name__ == '__main__':
    x_array = np.array(range(0, 100))
    x_array = np.divide(x_array, 100)
    x_array = np.multiply(x_array, np.pi)

    source_function = np.add(np.sin(x_array), 0.5)
    noised_function = get_noised_function(source_function, 0.5)

    filtered_function3 = get_filtered_function(noised_function, 1)
    filtered_function5 = get_filtered_function(noised_function, 2)

    plt.plot(x_array, source_function)
    plt.plot(x_array, noised_function)
    plt.plot(x_array, filtered_function3)
    plt.show()

    plt.plot(x_array, source_function)
    plt.plot(x_array, noised_function)
    plt.plot(x_array, filtered_function5)
    plt.show()