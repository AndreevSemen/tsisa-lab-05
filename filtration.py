import matplotlib.pyplot as plt
import numpy as np
import sys
import json


def omega_criterion(function):
    sum = 0.
    for k in range(1, function.size):
        sum += np.power(function[k] - function[k - 1], 2)

    return np.sqrt(sum)


def delta_criterion(average_func, noised_func):
    sum = 0.
    for k in range(0, average_func.size):
        sum += np.power(average_func[k] - noised_func[k], 2)

    return np.sqrt(np.divide(sum, average_func.size))


def get_J(omega, delta, lam):
    return (lam)*omega + (1 - lam)*delta


def euclidean_distance(omega, delta):
    return np.sqrt(np.power(omega, 2) + np.power(delta, 2))


def norm_vector(vector):
    sum = np.sum(vector)
    return np.divide(vector, sum)


def weighted_moving_average(function, M, alpha):
    extended_function = np.copy(function)

    dummy = np.array([0 for i in range(0, M)])
    extended_function = np.append(dummy, extended_function)
    extended_function = np.append(extended_function, dummy)

    filtered_function = np.copy(extended_function)

    for k in range(M, filtered_function.size - M):
        sum = 0.
        for j in range(k - M, k + M + 1):
            sum += np.multiply(np.power(filtered_function[j], 2), alpha[j + M - k])
        filtered_function[k] = np.sqrt(sum)

    filtered_function = filtered_function[M:-M]

    return filtered_function


def random_alpha_vector(M):
    rng = np.random.default_rng()

    r = 2*M + 1
    vector = np.zeros(shape=r)
    vector[M] = rng.uniform(0, 1)
    for m in range(0, M):
        vector[m] = vector[-(m + 1)] = rng.uniform(0, 1)

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
    min_distance = sys.float_info.max
    best_alpha = None

    log_json = json.loads('[]')

    for lam in np.linspace(0., 1., 11):
        local_best_alpha = None
        local_min_J_value = sys.float_info.max

        N = int(np.divide(np.log(1-0.95), np.log(1 - np.divide(0.01, np.pi))))

        for i in range(0, N):
            alpha = random_alpha_vector(M)
            filtered_function = weighted_moving_average(function, M, alpha)

            omega = omega_criterion(filtered_function)
            delta = delta_criterion(function, filtered_function)
            J_value = get_J(omega, delta, lam)
            if np.less(J_value, local_min_J_value):
                local_min_J_value = J_value
                local_best_alpha = alpha

        filtered_function = weighted_moving_average(function, M, local_best_alpha)

        omega = omega_criterion(filtered_function)
        delta = delta_criterion(filtered_function, function)
        J_value = get_J(omega, delta, lam)
        distance = euclidean_distance(omega, delta)

        experement_log = json.loads('{}')
        experement_log['lam'] = lam
        experement_log['alpha'] = []
        for item in local_best_alpha:
            experement_log['alpha'].append(item)
        experement_log['omega'] = omega
        experement_log['delta'] = delta
        experement_log['J'] = local_min_J_value
        experement_log['distance'] = distance

        log_json.append(experement_log)

        print('lam :', lam, 'omega :', omega, 'delta :', delta, 'J :', J_value, 'distance :', distance)
        if np.less(distance, min_distance):
            best_lambda = lam
            min_distance = distance
            best_alpha = local_best_alpha

    print("Best lambda and alpha : ", best_lambda, best_alpha)


    print(json.dumps(log_json, indent=4))

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

    plt.legend(['source function', 'noised function', 'filtered function (r = 3)'])

    plt.show()

    plt.plot(x_array, source_function)
    plt.plot(x_array, noised_function)
    plt.plot(x_array, filtered_function5)

    plt.legend(['source function', 'noised function', 'filtered function (r = 5)'])

    plt.show()
