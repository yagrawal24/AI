from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    dataset = np.load(filename)
    mean = np.mean(dataset, axis=0)
    dataset = dataset - mean

    return dataset


def get_covariance(dataset):
    transpose = np.transpose(dataset)
    covariance = np.dot(transpose, dataset)
    covariance = covariance / (len(dataset) - 1)

    return covariance


def get_eig(S, m):
    largest_m_values = [len(S) - m, len(S) - 1]
    eigen_values, eigen_vectors = eigh(S, subset_by_index=largest_m_values)
    # Formatting Eigen Values
    eigen_values = np.flipud(eigen_values)
    eigen_values = np.diag(eigen_values)
    # Formatting Eigen Vectors
    eigen_vectors = np.fliplr(eigen_vectors)

    return eigen_values, eigen_vectors


def get_eig_perc(S, perc):
    trace_val = S.trace()
    minval = perc * trace_val

    eigen_values, eigen_vectors = eigh(S, subset_by_value=[minval, np.inf])
    # Formatting Eigen Values
    eigen_values = np.flipud(eigen_values)
    eigen_values = np.diag(eigen_values)
    # Formatting Eigen Vectors
    eigen_vectors = np.fliplr(eigen_vectors)

    return eigen_values, eigen_vectors


def project_image(img, U):
    alphaj = []
    transpose_u = np.transpose(U)

    for i in transpose_u:
        alphaj.append(np.dot(img, i))

    x = 0
    total = np.zeros(len(U))
    for element in transpose_u:
        total += (alphaj[x] * element)
        x += 1

    return total


def display_image(orig, proj):
    reshaped_orig_image = np.reshape(orig, (32, 32))
    reshaped_orig_image = reshaped_orig_image.transpose()

    reshaped_proj_image = np.reshape(proj, (32, 32))
    reshaped_proj_image = reshaped_proj_image.transpose()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(32, 32)
    ax1.set_title('Original')
    ax2.plot(32, 32)
    ax2.set_title('Projection')

    pos1 = ax1.imshow(reshaped_orig_image, aspect='equal')
    pos2 = ax2.imshow(reshaped_proj_image, aspect='equal')

    fig.colorbar(pos1, ax=ax1)
    fig.colorbar(pos2, ax=ax2)

    plt.show()
    
def main():
    print("Input number between 1 and 1024 (inclusive) for dimension")
    dim = int(input())
    print("Input number between 0 and 2413 (inslusive) for picture from dataset")
    pic = int(input())
    x = load_and_center_dataset('YaleB_32x32.npy')
    S = get_covariance(x)
    Lambda, U = get_eig(S, dim)
    projection = project_image(x[pic], U)
    display_image(x[pic], projection)


if __name__ == '__main__':
    main()

