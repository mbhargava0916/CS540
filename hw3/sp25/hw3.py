from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    # Your implementation goes here!
    dataset = np.load(filename)
    dataset = dataset.astype(float)
    mean = np.mean(dataset, axis=0)
    centered_dataset = dataset - mean
    return centered_dataset
    # raise NotImplementedError

def get_covariance(dataset):
    # Your implementation goes here!
    n = dataset.shape[0]
    covariance_matrix = np.dot(dataset.T, dataset) / (n - 1)
    return covariance_matrix

def get_eig(S, k):
    # Your implementation goes here!
    eigenvalues, eigenvectors = eigh(S, subset_by_index=[S.shape[0]-k, S.shape[0]-1])
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    Lambda = np.diag(eigenvalues[:k])
    U = eigenvectors[:, :k]
    return Lambda, U

def get_eig_prop(S, prop):
    # Your implementation goes here!
    
    # Compute all eigenvalues and eigenvectors
    eigenvalue, eigenvector = eigh(S)
    eigenvalue = np.flip(eigenvalue)
    eigenvector = np.flip((eigenvector), axis=1)
    #change eigenvectors from columns to rows
    eigenvector = np.transpose(eigenvector)
    
    #sum of all eigenvalues
    sum = 0
    for i in eigenvalue:
        sum += i
    
    cer_prop_eigenvalues = []
    cer_prop_eigenvectors = []
    
    
    for val, vector in zip(eigenvalue,eigenvector):
        if val/sum > prop:
            cer_prop_eigenvalues.append(val)
            cer_prop_eigenvectors.append(vector)
        
    
    diagonal = np.diag(np.asarray(cer_prop_eigenvalues))
    return diagonal, np.transpose(np.asarray(cer_prop_eigenvectors))

def project_and_reconstruct_image(image, U):
    # Your implementation goes here!
    alpha = np.dot(U.T, image)
    reconstructed_image = np.dot(U, alpha)
    return reconstructed_image

def display_image(im_orig_fullres, im_orig, im_reconstructed):
    # Reshape im_orig_fullres to (218, 178, 3) for RGB images
    im_orig_fullres = im_orig_fullres.reshape(218, 178, 3)
    
    # Create the figure and subplots
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(9, 3), ncols=3)
    fig.tight_layout()

    # Display the original high-resolution image
    ax1.imshow(im_orig_fullres, aspect='equal')
    ax1.set_title("Original High Res")

    # Display the original low-resolution image
    ax2.imshow(im_orig.reshape(60, 50), cmap='gray', aspect='equal')
    ax2.set_title("Original")
    plt.colorbar(ax2.imshow(im_orig.reshape(60, 50), cmap='gray', aspect='equal'), ax=ax2)

    # Display the reconstructed image
    ax3.imshow(im_reconstructed.reshape(60, 50), cmap='gray', aspect='equal')
    ax3.set_title("Reconstructed")
    plt.colorbar(ax3.imshow(im_reconstructed.reshape(60, 50), cmap='gray', aspect='equal'), ax=ax3)

    return fig, ax1, ax2, ax3

def perturb_image(image, U, sigma):
    # Your implementation goes here!
    alpha = np.dot(U.T, image)
    perturbation = np.random.normal(0, sigma, alpha.shape)
    perturbed_alpha = alpha + perturbation
    perturbed_image = np.dot(U, perturbed_alpha)
    return perturbed_image

if __name__ == "__main__":
    # Define the simple covariance matrix
    x = np.array([[1,2,5],[3,4,7]])
    S = get_covariance(x)
    Lambda, U = get_eig(S, 3)
    print(Lambda)