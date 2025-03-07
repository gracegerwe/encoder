import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import requests
from io import BytesIO
from PIL import Image

class VisualCortexEncoder:
    def __init__(self, v1_neurons=1000, v4_neurons=500):
        # Remove the fixed seed to allow different initializations
        self.v1_neurons = v1_neurons
        self.v4_neurons = v4_neurons
        
        # Initialize neural properties with random values
        self.v1_orientations = np.random.rand(v1_neurons) * np.pi
        self.v1_spatial_freqs = np.exp(np.random.randn(v1_neurons) * 0.4 + 1.5)
        self.v1_positions_x = np.random.rand(v1_neurons)
        self.v1_positions_y = np.random.rand(v1_neurons)
        self.v1_receptive_sizes = np.exp(np.random.randn(v1_neurons) * 0.3 - 1.5)
        
        # Initialize V1-V4 connectivity
        self.v1_to_v4 = self._initialize_v4_weights()
        
        # Create Gabor filter bank
        self.gabor_filters = self._create_gabor_filters()

    def process_image(self, image):
        """
        Process an image through the visual cortex model.
        Each V1 neuron should respond to specific features at its receptive field location.
        """
        # Convert image to grayscale numpy array
        if isinstance(image, Image.Image):
            image = np.array(image.convert('L'))
        elif len(image.shape) == 3:
            image = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
        
        # Normalize and resize
        image = image / 255.0 if image.max() > 1 else image
        image = Image.fromarray((image * 255).astype(np.uint8))
        image = np.array(image.resize((128, 128), Image.Resampling.BILINEAR)) / 255.0
        
        # Add debug prints
        print("\nImage statistics:")
        print(f"Image mean: {np.mean(image):.3f}")
        print(f"Image std: {np.std(image):.3f}")
        
        # First, apply all Gabor filters to the full image to get feature maps
        feature_maps = []
        for gabor in self.gabor_filters:
            filtered = signal.correlate2d(image - np.mean(image), gabor, mode='same')
            feature_maps.append(filtered)  # Note: not taking absolute value yet
        
        # Process through V1 neurons
        v1_responses = np.zeros(self.v1_neurons)
        
        for i in range(self.v1_neurons):
            # Get neuron's preferred location
            x_center = int(self.v1_positions_x[i] * 128)
            y_center = int(self.v1_positions_y[i] * 128)
            
            # Get response from the feature map at the neuron's location
            rf_size = 5
            x_min = max(0, x_center - rf_size)
            x_max = min(128, x_center + rf_size)
            y_min = max(0, y_center - rf_size)
            y_max = min(128, y_center + rf_size)
            
            # Get local image patch for contrast normalization
            image_patch = image[y_min:y_max, x_min:x_max]
            local_contrast = np.std(image_patch) + 1e-8
            
            # Get response from the appropriate feature map
            orientation_idx = int((self.v1_orientations[i] / np.pi * 8) % 8)
            scale_idx = min(3, int(self.v1_spatial_freqs[i]))
            filter_idx = scale_idx * 8 + orientation_idx
            
            feature_response = feature_maps[filter_idx][y_min:y_max, x_min:x_max]
            v1_responses[i] = np.abs(feature_response).max() / local_contrast
        
        # Normalize V1 responses
        v1_responses = v1_responses / (v1_responses.max() + 1e-8)
        
        # After V1 processing
        print("\nV1 responses:")
        print(f"Max V1: {np.max(v1_responses):.3f}")
        print(f"Mean V1: {np.mean(v1_responses):.3f}")
        print(f"Std V1: {np.std(v1_responses):.3f}")
        
        # Compute V4 responses
        v4_responses = np.dot(self.v1_to_v4, v1_responses)
        v4_responses = v4_responses / (v4_responses.max() + 1e-8)
        
        # After V4 processing
        print("\nV4 responses:")
        print(f"Max V4: {np.max(v4_responses):.3f}")
        print(f"Mean V4: {np.mean(v4_responses):.3f}")
        print(f"Std V4: {np.std(v4_responses):.3f}")
        
        return {
            'v1_responses': v1_responses,
            'v4_responses': v4_responses
        }

    def _initialize_v4_weights(self):
        """Initialize V4 weights with neurobiological principles"""
        weights = np.random.randn(self.v4_neurons, self.v1_neurons) * 0.1
        
        for i in range(self.v4_neurons):
            preferred_orientation = np.random.rand() * np.pi
            preferred_x = np.random.rand()
            preferred_y = np.random.rand()
            
            orientation_similarity = np.abs(np.cos(2 * (self.v1_orientations - preferred_orientation)))
            position_similarity = np.exp(-((self.v1_positions_x - preferred_x)**2 + 
                                        (self.v1_positions_y - preferred_y)**2) / 0.1)
            
            weights[i] *= (1 + 5 * orientation_similarity * position_similarity)
            weights[i] /= np.abs(weights[i]).sum()
        
        return weights

    def _create_gabor_filters(self, num_orientations=8, num_scales=4):
        """Create Gabor filter bank"""
        filters = []
        kernel_size = 25
        
        for scale in range(num_scales):
            sigma = 5 * (scale + 1)
            lambda_val = sigma / 0.8
            
            for orientation in range(num_orientations):
                theta = np.pi * orientation / num_orientations
                kernel = self._gabor_kernel(sigma, theta, lambda_val, 0.5, 0, kernel_size)
                filters.append(kernel)
                
        return filters
    
    def _gabor_kernel(self, sigma, theta, lambda_val, gamma, psi, size):
        """Generate a Gabor kernel"""
        sigma_x = sigma
        sigma_y = sigma / gamma
        
        # Grid of coordinates centered at zero
        half_size = size // 2
        x, y = np.meshgrid(np.arange(-half_size, half_size + 1),
                          np.arange(-half_size, half_size + 1))
        
        # Rotation
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)
        
        # Calculate Gabor function
        gb = np.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2)) * \
             np.cos(2 * np.pi * x_theta / lambda_val + psi)
        
        return gb
    
    def visualize_responses(self, image, figsize=(18, 6)):
        """Visualize V1 and V4 responses to an input image in a single row with 3 plots."""
        # Convert PIL Image to numpy array if needed
        if hasattr(image, 'convert'):
            image_array = np.array(image)
        else:
            image_array = image

        responses = self.process_image(image_array)

        # Create figure with 1 row, 3 columns
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Show input image
        ax = axes[0]
        ax.axis("off")
        ax.set_title("Input Image")
        if len(image_array.shape) == 3:
            ax.imshow(image_array)
        else:
            ax.imshow(image_array, cmap='gray')

        # Show V1 responses as a 2D map
        v1_2d = np.zeros((40, 40))
        for i in range(min(1600, self.v1_neurons)):  # Only plot a subset if there are too many neurons
            x = int(self.v1_positions_x[i] * 40)
            y = int(self.v1_positions_y[i] * 40)
            x = min(39, max(0, x))  # Ensure within bounds
            y = min(39, max(0, y))  # Ensure within bounds
            v1_2d[y, x] = max(v1_2d[y, x], responses['v1_responses'][i])

        ax = axes[1]
        img = ax.imshow(gaussian_filter(v1_2d, sigma=1), cmap='viridis')
        ax.set_title("V1 Responses")
        fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

        # Show V4 connectivity
        ax = axes[2]
        v4_example = self.v1_to_v4[0, :]  # First V4 neuron's connections
        v4_2d = np.zeros((40, 40))
        for i in range(min(1600, self.v1_neurons)):
            x = int(self.v1_positions_x[i] * 40)
            y = int(self.v1_positions_y[i] * 40)
            x = min(39, max(0, x))  # Ensure within bounds
            y = min(39, max(0, y))  # Ensure within bounds
            v4_2d[y, x] = max(v4_2d[y, x], abs(v4_example[i]))

        img = ax.imshow(gaussian_filter(v4_2d, sigma=1), cmap='plasma')
        ax.set_title("Example V4 Neuron Connectivity")
        fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

        return responses

# Function to download and process a test image
def download_test_image(url=None):
    if url is None:
        # Default image: Lena test image
        url = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"
        
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        print(f"Error downloading image: {e}")
        # Create a simple test pattern instead
        img = Image.new('RGB', (256, 256), color='white')
        return img

# Main execution
if __name__ == "__main__":
    print("Downloading test image...")
    
    # Try different test images for different visual features
    image_urls = [
        "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png",  # Lena - standard test image
        "https://images.pexels.com/photos/414612/pexels-photo-414612.jpeg",  # Landscape with clear edges
        "https://images.pexels.com/photos/1170986/pexels-photo-1170986.jpeg"  # Face with varied features
    ]
    
    print("Creating visual cortex encoder...")
    # Use smaller neuron counts for faster execution
    encoder = VisualCortexEncoder(v1_neurons=500, v4_neurons=100)
    
    for i, url in enumerate(image_urls):
        try:
            print(f"\nProcessing test image {i+1}...")
            test_image = download_test_image(url)
            print("Visualizing neural responses...")
            encoder.visualize_responses(test_image)
            print(f"Visualization complete for image {i+1}!")
        except Exception as e:
            print(f"Error processing image {i+1}: {e}")
    
    print("\nAll done! You should see several figures with visualizations.")