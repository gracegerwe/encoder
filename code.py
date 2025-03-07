import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import requests
from io import BytesIO
from PIL import Image

class VisualCortexEncoder:
    def __init__(self, v1_neurons=1000, v4_neurons=500):
        """
        Initialize a computational model of V1 and V4 visual cortex areas.
        
        Parameters:
        v1_neurons (int): Number of simulated V1 neurons
        v4_neurons (int): Number of simulated V4 neurons
        """
        self.v1_neurons = v1_neurons
        self.v4_neurons = v4_neurons
        
        # Initialize V1 properties (orientation, spatial frequency, position)
        self.v1_orientations = np.random.uniform(0, np.pi, v1_neurons)
        self.v1_spatial_freqs = np.random.lognormal(mean=1.5, sigma=0.4, size=v1_neurons)
        self.v1_positions_x = np.random.uniform(0, 1, v1_neurons)
        self.v1_positions_y = np.random.uniform(0, 1, v1_neurons)
        self.v1_receptive_sizes = np.random.lognormal(mean=-1.5, sigma=0.3, size=v1_neurons)
        
        # Create V1-V4 connectivity matrix (each V4 neuron receives input from a subset of V1 neurons)
        self.v1_v4_weights = np.random.normal(0, 0.1, (v4_neurons, v1_neurons))
        
        # For each V4 neuron, strengthen connections from V1 neurons with similar properties
        for i in range(v4_neurons):
            # Select a preferred orientation and position for this V4 neuron
            preferred_orientation = np.random.uniform(0, np.pi)
            preferred_x = np.random.uniform(0, 1)
            preferred_y = np.random.uniform(0, 1)
            
            # Strengthen weights from V1 neurons with similar properties
            for j in range(v1_neurons):
                orientation_similarity = np.abs(np.cos(2 * (self.v1_orientations[j] - preferred_orientation)))
                position_similarity = np.exp(-((self.v1_positions_x[j] - preferred_x)**2 + 
                                             (self.v1_positions_y[j] - preferred_y)**2) / 0.1)
                
                # V4 neurons receive stronger input from V1 neurons with similar properties
                self.v1_v4_weights[i, j] *= (1 + 5 * orientation_similarity * position_similarity)
        
        # Normalize weights
        for i in range(v4_neurons):
            self.v1_v4_weights[i, :] /= np.sum(np.abs(self.v1_v4_weights[i, :]))
    
    def create_gabor_filters(self, image_size, num_orientations=8, num_scales=4):
        """Create a bank of Gabor filters to model V1 simple cells"""
        gabor_filters = []
        for scale in range(num_scales):
            sigma = 5 * (scale + 1)
            lambda_val = sigma / 0.8
            
            for orientation in range(num_orientations):
                theta = np.pi * orientation / num_orientations
                gabor_kernel = self._gabor_kernel(sigma, theta, lambda_val, 0.5, 0, image_size)
                gabor_filters.append(gabor_kernel)
                
        return gabor_filters
    
    def _gabor_kernel(self, sigma, theta, lambda_val, gamma, psi, size):
        """Generate a Gabor kernel"""
        sigma_x = sigma
        sigma_y = sigma / gamma
        
        # Grid of coordinates centered at zero
        half_size = size // 2
        x, y = np.meshgrid(np.arange(-half_size, half_size + 1), np.arange(-half_size, half_size + 1))
        
        # Rotation
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)
        
        # Calculate Gabor function
        gb = np.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2)) * np.cos(2 * np.pi * x_theta / lambda_val + psi)
        
        return gb
    
    def process_image(self, image, normalize=True):
        """
        Process an image through the visual cortex model
        
        Parameters:
        image: Input image (will be resized to a standard size)
        normalize: Whether to normalize the neural responses
        
        Returns:
        Dictionary containing simulated V1 and V4 neural responses
        """
        # Convert PIL Image to numpy array if needed
        if hasattr(image, 'convert'):
            image = np.array(image)
            
        # Resize and convert to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] > 1:
            # Simple grayscale conversion
            gray_image = np.mean(image, axis=2)
        else:
            gray_image = image
            
        # Resize to 128x128
        h, w = gray_image.shape
        resized_image = np.array(Image.fromarray(gray_image.astype(np.uint8)).resize((128, 128)))
        resized_image = resized_image / 255.0  # Normalize to [0,1]
        
        # Create Gabor filters to simulate V1 responses
        gabor_filters = self.create_gabor_filters(image_size=25)
        
        # Apply the filters to get V1 responses
        v1_responses = np.zeros(self.v1_neurons)
        for i in range(self.v1_neurons):
            # Create a mask for where this V1 neuron's receptive field is located
            x_center = int(self.v1_positions_x[i] * resized_image.shape[1])
            y_center = int(self.v1_positions_y[i] * resized_image.shape[0])
            rf_size = int(np.exp(self.v1_receptive_sizes[i]) * 20)  # Convert log size to pixels
            
            # Extract patch (with boundary handling)
            x_min = max(0, x_center - rf_size)
            x_max = min(resized_image.shape[1], x_center + rf_size)
            y_min = max(0, y_center - rf_size)
            y_max = min(resized_image.shape[0], y_center + rf_size)
            
            if x_max <= x_min or y_max <= y_min:
                continue  # Skip if receptive field is out of bounds
                
            patch = resized_image[y_min:y_max, x_min:x_max]
            
            # Skip if patch is too small
            if patch.shape[0] < 3 or patch.shape[1] < 3:
                continue
                
            # Choose a Gabor filter based on this neuron's orientation preference
            orientation_index = int((self.v1_orientations[i] / np.pi) * 8) % 8
            scale_index = min(3, int(self.v1_spatial_freqs[i])) 
            filter_index = scale_index * 8 + orientation_index
            
            # Resize filter to match patch if needed
            gabor = gabor_filters[filter_index]
            gabor_h, gabor_w = gabor.shape
            patch_h, patch_w = patch.shape
            
            if gabor_h > patch_h or gabor_w > patch_w:
                # Resize gabor to fit within patch
                new_h = min(gabor_h, patch_h)
                new_w = min(gabor_w, patch_w)
                gabor = np.array(Image.fromarray(gabor).resize((new_w, new_h)))
            
            # Convolve patch with Gabor filter and get maximum response
            if gabor.shape[0] <= patch.shape[0] and gabor.shape[1] <= patch.shape[1]:
                try:
                    response = signal.convolve2d(patch, gabor, mode='valid')
                    v1_responses[i] = np.max(np.abs(response))
                except ValueError:
                    continue  # Skip if convolution fails
        
        # Apply nonlinearity to V1 responses (rectification and saturation)
        v1_responses = np.maximum(0, v1_responses)  # ReLU-like
        v1_responses = 1 / (1 + np.exp(-v1_responses + 3))  # Sigmoid-like
        
        # Add realistic noise to V1 responses
        v1_responses += np.random.normal(0, 0.05, self.v1_neurons)
        v1_responses = np.maximum(0, v1_responses)  # Ensure no negative responses
        
        # Compute V4 responses through weighted connections
        v4_responses = np.dot(self.v1_v4_weights, v1_responses)
        
        # Apply nonlinearity and add noise to V4 responses
        v4_responses = np.maximum(0, v4_responses)  # ReLU-like
        v4_responses = 2 / (1 + np.exp(-v4_responses + 2)) - 1  # Scaled sigmoid
        v4_responses += np.random.normal(0, 0.03, self.v4_neurons)
        v4_responses = np.maximum(0, v4_responses)  # Ensure no negative responses
        
        # Normalize responses if requested
        if normalize:
            if np.max(v1_responses) > 0:
                v1_responses /= np.max(v1_responses)
            if np.max(v4_responses) > 0:
                v4_responses /= np.max(v4_responses)
        
        return {
            'v1_responses': v1_responses,
            'v4_responses': v4_responses
        }
    
    def visualize_responses(self, image, figsize=(15, 10)):
        """Visualize V1 and V4 responses to an input image"""
        # Convert PIL Image to numpy array if needed
        if hasattr(image, 'convert'):
            image_array = np.array(image)
        else:
            image_array = image
            
        responses = self.process_image(image_array)
        
        plt.figure(figsize=figsize)
        
        # Show input image
        plt.subplot(2, 2, 1)
        if len(image_array.shape) == 3:
            plt.imshow(image_array)
        else:
            plt.imshow(image_array, cmap='gray')
        plt.title('Input Image')
        plt.axis('off')
        
        # Show V1 responses map (2D arrangement)
        v1_2d = np.zeros((40, 40))
        for i in range(min(1600, self.v1_neurons)):  # Only plot a subset if there are too many neurons
            x = int(self.v1_positions_x[i] * 40)
            y = int(self.v1_positions_y[i] * 40)
            x = min(39, max(0, x))  # Ensure within bounds
            y = min(39, max(0, y))  # Ensure within bounds
            v1_2d[y, x] = max(v1_2d[y, x], responses['v1_responses'][i])
        
        plt.subplot(2, 2, 2)
        plt.imshow(gaussian_filter(v1_2d, sigma=1), cmap='viridis')
        plt.title('V1 Responses')
        plt.colorbar()
        
        # Show V4 neuron responses as a bar plot (top responding neurons)
        plt.subplot(2, 1, 2)
        num_to_show = min(20, self.v4_neurons)
        top_v4_indices = np.argsort(responses['v4_responses'])[-num_to_show:]
        plt.bar(range(len(top_v4_indices)), 
                responses['v4_responses'][top_v4_indices],
                tick_label=[f"V4_{i}" for i in top_v4_indices])
        plt.title(f'Top {num_to_show} V4 Neuron Responses')
        plt.ylabel('Activation')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # For visual comparison: Visualize some Gabor filters
        plt.figure(figsize=(10, 3))
        plt.suptitle("Sample Gabor Filters (V1 Simple Cells)")
        gabor_filters = self.create_gabor_filters(image_size=25)
        
        for i in range(8):  # Show 8 orientations at one scale
            plt.subplot(1, 8, i+1)
            plt.imshow(gabor_filters[i], cmap='gray')
            plt.axis('off')
            plt.title(f"{(i*22.5):.0f}°")
            
        plt.tight_layout()
            
        # Also visualize V4 connectivity
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        v4_example = self.v1_v4_weights[0, :]  # First V4 neuron's connections
        v4_2d = np.zeros((40, 40))
        for i in range(min(1600, self.v1_neurons)):
            x = int(self.v1_positions_x[i] * 40)
            y = int(self.v1_positions_y[i] * 40)
            x = min(39, max(0, x))  # Ensure within bounds
            y = min(39, max(0, y))  # Ensure within bounds
            v4_2d[y, x] = max(v4_2d[y, x], abs(v4_example[i]))
        
        plt.imshow(gaussian_filter(v4_2d, sigma=1), cmap='plasma')
        plt.title('Example V4 Neuron Connectivity')
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        # Show orientation tuning of this V4 neuron
        orientation_bins = np.linspace(0, np.pi, 16)
        orientation_weights = np.zeros(len(orientation_bins)-1)
        
        for i in range(self.v1_neurons):
            bin_idx = np.digitize(self.v1_orientations[i], orientation_bins) - 1
            if 0 <= bin_idx < len(orientation_weights):
                orientation_weights[bin_idx] += abs(v4_example[i])
                
        bin_centers = (orientation_bins[:-1] + orientation_bins[1:]) / 2
        bin_degrees = bin_centers * 180 / np.pi
        
        plt.bar(bin_degrees, orientation_weights / np.sum(orientation_weights), width=180/16)
        plt.title('V4 Neuron Orientation Preference')
        plt.xlabel('Orientation (degrees)')
        plt.ylabel('Normalized Weight')
        
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

# Main execution - just run this script directly
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