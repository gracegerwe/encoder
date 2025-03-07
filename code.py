import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from skimage import io, color, transform
from scipy.ndimage import gaussian_filter

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
        # Resize and convert to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] > 1:
            gray_image = color.rgb2gray(image)
        else:
            gray_image = image
            
        resized_image = transform.resize(gray_image, (128, 128))
        
        # Create Gabor filters to simulate V1 responses
        gabor_filters = self.create_gabor_filters(image_size=25)
        
        # Apply the filters to get V1 responses
        v1_responses = np.zeros(self.v1_neurons)
        for i in range(self.v1_neurons):
            # Create a mask for where this V1 neuron's receptive field is located
            x_center = int(self.v1_positions_x[i] * resized_image.shape[1])
            y_center = int(self.v1_positions_y[i] * resized_image.shape[0])
            rf_size = int(np.exp(self.v1_receptive_sizes[i]) * 20) # Convert log size to pixels
            
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
            if gabor.shape[0] > patch.shape[0] or gabor.shape[1] > patch.shape[1]:
                gabor = transform.resize(gabor, (
                    min(gabor.shape[0], patch.shape[0]),
                    min(gabor.shape[1], patch.shape[1])
                ))
            
            # Convolve patch with Gabor filter and get maximum response
            if gabor.shape[0] <= patch.shape[0] and gabor.shape[1] <= patch.shape[1]:
                response = signal.convolve2d(patch, gabor, mode='valid')
                v1_responses[i] = np.max(np.abs(response))
        
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
    
    def generate_training_dataset(self, image_paths, save_path=None):
        """
        Generate a dataset of simulated neural responses to images
        
        Parameters:
        image_paths: List of paths to input images
        save_path: Optional path to save the dataset
        
        Returns:
        Dictionary containing stimuli and corresponding neural responses
        """
        dataset = {
            'images': [],
            'v1_responses': [],
            'v4_responses': []
        }
        
        for img_path in image_paths:
            try:
                img = io.imread(img_path)
                neural_responses = self.process_image(img)
                
                # Store results
                dataset['images'].append(img_path)
                dataset['v1_responses'].append(neural_responses['v1_responses'])
                dataset['v4_responses'].append(neural_responses['v4_responses'])
                
                print(f"Processed {img_path}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # Convert to numpy arrays
        dataset['v1_responses'] = np.array(dataset['v1_responses'])
        dataset['v4_responses'] = np.array(dataset['v4_responses'])
        
        # Save dataset if requested
        if save_path:
            np.save(save_path, dataset)
            print(f"Dataset saved to {save_path}")
        
        return dataset
    
    def visualize_responses(self, image, figsize=(15, 10)):
        """Visualize V1 and V4 responses to an input image"""
        responses = self.process_image(image)
        
        plt.figure(figsize=figsize)
        
        # Show input image
        plt.subplot(2, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Input Image')
        plt.axis('off')
        
        # Show V1 responses map (2D arrangement)
        v1_2d = np.zeros((40, 40))
        for i in range(self.v1_neurons):
            if i < 1600:  # Only plot a subset if there are too many neurons
                x = int(self.v1_positions_x[i] * 40)
                y = int(self.v1_positions_y[i] * 40)
                v1_2d[y, x] = max(v1_2d[y, x], responses['v1_responses'][i])
        
        plt.subplot(2, 2, 2)
        plt.imshow(gaussian_filter(v1_2d, sigma=1), cmap='viridis')
        plt.title('V1 Responses')
        plt.colorbar()
        
        # Show V4 neuron responses as a bar plot (top responding neurons)
        plt.subplot(2, 1, 2)
        top_v4_indices = np.argsort(responses['v4_responses'])[-20:]
        plt.bar(range(len(top_v4_indices)), 
                responses['v4_responses'][top_v4_indices],
                tick_label=[f"V4_{i}" for i in top_v4_indices])
        plt.title('Top 20 V4 Neuron Responses')
        plt.ylabel('Activation')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()

# Example of training a basic decoder model using the simulated responses
def train_decoder_model(encoder, image_paths, epochs=100):
    """
    Train a simple decoder model that attempts to reconstruct visual features
    from simulated neural activity
    """
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    
    # Generate dataset of neural responses
    dataset = encoder.generate_training_dataset(image_paths)
    
    # Extract features from images (e.g., using a pre-trained CNN)
    import tensorflow as tf
    feature_extractor = tf.keras.applications.VGG16(
        include_top=False, 
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    
    # Extract features
    image_features = []
    for img_path in dataset['images']:
        img = io.imread(img_path)
        img = transform.resize(img, (224, 224))
        if len(img.shape) == 2:  # If grayscale
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 4:  # If RGBA
            img = img[:,:,:3]
        
        # Ensure values are in [0, 255]
        if img.max() <= 1.0:
            img = img * 255
            
        img = tf.keras.applications.vgg16.preprocess_input(img)
        features = feature_extractor.predict(np.expand_dims(img, axis=0))
        image_features.append(features.flatten())
    
    image_features = np.array(image_features)
    
    # Combine V1 and V4 responses as input features
    neural_features = np.concatenate(
        [dataset['v1_responses'], dataset['v4_responses']], axis=1)
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        neural_features, image_features, test_size=0.2)
    
    # Build a simple decoder model
    decoder = Sequential([
        Dense(1024, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(1024, activation='relu'),
        Dropout(0.3),
        Dense(y_train.shape[1], activation='linear')
    ])
    
    decoder.compile(optimizer=Adam(1e-4), loss='mse')
    
    # Train the model
    history = decoder.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=32
    )
    
    return decoder, history

# Usage example
def example_usage():
    # Create the encoder
    encoder = VisualCortexEncoder(v1_neurons=2000, v4_neurons=500)
    
    # Process a sample image
    sample_image = io.imread('sample_image.jpg')
    responses = encoder.process_image(sample_image)
    
    # Visualize the responses
    encoder.visualize_responses(sample_image)
    
    # Generate a training dataset
    image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg', ...]  # List of image paths
    dataset = encoder.generate_training_dataset(image_paths, save_path='neural_responses.npy')
    
    # Train a decoder model
    decoder_model, training_history = train_decoder_model(encoder, image_paths, epochs=50)
    
    # Save the trained decoder model
    decoder_model.save('visual_decoder_model.h5')