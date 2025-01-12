import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load and preprocess images
def load_image(image_path, max_size=400):
    image = Image.open(image_path)
    # Resize the image to the max_size (for faster processing)
    image = image.convert("RGB")
    width, height = image.size
    aspect_ratio = height / width
    new_width = min(width, max_size)
    new_height = int(aspect_ratio * new_width)
    image = image.resize((new_width, new_height))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# De-process the image for visualization
def imshow(tensor, title=None):
    tensor = tensor.squeeze(0)  # Remove batch dimension
    unloader = transforms.ToPILImage()
    image = unloader(tensor.cpu())
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.pause(0.001)

# Load pre-trained VGG19 model
def get_model():
    vgg = models.vgg19(pretrained=True).features
    for param in vgg.parameters():
        param.requires_grad = False  # We don't need to update the model's weights
    return vgg

# Define a function to compute the content loss
def content_loss(content, target):
    return torch.mean((content - target) ** 2)

# Define a function to compute the style loss
def style_loss(style, target):
    # Compute the Gram matrix
    def gram_matrix(x):
        batch_size, h, w, f_map_num = x.size()
        features = x.view(batch_size, f_map_num, h * w)
        G = torch.bmm(features, features.transpose(1, 2))  # Gram matrix
        G = G.div(f_map_num * h * w)
        return G

    A = gram_matrix(style)
    B = gram_matrix(target)
    return torch.mean((A - B) ** 2)

# Main function to run the Neural Style Transfer
def neural_style_transfer(content_image_path, style_image_path, num_steps=500, style_weight=1000000, content_weight=1):
    # Load the content and style images
    content_image = load_image(content_image_path)
    style_image = load_image(style_image_path)

    # Initialize the target image as the content image
    target_image = content_image.clone().requires_grad_(True)

    # Get the model
    model = get_model().to(content_image.device)

    # Define the optimizer (gradient descent)
    optimizer = optim.LBFGS([target_image])

    # Define the layers to use for style and content extraction
    content_layers = ['21']  # Layer 21 of VGG19 corresponds to content features
    style_layers = ['0', '5', '10', '19', '28']  # Layers for style features

    # Extract content and style features from the pre-trained model
    def get_features(image):
        features = []
        x = image
        for i, layer in enumerate(model):
            x = layer(x)
            if str(i) in content_layers:
                features.append(x)
            if str(i) in style_layers:
                features.append(x)
        return features

    content_features = get_features(content_image)
    style_features = get_features(style_image)

    # Optimization loop
    iteration = 0
    while iteration <= num_steps:
        def closure():
            nonlocal iteration
            target_features = get_features(target_image)

            # Compute content loss
            content_loss_val = content_weight * content_loss(content_features[0], target_features[0])

            # Compute style loss
            style_loss_val = 0
            for i in range(len(style_layers)):
                style_loss_val += style_weight * style_loss(style_features[i], target_features[i])

            # Total loss
            total_loss = content_loss_val + style_loss_val

            # Zero the gradients
            optimizer.zero_grad()

            # Backpropagation
            total_loss.backward()

            # Print loss every 50 iterations
            if iteration % 50 == 0:
                print(f"Iteration {iteration}, Total Loss: {total_loss.item()}")

            iteration += 1
            return total_loss

        optimizer.step(closure)

    # Show the final styled image
    imshow(target_image, title="Styled Image")

    return target_image

# Example usage
content_image_path = 'path_to_your_content_image.jpg'
style_image_path = 'path_to_your_style_image.jpg'
styled_image = neural_style_transfer(content_image_path, style_image_path)

