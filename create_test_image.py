#!/usr/bin/env python3
"""
Create test images with custom dimensions for upscaling tests
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_test_image():
    """Create a test image with non-square custom dimensions (not divisible by 8)"""
    
    # Custom dimensions that are not perfect squares and not easily divisible by 8
    width = 423  # Not divisible by 8, not square
    height = 317  # Not divisible by 8, different from width
    
    # Create image with gradient background
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Draw gradient background
    for y in range(height):
        red = int(255 * y / height)
        color = (red, 100, 255 - red)
        draw.line([(0, y), (width, y)], fill=color)
    
    # Add some geometric shapes for upscaling test
    # Draw circles
    draw.ellipse([50, 50, 150, 150], fill='red', outline='black', width=2)
    draw.ellipse([width-150, height-150, width-50, height-50], fill='blue', outline='white', width=3)
    
    # Draw rectangles
    draw.rectangle([width//2-50, height//2-30, width//2+50, height//2+30], fill='green', outline='yellow', width=2)
    
    # Add text
    try:
        # Try to use a better font if available
        font = ImageFont.truetype("Arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    text = "UPSCALE TEST"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = (width - text_width) // 2
    draw.text((text_x, 20), text, fill='black', font=font)
    
    # Add dimension info
    dim_text = f"{width}x{height}"
    bbox = draw.textbbox((0, 0), dim_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = (width - text_width) // 2
    draw.text((text_x, height - 40), dim_text, fill='white', font=font)
    
    # Save the test image
    output_path = 'test-assets/test-upscale.jpg'
    os.makedirs('test-assets', exist_ok=True)
    image.save(output_path, 'JPEG', quality=85)
    
    print(f"Created test image: {output_path}")
    print(f"Dimensions: {width}x{height} (aspect ratio: {width/height:.3f})")
    print(f"Not divisible by 8: width%8={width%8}, height%8={height%8}")
    
    return output_path

if __name__ == "__main__":
    create_test_image()