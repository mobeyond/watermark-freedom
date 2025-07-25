import os
import subprocess

def main():
    tin_dir = './tin'
    watermarked_dir = './tin_watermarked'
    
    # Ensure watermarked directory exists
    os.makedirs(watermarked_dir, exist_ok=True)
    
    # Process each image in the tin directory
    for filename in os.listdir(tin_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(tin_dir, filename)
            output_path = os.path.join(watermarked_dir, filename)
            
            try:
                # Apply watermark with custom parameters for gauguin_256.jpg and seabackground
                print(f"Watermarking {filename}...")
                if filename == 'gauguin_256.jpg':
                    # Custom mask parameters for this image
                    subprocess.run(['python', 'mark.py', '--cover', input_path, '--output', output_path, '--use_corners', '--x', '50', '--y', '50', '--width', '250', '--height', '250'], check=True)
                elif filename == 'cropped_seabackground.jpg':
                    # Custom mask parameters for this image
                    subprocess.run(['python', 'mark.py', '--cover', input_path, '--output', output_path, '--use_corners', '--x', '75', '--y', '75', '--width', '150', '--height', '150'], check=True)
                else:
                    subprocess.run(['python', 'mark.py', '--cover', input_path, '--output', output_path, '--use_corners'], check=True)
                
                # Verify watermark
                print(f"Verifying {filename}...")
                result = subprocess.run(
                    ['python', 'verify.py', '--watermarked', output_path],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                # Output verification results
                print(f"Verification results for {filename}:")
                print(result.stdout)
                print("-" * 40)
                
            except subprocess.CalledProcessError as e:
                print(f"Error processing {filename}:")
                print(e.stderr)
                print("-" * 40)

if __name__ == "__main__":
    main()
