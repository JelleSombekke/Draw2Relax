import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, shutil
from PIL import Image
from io import BytesIO
import base64

def encode_image_to_base64(img_array):
    pil_img = Image.fromarray(img_array, mode='RGBA')
    buffer = BytesIO()
    pil_img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def calculate_contours(img, base_N, growth_constant, mode=None):

    _, thresh = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY_INV)

    # Find contours on upscaled binary
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    reconstructed_contours = []

    if mode == 'plot':
        plt.figure(figsize=(10, 10))
        plt.imshow(thresh, cmap='gray')
        plt.title('Reconstructed Contours (Fourier Descriptors) with threshold')
        plt.axis('off')

        output_canvas = np.ones_like(img, dtype=np.uint8) * 255
    
        plt.figure(figsize=(30, 30))
        plt.imshow(output_canvas, cmap='gray')
        plt.title('Reconstructed Contours (Fourier Descriptors) with threshold')
        plt.axis('off')


    for contour in contours:
        contour_array = contour.squeeze()

        # Skip degenerate contours
        if contour_array.ndim != 2 or contour_array.shape[0] < 10:
            continue

        # Convert to complex form
        complex_contour = contour_array[:, 0] + 1j * contour_array[:, 1]

        # Compute Fourier Descriptors
        fourier_result = np.fft.fft(complex_contour)

        # Keep only N descriptors based on the size of the contour
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        N = int(base_N * (perimeter / (perimeter + growth_constant)))
        if N < 5:
            continue

        descriptors_filtered = np.zeros_like(fourier_result)
        descriptors_filtered[:N // 2] = fourier_result[:N // 2]
        descriptors_filtered[-N // 2:] = fourier_result[-N // 2:]

        # Reconstruct shape
        reconstructed = np.fft.ifft(descriptors_filtered)
        reconstructed = np.array([reconstructed.real, reconstructed.imag]).T.astype(np.int32)

        reconstructed_contours.append(reconstructed)

        if mode == 'plot':
            plt.plot(reconstructed[:, 0], reconstructed[:, 1], linewidth=1)
        
        elif mode == None:
            continue

    return reconstructed_contours


def draw_contours(reconstructed_contours, img, mode=None):
    # Create a white canvas (background)
    h, w = img.shape
    output_canvas = np.zeros((h, w, 4), dtype=np.uint8)

    for contour in reconstructed_contours:
        # Create masks
        mask_inside = np.zeros_like(img, dtype=np.uint8)
        cv2.drawContours(mask_inside, [contour], -1, 255, thickness=cv2.FILLED)
        mask_dilated = cv2.dilate(mask_inside, np.ones((10, 10), np.uint8), iterations=2)

        # Get outside mask = dilated - original
        mask_outside = cv2.subtract(mask_dilated, mask_inside)

        # Calculate mean intensity inside and outside
        mean_inside = cv2.mean(img, mask=mask_inside)[0]
        mean_outside = cv2.mean(img, mask=mask_outside)[0]

        if mean_inside < mean_outside:
            # Inside is darker → draw black contour (visible)
            color = (0, 0, 0, 255)  # black, opaque
        else:
            # Outside is darker → draw "white" (transparent visually)
            color = (255, 255, 255, 0)  # fully transparent white (invisible)

        # Always draw the contour, but with color depending on logic
        cv2.drawContours(output_canvas, [contour], -1, color, thickness=cv2.FILLED)

    if mode == 'plot':
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(output_canvas, cv2.COLOR_BGRA2RGBA))
        plt.title('Contour overlay')
        plt.axis('off')
        plt.show()

    elif mode == None:
        return output_canvas
    
def compute_displacement_fields(img, circular_structures, scale=1.5, falloff=2.5):
    h, w = img.shape[:2]
    Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    num_shapes = len(circular_structures)
    dx_fields = np.zeros((num_shapes, h, w), dtype=np.float32)
    dy_fields = np.zeros((num_shapes, h, w), dtype=np.float32)
    weights = np.zeros((num_shapes, h, w), dtype=np.float32)

    for idx, (cx, cy, rx, ry) in enumerate(circular_structures):
        dx = X - cx
        dy = Y - cy

        norm_dx = dx / rx
        norm_dy = dy / ry
        dist = np.sqrt(norm_dx**2 + norm_dy**2)

        inner_mask = dist <= 1.0
        outer_mask = (dist > 1.0) & (dist <= falloff)

        disp_x = np.zeros_like(dx, dtype=np.float32)
        disp_y = np.zeros_like(dy, dtype=np.float32)
        influence = np.zeros_like(dx, dtype=np.float32)

        # --- Inner region: scale uniformly inside the ellipse
        if np.any(inner_mask):
            disp_x[inner_mask] = -dx[inner_mask] * (scale - 1.0)
            disp_y[inner_mask] = -dy[inner_mask] * (scale - 1.0)
            influence[inner_mask] = 1.0

        # --- Outer transition zone: push outward with cosine falloff
        if np.any(outer_mask):
            t = (dist[outer_mask] - 1.0) / (falloff - 1.0)  # 0 at edge, 1 at falloff
            falloff_strength = 0.5 * (1 + np.cos(np.pi * t))  # smoothstep: 1→0
            norm = np.maximum(np.sqrt(dx[outer_mask]**2 + dy[outer_mask]**2), 1e-6)

            disp_x[outer_mask] = -dx[outer_mask] / norm * (scale - 1.0) * falloff_strength * rx
            disp_y[outer_mask] = -dy[outer_mask] / norm * (scale - 1.0) * falloff_strength * ry

            influence[outer_mask] = falloff_strength

        dx_fields[idx] = disp_x * influence
        dy_fields[idx] = disp_y * influence
        weights[idx] = influence

    total_weight = np.sum(weights, axis=0)
    dx_combined = np.sum(dx_fields, axis=0)
    dy_combined = np.sum(dy_fields, axis=0)

    dx_final = np.divide(dx_combined, total_weight, out=np.zeros_like(dx_combined), where=total_weight > 0)
    dy_final = np.divide(dy_combined, total_weight, out=np.zeros_like(dy_combined), where=total_weight > 0)

    map_x = (X + dx_final).astype(np.float32)
    map_y = (Y + dy_final).astype(np.float32)

    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    return warped, dx_final, dy_final


def make_animation_frames(img, start_N, end_N, n_iterations, growth_constant, location='Animation_img', mode=None):
    # clear animation img folder
    folder = f'./{location}'
    base64_frames = []
    if mode=='folder':
        os.makedirs(folder, exist_ok=True)
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    # Ease-in curve: slow start, fast end
    x = np.linspace(0, 1, n_iterations)
    curve = 1 - x**(1/5)  # fast falloff at first

    # Scale to range and convert to integers
    descriptor_values = np.round(end_N + (start_N - end_N) * curve).astype(int)
    prev_output_canvas = img

    # Iterate over different descriptor counts
    for i, desc in enumerate(descriptor_values):

        reconstructed_contours = calculate_contours(img, desc, growth_constant, mode=None)

        # Draw contours
        output_canvas = draw_contours(reconstructed_contours, prev_output_canvas, mode=None)
        prev_output_canvas = output_canvas.copy()
        
        if mode == 'plot':
            plt.figure(figsize=(10, 10))
            plt.imshow(output_canvas, cmap='gray')
            plt.axis('off')
            plt.savefig(f'{location}/{i:003}.png', bbox_inches='tight', pad_inches=0)
            plt.close()
        elif mode == 'folder':
            frame_path = os.path.join(folder, f"{i:003}.png")
            cv2.imwrite(frame_path, output_canvas)
        else:
            base64_img = encode_image_to_base64(output_canvas)
            base64_frames.append(base64_img)

    return base64_frames, [os.path.join(folder, f"{i:003}.png") for i in range(n_iterations)]



def make_circ_animation_frames(img, start_N, end_N, n_iterations, growth_constant, circles_structures, location='Animation_circ_img', mode=None):
    # clear animation img folder
    folder = f'./{location}'
    base64_frames = []
    if mode=='folder':
        os.makedirs(folder, exist_ok=True)
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    # Ease-in curve: slow start, fast end
    x = np.linspace(0, 1, n_iterations)
    curve = 1 - x**(1/5)  # fast falloff at first

    curve_y = 1 - x
    scale = (1.5 + (0.25 - 1.5) * curve_y)

    # Scale to range and convert to integers
    descriptor_values = np.round(end_N + (start_N - end_N) * curve).astype(int)

    # Iterate over different descriptor counts
    for i, desc in enumerate(descriptor_values):

        img_warped, _, _ = compute_displacement_fields(img, circles_structures, scale[i], falloff=2.5)

        reconstructed_contours = calculate_contours(img_warped, desc, growth_constant, mode=None)

        # Draw contours
        output_canvas = draw_contours(reconstructed_contours, img_warped, mode=None)

        if mode == 'plot':
            plt.figure(figsize=(10, 10))
            plt.imshow(output_canvas, cmap='gray')
            plt.axis('off')
            plt.savefig(f'{location}/{i:003}.png', bbox_inches='tight', pad_inches=0)
            plt.close()
        elif mode == 'folder':
            frame_path = os.path.join(folder, f"{i:003}.png")
            cv2.imwrite(frame_path, output_canvas)
        else:
            base64_img = encode_image_to_base64(output_canvas)
            base64_frames.append(base64_img)

    return base64_frames, [os.path.join(folder, f"{i:003}.png") for i in range(n_iterations)]

