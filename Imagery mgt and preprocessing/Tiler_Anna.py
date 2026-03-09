##########
import os
import zipfile
from PIL import Image
import tqdm
from tqdm import tqdm
import glob


def tile_img(image_path, output_folder, tile_size= 1024, overlap= 0, output_widget=None, zip_output=None):
    """
    Crops an image into square tiles of specified size with overlap and saves them to an output folder.

    Parameters:
    - image_path (str): Path to the input image file.
    - tile_size (int, optional): Size of each crop tile (width and height in pixels). Default is 640.
    - overlap (int, optional): Overlap between tiles in pixels. Default is 290.
    - zip_output (bool, optional): Whether to zip the tiled images folder. Default is False.
    - output_widget (widgets.Output, optional): Output widget to display messages.
    - overwrite_tiles (bool, optional): Whether to overwrite the existing tiled images folder. Default is False.

    Returns:
    - None. Saves each tile as a separate image file in the specified output folder.
    """
    # Open the image
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Check if output folder exists and handle overwrite
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    print(base_filename)

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Calculate step size based on desired overlap
    step_size = tile_size - overlap

    # Create a list of all tile coordinates to iterate over
    tile_coordinates = []
    for top in range(0, img_height, step_size):
        for left in range(0, img_width, step_size):
             # Adjust the step size for the last crops on the right and bottom edges
            current_left = left
            current_top = top
            if current_left + tile_size > img_width:
                current_left = img_width - tile_size  # Shift the crop window left to maintain crop size
            if current_top + tile_size > img_height:
                current_top = img_height - tile_size  # Shift the crop window up to maintain crop size
            tile_coordinates.append((current_left, current_top))
    # print("tile coordinates: ", tile_coordinates)
    x = len(tile_coordinates)
    print("number of tile coordinates: ", x)

    with tqdm(total=len(tile_coordinates), desc="Tiling Image") as pbar:
        for left, top in tile_coordinates:
            # Define crop boundaries
            right = left + tile_size
            bottom = top + tile_size
           #print(right)

            # Tile the image
            tiled_img = img.crop((left, top, right, bottom))

            # Save the cropped tile with the original filename included

            tile_filename = f"{base_filename}_[T{top}]_[L{left}].jpg"
           # tile_filename = f"{top}_{left}.jpg"
            tiled_img.save(os.path.join(output_folder, tile_filename))
            pbar.update(1) # Update the progress bar

    if output_widget:
         with output_widget:
             print("Tiling complete.", flush=True) # Explicitly flush output

    if zip_output:
      if output_widget:
          with output_widget:
              print(f"Zipping {output_folder}...", flush=True)
      try:
          with zipfile.ZipFile(f"{output_folder}.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
              for root, _, files in os.walk(output_folder):
                  for file in files:
                      zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(output_folder, '..')))
          if output_widget:
              with output_widget:
                  print(f"Zipped {output_folder} to {output_folder}.zip", flush=True)
      except Exception as e:
          if output_widget:
              with output_widget:
                  print(f"Error zipping folder: {e}", flush=True)

    return output_folder


##
tile_size = 1024
overlap = 0
output_folder= "C:/users/aware/desktop/Seabird_detection/Hive_background_empty_parents/"

source_dir = glob.glob("C:/users/aware/desktop/Seabird_detection/Hive_background_empty_parents/*.jpg")

for images in source_dir:
    image_path = images
    print(image_path)
    tile_img(image_path, output_folder, tile_size, overlap)
