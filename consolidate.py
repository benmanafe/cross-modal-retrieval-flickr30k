# import os
# import time

# # --- 1. CONFIGURE THESE SETTINGS ---

# # The path to your repository.
# # Example: r"C:\Users\YourName\Documents\GitHub\my-repo"
# REPO_PATH = r"D:\GitHub\Downloads\cross-modal-retrieval-flickr30k"

# # The name of the folder *inside* your repo that holds the batch folders.
# # Example: "images"
# IMAGES_FOLDER_NAME = "flickr30k_images"

# # --- 2. THE SCRIPT ---

# def consolidate_images():
#     """
#     Moves all images from batch_XXX subfolders up one level
#     and deletes the empty batch folders.
#     """
#     print("--- Starting Image Consolidation ---")
    
#     # This is the parent folder, e.g., "C:\...\my-repo\images"
#     parent_image_dir = os.path.join(REPO_PATH, IMAGES_FOLDER_NAME)

#     if not os.path.isdir(parent_image_dir):
#         print(f"Error: Folder not found at {parent_image_dir}")
#         print("Please check your REPO_PATH and IMAGES_FOLDER_NAME variables.")
#         return

#     # Find all batch folders
#     try:
#         all_dirs = os.listdir(parent_image_dir)
#     except FileNotFoundError:
#         print(f"Error: Path not found {parent_image_dir}. Check your config.")
#         return
        
#     batch_folders = sorted([
#         d for d in all_dirs
#         if os.path.isdir(os.path.join(parent_image_dir, d)) and d.startswith('batch_')
#     ])

#     if not batch_folders:
#         print("No 'batch_XXX' folders found to consolidate.")
#         return

#     print(f"Found {len(batch_folders)} batch folders. Starting move...")
#     total_moved = 0
#     total_conflicts = 0

#     for folder_name in batch_folders:
#         batch_dir_path = os.path.join(parent_image_dir, folder_name)
        
#         try:
#             images_in_batch = os.listdir(batch_dir_path)
#             print(f"Processing {folder_name} ({len(images_in_batch)} images)...")

#             for image_file in images_in_batch:
#                 current_path = os.path.join(batch_dir_path, image_file)
#                 new_path = os.path.join(parent_image_dir, image_file)
                
#                 # Check for file name conflicts
#                 if os.path.exists(new_path):
#                     print(f"  [!] Conflict: {image_file} already exists. Skipping.")
#                     total_conflicts += 1
#                 else:
#                     # Move the file up
#                     os.rename(current_path, new_path)
#                     total_moved += 1
            
#             # After moving all files, delete the now-empty folder
#             os.rmdir(batch_dir_path)
#             print(f"  Finished and removed {folder_name}.")

#         except Exception as e:
#             print(f"Error processing {folder_name}: {e}")

#     print("\n--- Consolidation Complete ---")
#     print(f"Successfully moved: {total_moved} images.")
#     print(f"File name conflicts (skipped): {total_conflicts}")

# if __name__ == "__main__":
#     start_time = time.time()
#     consolidate_images()
#     end_time = time.time()
#     print(f"Total time taken: {end_time - start_time:.2f} seconds.")


import os
import subprocess
import math
import time

# --- 1. CONFIGURE THESE SETTINGS ---

# I've set this to the path from your error message.
REPO_PATH = r"D:\GitHub\Downloads\cross-modal-retrieval-flickr30k" 

# The name of the folder *inside* your repo that holds the 31,000 images.
IMAGES_FOLDER_NAME = "flickr30k_images"

# How many images to commit at a time.
BATCH_SIZE = 1000

# --- 2. THE SCRIPT ---

def run_git_command(command_list, working_dir):
    """Runs a Git command and returns True/False on success/failure."""
    print(f"\nRunning: {' '.join(command_list[:50])}...") # Print only part of long commands
    try:
        result = subprocess.run(
            command_list,
            cwd=working_dir,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8' # Specify encoding
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("--- ERROR ---")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False
    except FileNotFoundError as e:
        print(f"--- ERROR: Command not found ---")
        print(f"{e}")
        return False

def commit_in_batches():
    """Finds all images in the single folder and commits them in batches."""
    print("--- Starting Batch Commit Process ---")
    image_dir = os.path.join(REPO_PATH, IMAGES_FOLDER_NAME)

    if not os.path.isdir(image_dir):
        print(f"Error: Folder not found at {image_dir}")
        return

    # Find all files (not folders) in the image directory
    try:
        all_files = [
            f for f in os.listdir(image_dir) 
            if os.path.isfile(os.path.join(image_dir, f))
        ]
    except FileNotFoundError:
        print(f"Error: Path not found {image_dir}. Check your config.")
        return

    # Filter for common image types
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.psd', '.svg')
    image_files = [
        f for f in all_files 
        if f.lower().endswith(image_extensions)
    ]

    if not image_files:
        print("No image files found in the folder to commit.")
        return

    total_images = len(image_files)
    total_batches = math.ceil(total_images / BATCH_SIZE)
    print(f"Found {total_images} images. Committing in {total_batches} batches.")

    for i in range(total_batches):
        batch_num = i + 1
        print(f"\n--- Processing Batch {batch_num} of {total_batches} ---")
        
        start_index = i * BATCH_SIZE
        end_index = start_index + BATCH_SIZE
        batch_files = image_files[start_index:end_index]

        # Build the 'git add' command with relative paths
        # We need the relative path for Git, using forward slashes
        files_to_add = [f"{IMAGES_FOLDER_NAME}/{file_name}".replace("\\", "/") for file_name in batch_files]
        add_command = ["git", "add"] + files_to_add
        
        # 1. Git Add
        if not run_git_command(add_command, REPO_PATH):
            print(f"Failed to 'git add' batch {batch_num}. Stopping.")
            break

        # 2. Git Commit
        commit_msg = f"Consolidate images batch {batch_num}/{total_batches}"
        if not run_git_command(["git", "commit", "-m", commit_msg], REPO_PATH):
            # This might fail if there are no changes (e.g., already committed)
            print(f"Warning: 'git commit' for batch {batch_num} failed.")
            print("This might be OK if files were already committed. Continuing...")
            continue # Don't stop, just try the next batch

        # 3. Git Push
        if not run_git_command(["git", "push"], REPO_PATH):
            print(f"Failed to 'git push' batch {batch_num}. Stopping.")
            break
            
        print(f"Successfully pushed batch {batch_num}.")
        time.sleep(2) # A small pause
        
    print("--- Batch Commit Complete ---")

if __name__ == "__main__":
    if not os.path.isdir(REPO_PATH):
        print(f"ERROR: The REPO_PATH is not a valid directory.")
        print(f"Please update it to: {REPO_PATH}")
    else:
        # Before we start, let's make sure LFS is tracking the right path
        print("Verifying LFS tracking...")
        run_git_command(["git", "lfs", "track", f"{IMAGES_FOLDER_NAME}/*.jpg"], REPO_PATH)
        run_git_command(["git", "lfs", "track", f"{IMAGES_FOLDER_NAME}/*.png"], REPO_PATH)
        run_git_command(["git", "lfs", "track", f"{IMAGES_FOLDER_NAME}/*.jpeg"], REPO_PATH)
        run_git_command(["git", "add", ".gitattributes"], REPO_PATH)
        run_git_command(["git", "commit", "-m", "Ensure LFS tracking is set for consolidated folder"], REPO_PATH)
        run_git_command(["git", "push"], REPO_PATH)

        # Now run the main batch function
        commit_in_batches()