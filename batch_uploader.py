import os
import subprocess
import math
import time

# --- 1. CONFIGURE THESE SETTINGS ---

# The path to your repository. Use 'r' for raw strings on Windows.
# Example: r"C:\Users\YourName\Documents\GitHub\my-repo"
REPO_PATH = r"D:\GitHub\Downloads\cross-modal-retrieval-flickr30k"

# The name of the folder *inside* your repo that holds the 31,000 images.
# Example: "images" or "assets/photos"
IMAGES_FOLDER_NAME = "flickr30k_images"

# How many images to put in each batch. 1000 is a good start.
BATCH_SIZE = 1000

# --- 2. THE SCRIPT ---

def run_git_command(command_list, working_dir):
    """
    Runs a subprocess command, prints output, and returns True/False.
    """
    print(f"\nRunning: {' '.join(command_list)}")
    try:
        # Run the command
        result = subprocess.run(
            command_list,
            cwd=working_dir,
            check=True,  # This will raise an error if the command fails
            capture_output=True,
            text=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        # Print error details if the command fails
        print("--- ERROR ---")
        print(e.stdout)
        print(e.stderr)
        return False
    except FileNotFoundError as e:
        print(f"--- ERROR ---")
        print(f"Command not found: {e}")
        print("Make sure 'git' is installed and in your system's PATH.")
        return False


def organize_images_into_batches():
    """
    Moves all loose image files in the target folder into new
    'batch_XXX' subfolders.
    """
    print("--- Starting File Organization ---")
    image_dir = os.path.join(REPO_PATH, IMAGES_FOLDER_NAME)
    
    if not os.path.isdir(image_dir):
        print(f"Error: Folder not found at {image_dir}")
        print("Please check your REPO_PATH and IMAGES_FOLDER_NAME variables.")
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
        print("No loose image files found. Already organized?")
        return

    total_images = len(image_files)
    total_batches = math.ceil(total_images / BATCH_SIZE)
    print(f"Found {total_images} images. Creating {total_batches} batches.")

    for i in range(total_batches):
        batch_num = i + 1
        batch_name = f"batch_{batch_num:03d}"  # e.g., "batch_001"
        batch_dir_path = os.path.join(image_dir, batch_name)

        # Create the new batch folder
        os.makedirs(batch_dir_path, exist_ok=True)
        
        # Get the slice of files for this batch
        start_index = i * BATCH_SIZE
        end_index = start_index + BATCH_SIZE
        batch_files = image_files[start_index:end_index]

        # Move each file
        for file_name in batch_files:
            source_path = os.path.join(image_dir, file_name)
            target_path = os.path.join(batch_dir_path, file_name)
            try:
                os.rename(source_path, target_path)
            except Exception as e:
                print(f"Could not move {file_name}: {e}")
        
        print(f"Created {batch_name} and moved {len(batch_files)} images.")
    
    print("--- File Organization Complete ---")


def push_all_batches():
    """
    Finds all 'batch_XXX' folders and adds, commits, and pushes
    them one by one.
    """
    print("--- Starting Batch Push Process ---")
    image_dir = os.path.join(REPO_PATH, IMAGES_FOLDER_NAME)
    
    # Find all batch folders
    try:
        all_dirs = os.listdir(image_dir)
    except FileNotFoundError:
        print(f"Error: Path not found {image_dir}. Check your config.")
        return
        
    batch_folders = sorted([
        d for d in all_dirs
        if os.path.isdir(os.path.join(image_dir, d)) and d.startswith('batch_')
    ])

    if not batch_folders:
        print("No 'batch_XXX' folders found to push.")
        print("Did you run Option 1 to organize the files first?")
        return

    total_batches = len(batch_folders)
    print(f"Found {total_batches} batch folders to push.")
    
    for i, folder_name in enumerate(batch_folders):
        print(f"\n--- Processing Batch {i+1} of {total_batches}: {folder_name} ---")
        
        # We need the relative path for Git, using forward slashes
        folder_to_add = f"{IMAGES_FOLDER_NAME}/{folder_name}/"
        commit_msg = f"Add image {folder_name}"

        # 1. Git Add
        if not run_git_command(["git", "add", folder_to_add], REPO_PATH):
            print(f"Failed to 'git add' {folder_name}. Stopping script.")
            break
            
        # 2. Git Commit
        if not run_git_command(["git", "commit", "-m", commit_msg], REPO_PATH):
            print(f"Failed to 'git commit' {folder_name}. Stopping script.")
            print("This might mean the batch was empty or already committed.")
            # We don't break here, as it might just be an empty commit
            continue

        # 3. Git Push
        if not run_git_command(["git", "push"], REPO_PATH):
            print(f"Failed to 'git push' {folder_name}. Stopping script.")
            print("Please check your internet connection and LFS status.")
            print("You can re-run this script later to resume.")
            break
            
        print(f"Successfully pushed {folder_name}.")
        time.sleep(2) # A small pause
        
    print("--- Batch Push Complete ---")


def main_menu():
    """
    Shows the main menu to the user.
    """
    print("\n" + "="*40)
    print("  GitHub Batch Uploader")
    print("="*40)
    print(f"Repo Path: {REPO_PATH}")
    print(f"Image Folder: {IMAGES_FOLDER_NAME}")
    print("="*40)
    print("What do you want to do?")
    print("\n[1] Organize my 31,000 loose images into batch folders")
    print("    (Run this FIRST)")
    
    print("\n[2] Push all existing batch folders to GitHub")
    print("    (Run this SECOND, after running [1])")
    
    print("\n[q] Quit")
    
    return input("\nEnter your choice (1, 2, or q): ")

if __name__ == "__main__":
    # Check if Git is installed
    if not run_git_command(["git", "--version"], REPO_PATH):
        print("Git not found. Please install Git and try again.")
    else:
        while True:
            choice = main_menu()
            if choice == '1':
                organize_images_into_batches()
            elif choice == '2':
                push_all_batches()
            elif choice.lower() == 'q':
                print("Exiting.")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or q.")