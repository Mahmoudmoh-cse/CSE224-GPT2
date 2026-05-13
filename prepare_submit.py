# Creates a zip file for submission on Gradescope.

import os
import zipfile

OUTPUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
PREDICTION_DIR = os.path.join(OUTPUT_DIR, "predictions")


def collect_required_files():
    files = [p for p in os.listdir('.') if p.endswith('.py')]
    if os.path.isdir(PREDICTION_DIR):
        files += [os.path.join(PREDICTION_DIR, p) for p in os.listdir(PREDICTION_DIR)]
    files += [os.path.join('models', p) for p in os.listdir('models')]
    files += [os.path.join('modules', p) for p in os.listdir('modules')]
    return files

def main():
    aid = 'cs224n_default_final_project_submission'
    zip_path = os.path.join(OUTPUT_DIR, f"{aid}.zip")

    with zipfile.ZipFile(zip_path, 'w') as zz:
        for file in collect_required_files():
            if file.startswith(PREDICTION_DIR):
                arcname = os.path.join("predictions", os.path.basename(file))
            else:
                arcname = os.path.join(".", file)
            zz.write(file, arcname)
    print(f"Submission zip file created: {zip_path}")

if __name__ == '__main__':
    main()
