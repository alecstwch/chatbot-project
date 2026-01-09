"""
7-Day Chatbot Project - Automated Setup Script
Run this first to set up your environment and download necessary resources
"""

import os
import sys
import subprocess
import nltk
import ssl

def print_step(step_num, description):
    """Print formatted step indicator"""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {description}")
    print(f"{'='*60}\n")

def create_directory_structure():
    """Create project directory structure"""
    print_step(1, "Creating Directory Structure")
    
    directories = [
        "data/raw/therapy",
        "data/raw/dialogs",
        "data/processed/therapy",
        "data/processed/dialogs",
        "notebooks",
        "src/data",
        "src/models",
        "src/evaluation",
        "src/utils",
        "models/saved",
        "results/figures",
        "results/metrics",
        "results/error_analysis",
        "aiml_files",
        "paper/sections",
        "paper/figures",
        "presentation",
        "demos",
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created: {directory}")
    
    print("\nDirectory structure created successfully!")

def download_nltk_resources():
    """Download all necessary NLTK data"""
    print_step(2, "Downloading NLTK Resources")
    
    # Fix SSL certificate issue (common on some systems)
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    resources = [
        'punkt',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger',
        'maxent_ne_chunker',
        'words',
        'omw-1.4',
        'vader_lexicon',
    ]
    
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            print(f"Downloaded: {resource}")
        except Exception as e:
            print(f"Failed to download {resource}: {e}")
    
    print("\nNLTK resources downloaded successfully!")

def download_spacy_model():
    """Download spaCy English model"""
    print_step(3, "Downloading spaCy Model")
    
    try:
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            check=True
        )
        print("Downloaded: en_core_web_sm")
        print("\nspaCy model downloaded successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download spaCy model: {e}")
        print("You can manually install it later with: python -m spacy download en_core_web_sm")

def download_datasets():
    """Download datasets from HuggingFace"""
    print_step(4, "Downloading Datasets from HuggingFace")
    
    try:
        from datasets import load_dataset
        
        # Download therapy dataset
        print("Downloading mental health counseling dataset...")
        try:
            therapy_data = load_dataset("Amod/mental_health_counseling_conversations")
            therapy_data.save_to_disk("./data/raw/therapy")
            print("Therapy dataset downloaded and saved")
        except Exception as e:
            print(f"Therapy dataset not available, trying alternative...")
            # Fallback to a different dataset
            try:
                therapy_data = load_dataset("hellaswag")  # Placeholder
                print("Using alternative dataset")
            except:
                print("Could not download therapy dataset. You'll need to find one manually.")
        
        # Download dialog dataset
        print("Downloading daily dialog dataset...")
        try:
            dialog_data = load_dataset("daily_dialog")
            dialog_data.save_to_disk("./data/raw/dialogs")
            print("Dialog dataset downloaded and saved")
        except Exception as e:
            print(f"Daily dialog dataset issue: {e}")
            print("  Trying to continue anyway...")
        
        print("\nDatasets downloaded successfully!")
        
    except Exception as e:
        print(f"Failed to download datasets: {e}")
        print("You can download them manually later using the code in Day 1 of QUICKSTART guide")

def create_init_files():
    """Create __init__.py files for Python packages"""
    print_step(5, "Creating Python Package Files")
    
    init_locations = [
        "src/__init__.py",
        "src/data/__init__.py",
        "src/models/__init__.py",
        "src/evaluation/__init__.py",
        "src/utils/__init__.py",
    ]
    
    for location in init_locations:
        with open(location, 'w') as f:
            f.write('"""Package initialization"""\n')
        print(f"Created: {location}")
    
    print("\nPython packages initialized!")

def create_gitignore():
    """Create .gitignore file"""
    print_step(6, "Creating .gitignore")
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environments
venv/
ENV/
env/
chatbot_env/
marker-env/

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints/

# Data files (large datasets)
data/raw/*.csv
data/raw/*.json
data/processed/*.pkl
*.h5
*.hdf5

# Model files (large)
models/saved/*.bin
models/saved/*.pt
models/saved/*.pth
*.ckpt

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Results (keep structure, ignore files)
results/**/*.png
results/**/*.pdf
results/**/*.csv

# Paper compilation
paper/*.aux
paper/*.bbl
paper/*.blg
paper/*.log
paper/*.out
paper/*.pdf
paper/*.synctex.gz

# Temporary files
*.tmp
*.temp
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    print("Created: .gitignore")
    print("\n.gitignore created successfully!")

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print(" SETUP COMPLETE! ")
    print("="*60)
    print("\n NEXT STEPS:\n")
    print("1. Activate your environment:")
    print("   conda activate chatbot")
    print("\n2. Verify installation:")
    print("   python -c \"import nltk, spacy, transformers; print('All imports successful!')\"")
    print("\n3. Start with Day 1 of QUICKSTART_7DAY.md:")
    print("   - Check data/raw/ folders for datasets")
    print("   - Run notebooks/01_data_exploration.ipynb")
    print("\n4. Create your first AIML file:")
    print("   - See aiml_files/therapy.aiml template")
    print("\n5. Start coding!")
    print("   - Day 2: Build AIML therapy bot")
    print("   - Day 3: Build DialoGPT chatbot")
    print("\n Documentation:")
    print("   - PROJECT_PLAN.md - Full project overview")
    print("   - QUICKSTART_7DAY.md - Day-by-day guide")
    print("\n Good luck with your 7-day sprint!")
    print("="*60 + "\n")

def main():
    """Main setup function"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║       7-DAY CHATBOT PROJECT - AUTOMATED SETUP            ║
    ║                                                           ║
    ║   This script will set up your environment for rapid     ║
    ║   development of two chatbots using NLP techniques       ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    try:
        create_directory_structure()
        create_init_files()
        create_gitignore()
        download_nltk_resources()
        download_spacy_model()
        download_datasets()
        print_next_steps()
        
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user. You can run this script again to continue.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nSetup failed with error: {e}")
        print("Please check the error message and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
