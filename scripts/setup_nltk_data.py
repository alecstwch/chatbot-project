"""
Download Required NLTK Data

Script to download all required NLTK resources.
Run this once after environment setup.
"""

import nltk
import sys


def download_nltk_resources():
    """Download all required NLTK resources"""
    resources = [
        ('punkt', 'tokenizers'),
        ('punkt_tab', 'tokenizers'),
        ('stopwords', 'corpora'),
        ('wordnet', 'corpora'),
        ('omw-1.4', 'corpora'),
        ('averaged_perceptron_tagger', 'taggers'),
        ('maxent_ne_chunker', 'chunkers'),
        ('words', 'corpora'),
    ]
    
    print("Downloading NLTK resources...")
    print("=" * 60)
    
    failed = []
    
    for resource, category in resources:
        try:
            nltk.data.find(f'{category}/{resource}')
            print(f"[OK] {resource:30} (already downloaded)")
        except LookupError:
            print(f"⬇ {resource:30} (downloading...)")
            try:
                nltk.download(resource, quiet=True)
                print(f"[OK] {resource:30} (downloaded)")
            except Exception as e:
                print(f"[FAIL] {resource:30} (failed: {e})")
                failed.append(resource)
    
    print("=" * 60)
    
    if failed:
        print(f"\n[WARNING] Failed to download: {', '.join(failed)}")
        return False
    else:
        print("\n[DONE] All NLTK resources downloaded successfully!")
        return True


if __name__ == "__main__":
    success = download_nltk_resources()
    sys.exit(0 if success else 1)
