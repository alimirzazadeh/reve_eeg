import re

def clean_summary(text):
    """
    Clean summary text by:
    1. Removing first word if it is 'assistant' or 'minutes'
    2. Removing numbered list markers (1., 2., etc.)
    3. Cleaning up formatting artifacts
    """
    text = text.strip()

    # Remove first word if it is 'assistant' or 'minutes' (case-insensitive)
    words = text.split(None, 1)  # Split into first word and rest
    if words:
        first_word = words[0].lower().rstrip(':,.')
        if first_word in ('assistant', 'minutes'):
            text = words[1] if len(words) > 1 else ''

    # Remove numbered list markers (e.g., "1.", "2.", "10.")
    text = re.sub(r'^\d+\.\s*', '', text)  # At start of text
    text = re.sub(r'\n\d+\.\s*', '\n', text)  # After newlines

    # Clean up formatting artifacts
    text = text.replace('\n', ' ')
    text = text.replace('**', '')
    text = text.replace('--', '')

    # Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text)

    text = text.strip()

    return text

if __name__ == "__main__":
    import os

    summary_dir = '/Users/pengcao/EEG_FM/report_generation/data/claude_summary_instruction_example_1/summaries'

    files = [f for f in os.listdir(summary_dir) if f.endswith('.txt')]
    files.sort()


    for fname in files:
        fpath = os.path.join(summary_dir, fname)
        with open(fpath, 'r') as f:
            original = f.read()

        cleaned = clean_summary(original)

        cleaned_fpath = fpath.replace('.txt.txt', '_cleaned.txt.txt')

        with open(cleaned_fpath, 'w') as f:
            f.write(cleaned)
