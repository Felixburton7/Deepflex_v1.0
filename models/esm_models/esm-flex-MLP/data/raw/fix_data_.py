import re
import os
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_histidine_variants(input_filename: str, output_filename: str):
    """
    Replaces common non-standard histidine residue names (HSD, HSE, HSP)
    with 'HIS' in a CSV file.

    Args:
        input_filename: Path to the input CSV file.
        output_filename: Path where the modified CSV file will be saved.
    """
    logger.info(f"Reading input file: {input_filename}")
    try:
        with open(input_filename, 'r', encoding='utf-8') as infile:
            content = infile.read()
        logger.info(f"Read {len(content)} characters from the input file.")
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_filename}")
        return
    except Exception as e:
        logger.error(f"Error reading input file {input_filename}: {e}")
        return

    # Define patterns for common histidine variants (case-insensitive matching)
    # Using \b ensures we match whole words/codes
    replacements = {
        r'\bHSD\b': 'HIS',
        r'\bHSE\b': 'HIS',
        r'\bHSP\b': 'HIS',
        # Add more variants if needed, e.g., r'\bHID\b': 'HIS'
    }

    modified_content = content
    replacements_made = 0
    for pattern, replacement in replacements.items():
        modified_content, count = re.subn(pattern, replacement, modified_content, flags=re.IGNORECASE)
        if count > 0:
            logger.info(f"Replaced {count} occurrences of pattern '{pattern}' with '{replacement}'.")
            replacements_made += count

    if replacements_made == 0:
         logger.info("No histidine variants found or replaced.")
    else:
         logger.info(f"Total replacements made: {replacements_made}")

    logger.info(f"Writing modified content to: {output_filename}")
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with open(output_filename, 'w', encoding='utf-8') as outfile:
            outfile.write(modified_content)
        logger.info(f"Conversion complete. Modified file saved as {output_filename}")
    except IOError as e:
        logger.error(f"Error writing output file {output_filename}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error writing output file {output_filename}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Standardize histidine residue names (HSD, HSE, etc.) to HIS in a CSV file.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the modified output CSV file.')
    args = parser.parse_args()

    fix_histidine_variants(args.input, args.output)
