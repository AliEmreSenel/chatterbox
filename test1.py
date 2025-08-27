import argparse


def filter_lines(input_file, output_file):
    # Create full extended ASCII character set (0-255)
    extended_ascii = set(chr(i) for i in range(256))

    # Turkish-specific characters that might not be in extended ASCII
    turkish_chars = "çğıöşüİĞÜŞÖÇ"

    # Combine extended ASCII with Turkish-specific characters
    allowed_chars = extended_ascii.union(set(turkish_chars))

    with (
        open(input_file, "r", encoding="utf-8") as infile,
        open(output_file, "w", encoding="utf-8") as outfile,
    ):
        for line in infile:
            # Check if all characters in the line are allowed
            if all(char in allowed_chars for char in line):
                outfile.write(line)


def main():
    parser = argparse.ArgumentParser(
        description="Filter lines based on extended ASCII and Turkish alphabet characters."
    )
    parser.add_argument("input_file", help="Path to the input file")
    parser.add_argument("output_file", help="Path to the output file")

    args = parser.parse_args()

    filter_lines(args.input_file, args.output_file)
    print(f"Filtering complete. Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
