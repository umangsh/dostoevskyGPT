import os


def merge_files(input_directory, output_file):
    # Get a list of all .txt files in the input directory
    txt_files = [f for f in os.listdir(input_directory) if f.endswith('.txt')]

    # Initialize an empty string to store the merged contents
    merged_contents = ''

    # Iterate through the list of files
    for txt_file in txt_files:
        # Construct the full file path
        file_path = os.path.join(input_directory, txt_file)

        # Read the contents of the file and append them to the merged_contents string
        with open(file_path, 'r') as f:
            content = f.read()
            merged_contents += content + '\n\n'

    # Write the merged contents to the output file
    with open(output_file, 'w') as f:
        f.write(merged_contents)


def main():
    # Set input directory and output file paths
    input_directory = '/Users/umang/Desktop/github/dostoevskyGPT/books_txt'
    output_file = '/Users/umang/Desktop/github/dostoevskyGPT/data/dataset.txt'

    # Call the merge_files function
    merge_files(input_directory, output_file)

if __name__ == '__main__':
    main()