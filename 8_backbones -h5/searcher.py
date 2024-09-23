import os
print('start')
def find_text_in_py_files(directory,  search_text1, search_text2, search_text3):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".doc"):
                file_path = os.path.join(root, file)
                with open(file_path, encoding='utf-8') as f:
                    content = f.read()
                    if search_text1 in content  and search_text2 in content and search_text3 in content:
                        print("Text found in:", file_path)

# Example usage
directory_path = os.getcwd()
search_text1 = "anatomical guarantees"
search_text2 = " "
search_text3 = " "

find_text_in_py_files(directory_path, search_text1, search_text2, search_text3)