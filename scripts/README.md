## Markdown file merge tool

This will help merge all the *.md files in a tree into 1 and form chapter structures.

Change the folder/book name in the source file [merge_into_book.py](merge_into_book.py):

```python3
# for example:   book_name = "A5-现代软件工程（更新中）"
book_name = "the book folder name"
```

and then run:

```powershell
python3 merge_into_book.py
```

### Dependencies

Please install the below tools and make sure they are added to Path.

- https://pandoc.org/installing.html
- https://miktex.org/download

To check whether these binaries are available, open your terminal and type:

```powershell
pandoc --version
pdflatex --version
miktex --help
```