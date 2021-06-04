from pathlib import Path
from memory import write_table

TEMPLATE_PATH = "../README-TEMPLATE.md"
README_PATH = "../README.md"

TABLE_PLACEHOLDER = "%memory_table"


def create_readme():
    with Path(TEMPLATE_PATH).open("r") as fp_template, Path(README_PATH).open("w") as fp_readme:
        text = fp_template.read()
        print(write_table(140))
        text = text.replace(TABLE_PLACEHOLDER, write_table(140))
        fp_readme.write(text)


if __name__ == '__main__':
    create_readme()
