docker run --rm -i -t -v "$PWD":/tmp -w /tmp blang/latex:ubuntu pdflatex report
docker run --rm -i -t -v "$PWD":/tmp -w /tmp blang/latex:ubuntu bibtex report
docker run --rm -i -t -v "$PWD":/tmp -w /tmp blang/latex:ubuntu pdflatex report
docker run --rm -i -t -v "$PWD":/tmp -w /tmp blang/latex:ubuntu pdflatex report
