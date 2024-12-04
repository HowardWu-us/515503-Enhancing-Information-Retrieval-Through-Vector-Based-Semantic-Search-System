# Makefile
LATEX = xelatex
PDFLATEX = latexmk -xelatex

SOURCES = main.tex
OUTPUT = main.pdf

all: $(OUTPUT)

$(OUTPUT): $(SOURCES)
	$(PDFLATEX) $(SOURCES)

clean:
	rm -f *.aux *.log *.out *.pdf *.toc