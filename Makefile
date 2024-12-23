# Makefile
LATEX = xelatex
PDFLATEX = latexmk -xelatex

SOURCES = main.tex sections/*.tex
MAIN_TEX = main.tex
OUTPUT = main.pdf

all: $(OUTPUT)

$(OUTPUT): $(SOURCES)
	$(PDFLATEX) $(MAIN_TEX)

clean:
	rm -f *.aux *.log *.out *.pdf *.toc *.fls *.fdb_latexmk *.xdv
	rm -rf output/