LATEX = xelatex
PDFLATEX = latexmk -xelatex
BIBTEX = bibtex

SOURCES = main.tex sections/*.tex references.bib
MAIN_TEX = main.tex
OUTPUT = main.pdf

all: $(OUTPUT)

$(OUTPUT): $(SOURCES)
	$(PDFLATEX) $(MAIN_TEX)
	$(BIBTEX) main
	$(PDFLATEX) $(MAIN_TEX)
	$(PDFLATEX) $(MAIN_TEX)

clean:
	rm -f *.aux *.log *.out *.pdf *.toc *.fls *.fdb_latexmk *.xdv *.bbl *.blg *.lof *.lot *.synctex.gz
	rm -rf output/

clean-artifacts:
	rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk *.xdv *.bbl *.blg *.lof *.lot *.synctex.gz

clean-all: clean clean-artifacts