# Makefile for membrane_v48_en (v4.8 companion, English version)
# Uses pdflatex (no Japanese content)

MAIN   = membrane_v48_en
LATEX  = pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX = bibtex

.PHONY: all clean distclean arxiv

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex refs_v48_en.bib
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN) || true
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

quick:
	$(LATEX) $(MAIN)

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).bbl $(MAIN).blg \
	      $(MAIN).toc $(MAIN).out $(MAIN).lof $(MAIN).lot \
	      $(MAIN).synctex.gz

distclean: clean
	rm -f $(MAIN).pdf

arxiv: $(MAIN).pdf
	mkdir -p arxiv_submission
	cp $(MAIN).tex arxiv_submission/
	cp $(MAIN).bbl arxiv_submission/
	cp refs_v48_en.bib arxiv_submission/
	cp README_v48_en_tex.md arxiv_submission/README.md 2>/dev/null || true
	cd arxiv_submission && tar czf ../$(MAIN)_arxiv.tar.gz *
	@echo ""
	@echo "arXiv submission package: $(MAIN)_arxiv.tar.gz"
	@ls -la $(MAIN)_arxiv.tar.gz
