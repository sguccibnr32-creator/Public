# =============================================================================
# Makefile -- membrane_v48 companion paper
#   Compile: lualatex + bibtex + lualatex x2
# =============================================================================

MAIN     = membrane_v48
LATEX    = lualatex -interaction=nonstopmode -halt-on-error
BIBTEX   = bibtex

.PHONY: all clean distclean arxiv

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN) || true
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

# Quick rebuild (no bibtex)
quick:
	$(LATEX) $(MAIN)

# Clean intermediate files
clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).bbl $(MAIN).blg \
	      $(MAIN).toc $(MAIN).out $(MAIN).lof $(MAIN).lot \
	      $(MAIN).synctex.gz

# Clean everything including PDF
distclean: clean
	rm -f $(MAIN).pdf

# Prepare arXiv submission tarball
# Includes: .tex, .bbl (pre-processed), .bib (as backup)
arxiv: $(MAIN).pdf
	mkdir -p arxiv_submission
	cp $(MAIN).tex arxiv_submission/
	cp $(MAIN).bbl arxiv_submission/
	cp references.bib arxiv_submission/
	cp README_tex.md arxiv_submission/README.md 2>/dev/null || true
	cd arxiv_submission && tar czf ../$(MAIN)_arxiv.tar.gz *
	@echo ""
	@echo "arXiv submission package: $(MAIN)_arxiv.tar.gz"
	@ls -la $(MAIN)_arxiv.tar.gz
