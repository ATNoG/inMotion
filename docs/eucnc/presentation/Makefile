SRC_NAME := presentation

TARGETS := p.pdf p-notes.pdf

# Another template, i am so fucking nerd
DEPS := \
    beamerthemeauriga.sty \
    beamercolorthemeauriga.sty \

LATEX_ENGINE := lualatex

.PHONY: all clean

all: $(TARGETS)

clean:
	rm -rf *.pdf *.aux *.bbl *.blg *.log *.nav *.out *.snm *.toc *.vrb


p.pdf: $(SRC_NAME).md
	pandoc -t beamer \
		-s \
		--slide-level=2 \
		--pdf-engine=$(LATEX_ENGINE) \
		-o $@ $<

p-notes.pdf: $(SRC_NAME).md
	pandoc -t beamer \
		-s \
		--slide-level=2 \
		--pdf-engine=$(LATEX_ENGINE) \
		-V classoption=notes=only \
		-o $@ $<
