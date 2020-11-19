SHELL := /bin/bash
OUTPUT_DIR = .
BOOKDOWN_FILES = _bookdown_files
PKGNAME = `sed -n "s/Package: *\([^ ]*\)/\1/p" DESCRIPTION`
PKGVERS = `sed -n "s/Version: *\([^ ]*\)/\1/p" DESCRIPTION`
PUBLISH_DIR = ./public
PUBLISH_BOOK_DIR = public
CONDA_ENV = r-torch
CONDA_TYPE = miniconda3
ENV_RECIPE = environment.yml
# Detect operating system. Sort of tricky for Windows because of MSYS, cygwin, MGWIN
ifeq ($(OS), Windows_NT)
    OSFLAG = WINDOWS
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S), Linux)
        OSFLAG = LINUX
    endif
    ifeq ($(UNAME_S), Darwin)
        OSFLAG = OSX
    endif
endif
# conda exists? Works in Linux
ifeq (,$(shell which conda))
    HAS_CONDA=False
else
    HAS_CONDA=True
    # ENV_DIR=$(shell conda info --base)
	CONDA_BASE_DIR=$(shell conda info --base)
    MY_ENV_DIR=$(CONDA_BASE_DIR)/envs/$(CONDA_ENV)
    CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate
endif


# create conda environment
create_condaenv:
	source ${HOME}/${CONDA_TYPE}/etc/profile.d/conda.sh ;\
	conda deactivate
	conda remove --name ${CONDA_ENV} --all -y
	conda env create -f ${ENV_RECIPE}

remove_condaenv:
	source ${HOME}/${CONDA_TYPE}/etc/profile.d/conda.sh ;\
	conda deactivate
	conda remove --name ${CONDA_ENV} --all -y

# activate conda only if environment exists
conda_activate:
ifeq (True,$(HAS_CONDA))
ifneq ("$(wildcard $(MY_ENV_DIR))","") 
	source ${HOME}/${CONDA_TYPE}/etc/profile.d/conda.sh ;\
	conda activate $(CONDA_ENV)
else
	@echo ">>> Detected conda, but $(CONDA_ENV) is missing in $(CONDA_BASE_DIR). Install conda first ..."
endif
else
	@echo ">>> Install conda first."
	exit
endif


.PHONY: gitbook bs4book bs4book_open
gitbook_render:
	Rscript -e "bookdown::render_book(input='.', output_format = 'bookdown::gitbook', config_file='_bookdown.yml')"

bs4book_render:
	export RSTUDIO_PANDOC="/usr/lib/rstudio/bin/pandoc";\
	Rscript -e 'bookdown::render_book("index.Rmd", "bookdown::bs4_book")'


# activate conda environment first; then render; finally, open index.html in a browser
bs4book_open: conda_activate bs4book_render open_book

gitbook_open: conda_activate gitbook_render open_book



open_book:
ifeq ($(OSFLAG), OSX)
    @open -a firefox  $(PUBLISH_BOOK_DIR)/index.html
endif
ifeq ($(OSFLAG), LINUX)
	@firefox  $(PUBLISH_BOOK_DIR)/index.html
endif
ifeq ($(OSFLAG), WINDOWS)
	@"C:\Program Files\Mozilla Firefox\firefox" $(PUBLISH_BOOK_DIR)/index.html
endif



git_push:
	git push ;\
	git subtree push --prefix ${PUBLISH_BOOK_DIR} origin gh-pages	


.PHONY: clean
clean: tidy
		find $(OUTPUT_DIR) -maxdepth 1 -name \*.tex -delete
		find $(FIGURE_DIR) -maxdepth 1 -name \*.png -delete ;\
		$(RM) -rf $(BOOKDOWN_FILES_DIRS)
		if [ -f ${MAIN_RMD} ]; then rm -rf ${MAIN_RMD};fi ;\
		if [ -f ${LIBRARY} ]; then rm ${LIBRARY};fi ;\
		if [ -d ${PUBLISH_BOOK_DIR} ]; then rm -rf ${PUBLISH_BOOK_DIR};fi
		if [ -d ${CHECKPOINTS} ]; then rm -rf ${CHECKPOINTS};fi

# delete unwanted files and folders in bookdown folder
.PHONY: tidy
tidy:
	# delete all markdown files but keep README.md
	find $(OUTPUT_DIR) -maxdepth 1 -name \*.md -not -name 'README.md' -delete
	find $(OUTPUT_DIR) -maxdepth 1 -name \*-book.html -delete
	find $(OUTPUT_DIR) -maxdepth 1 -name \*.png -delete
	find $(OUTPUT_DIR) -maxdepth 1 -name \*.log -delete
	find $(OUTPUT_DIR) -maxdepth 1 -name \*.rds -delete
	find $(OUTPUT_DIR) -maxdepth 1 -name \*.ckpt -delete
	find $(OUTPUT_DIR) -maxdepth 1 -name \*.nb.html -delete
	if [ -d ${BOOKDOWN_FILES} ]; then rm -rf ${BOOKDOWN_FILES};fi \



# provide some essential info about the tikz files
.PHONY: info
info:
	@echo "OS is:" $(OSFLAG)
	@echo "Bookdown publication folder:" $(PUBLISH_BOOK_DIR)
	@echo "Has Conda?:" ${HAS_CONDA}
	@# @echo "Environment:" ${ENV_DIR}
	@echo "Conda Base  Dir:" ${CONDA_BASE_DIR}
	@echo "Environment Dir:" ${MY_ENV_DIR}
	@echo ${CONDA_ENV}